from gurobipy import GRB,Model,quicksum
import itertools
import time
import pandas as pd
import math
import os
from collections import defaultdict

class ScenarioTree:
    def __init__(self, technologyTrees):
        self.technologies = technologyTrees
        self.numStages = self.technologies[0].nodes[-1].stage
        self.numSubperiods = self.technologies[0].nodes[-1].numSubperiods
        self.numSubterms = self.technologies[0].numSubterms
        self.nodes = []

        prerootTechNodeList = []
        for tech in self.technologies:
            prerootTechNodeList.append(tech.nodes[0])
        self.preroot = ScenarioNode(0, None, 1, self, prerootTechNodeList)

        rootTechNodeList = []
        for tech in self.technologies:
            rootTechNodeList.append(tech.nodes[1])
        self.root = ScenarioNode(1, self.preroot, 1, self, rootTechNodeList)
        self.preroot.children.append(self.root)

        leaves = [self.root]
        for n in range(1,self.numStages):
            nextLeaves = []
            for leaf in leaves:
                tempList = []
                for techNode in leaf.techNodeList:
                    tempList.append(techNode.children)
                tempProduct = itertools.product(*tempList)
                for element in tempProduct:
                    leaf.AddChild(element)
                for child in leaf.children:
                    nextLeaves.append(child)
            leaves = []
            for leaf in nextLeaves:
                leaves.append(leaf)

class ScenarioNode:
    def __init__(self, id_In, parent_In, probability_In, tree_In, techNodeList_In):
        self.id = id_In
        self.parent = parent_In
        self.tree = tree_In

        if self.parent is None:
            self.stage = 0
            self.numSubperiods = 1
            self.stageSubperiods = [0]
            self.allSubperiods = [0]
        else:
            self.stage = self.parent.stage + 1
            self.numSubperiods = self.tree.numSubperiods
            self.stageSubperiods = [1 + (self.stage-1) * self.numSubperiods + t for t in range(self.numSubperiods)]
            self.allSubperiods = [0] + [1 + (s-1) * self.numSubperiods + t for s in range(1,self.stage+1) for t in range(self.numSubperiods)]

        self.numSubterms = self.tree.numSubterms
        self.stageSubterms = [p for p in range(1, self.numSubterms+1)]
        self.probability = probability_In
        self.techNodeList = techNodeList_In
        self.productiontechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'production']
        self.storagetechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == 'storage']

        self.tech_types = []
        for tech in self.techNodeList:
            self.tech_types.append(tech.tree.type)

        self.tree.nodes.append(self)
        self.children = []

    def AddChild(self, techNodeList):
        prob = 1
        for techNode in techNodeList:
            prob *= techNode.probability
        child = ScenarioNode(len(self.tree.nodes), self, prob, self.tree, techNodeList)
        self.children.append(child)

    def FindAncestorFromDiff(self, t, t_):
        ancestor = self
        amount_subperiods = len(ancestor.stageSubperiods) 
        node_no_1 = (t-1) // amount_subperiods
        node_no_2 = (t_-1) // amount_subperiods
        how_many_more_ancestors = node_no_2 - node_no_1
        for _ in range(how_many_more_ancestors):
            if ancestor.parent:
                ancestor = ancestor.parent
        return ancestor

    def AddVariables(self, model):
        self.v_Plus = {}
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.productiontechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], vtype=GRB.INTEGER, name="plus_"+str(self.id))) # purchase
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.storagetechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        self.v_Minus = {}
        self.v_Minus.update(model.addVars([(tech.tree.type, v, t, t_)  for tech in self.productiontechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.INTEGER, name="minus_"+str(self.id))) # salvage
        self.v_Minus.update(model.addVars([(tech.tree.type, v, t, t_)  for tech in self.storagetechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, name="minus_"+str(self.id))) # salvage
        self.v_Existing = model.addVars([(tech.tree.type, v, t, t_) for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, name="exist_"+str(self.id)) # exist
        self.g_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="grid_purchase_"+str(self.id)) # grid purchase amount
        self.i_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="carry_"+str(self.id)) # inventory carriage amount
        self.i_Charging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="charge_"+str(self.id)) # inventory charging amount
        self.i_Discharging = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="discharge_"+str(self.id)) # inventory discharging amount

    def AddObjectiveCoefficients(self, model, grid_electricity_cost, discount_factor):
        for i, tech in enumerate(self.techNodeList):
            for v in range(tech.NumVersion):
                for t_ in self.stageSubperiods:
                    self.v_Plus[tech.tree.type,v,t_].obj = self.probability * tech.cost[v] * (discount_factor**(t_))
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            self.v_Minus[tech.tree.type,v,t,t_].obj = -self.probability * tech.salvage_value[v] * (1 - (tech.depreciation_rate[v] * (t_ - t))) * (discount_factor**(t_))
                            self.v_Existing[tech.tree.type,v,t,t_].obj = self.probability * self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcost[v] * ((self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcostchangebyage[v])**(t_ - t)) * ((self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcostchangebyyear[v])**(t)) * (discount_factor**(t_))

        for t in self.stageSubperiods:
            for p in self.stageSubterms:
                self.g_Purchase[t,p].obj = self.probability * grid_electricity_cost[t] * (discount_factor**(t))

    def AddTechnologyBalanceConstraints(self, model):
        for tech in self.techNodeList:
            for v in range(tech.NumVersion):
                for t_ in self.stageSubperiods:
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            model.addConstr(self.v_Existing[tech.tree.type,v,t,t_] == self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]
                                - quicksum(self.FindAncestorFromDiff(t__, t_).v_Minus[tech.tree.type,v,t,t__] for t__ in range(t,t_+1)), name = f'N{self.id}_Balance_{tech.tree.type}_{v}_{t}_{t_}')

    def AddDemandConstraints(self, model, demand):
        if self.id != 0:
            for t_ in self.stageSubperiods:
                for p, periodic_demand in enumerate(demand[t_]):
                    model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p]*self.v_Existing[tech.tree.type,v,t,t_]*(1 - (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.productiontechNodeList) for v in range(self.productiontechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].lifetime[v]) + self.g_Purchase[t_, p+1] - self.i_Charging[t_, p+1] + self.i_Discharging[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')

    def AddInventoryBalanceConstraints(self, model): #Global charging/discharging efficiencies are used.
        if self.id != 0:
            for t_ in self.stageSubperiods:
                for p in self.stageSubterms:
                    if p == 1:
                        model.addConstr(self.i_Carrying[t_,p] == self.FindAncestorFromDiff(t_-1,t_).i_Carrying[t_-1, self.numSubterms] + self.storagetechNodeList[0].storage_charge_efficiency[0] * self.i_Charging[t_,p] - (1 / self.storagetechNodeList[0].storage_discharge_efficiency[0]) * self.i_Discharging[t_,p], name = f'N{self.id}_InventoryBalance_{t_}_{p}')
                    else:
                        model.addConstr(self.i_Carrying[t_,p] == self.i_Carrying[t_,p-1] + self.storagetechNodeList[0].storage_charge_efficiency[0] * self.i_Charging[t_,p] - (1 / self.storagetechNodeList[0].storage_discharge_efficiency[0]) * self.i_Discharging[t_,p], name = f'N{self.id}_InventoryBalance_{t_}_{p}')

    def AddBatteryCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for p in self.stageSubterms:
                model.addConstr(self.i_Carrying[t_,p] <= quicksum(self.v_Existing[tech.tree.type,v,t,t_]*self.FindAncestorFromDiff(t,t_).storagetechNodeList[i].storage_capacity[v]*(1 - (self.FindAncestorFromDiff(t,t_).storagetechNodeList[i].degradation_rate[v] * (t_ - t))) for i, tech in enumerate(self.storagetechNodeList) for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t + self.FindAncestorFromDiff(t,t_).storagetechNodeList[i].lifetime[v]), name = f'N{self.id}_BatteryCapacity_{t_}_{p}')

    def AddEmissionConstraints(self, model, emission_limits):
        for t in self.stageSubperiods:
            if emission_limits[t] != None:
                model.addConstr(quicksum(self.g_Purchase[t,p] for p in self.stageSubterms) <= emission_limits[t], name = f'N{self.id}_Emission_{t}')

    def AddBudgetConstraints(self, model, budget):
        for t in self.stageSubperiods:
            if budget[t] != None:
                model.addConstr(quicksum(tech.cost[v] * self.v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion)) <= budget[t], name = f'N{self.id}_Budget_{t}')

    def AddSpatialConstraints(self, model, spatial_limit):
        for t_ in self.stageSubperiods:
            if spatial_limit != None:
                model.addConstr(quicksum(tech.spatial_requirement[v] * self.v_Existing[tech.tree.type,v,t,t_] for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t+tech.lifetime[v]) <= spatial_limit, name = f'N{self.id}_Spatial_{t_}')

    def InitializeCurrentTech(self, initial_tech):
        if self.parent == None:
            for i, tech in enumerate(self.techNodeList):
                for v in range(tech.NumVersion):
                    if initial_tech[i][v] != 0:
                        self.v_Plus[tech.tree.type, v, 0].lb = initial_tech[i][v]
                    else:
                        self.v_Plus[tech.tree.type, v, 0].ub = 0

    def AddUpperBoundsForIP(self, model, demand):
        for i, tech in enumerate(self.productiontechNodeList):
            for v in range(tech.NumVersion):
                for t in self.stageSubperiods:
                    ub_v = math.ceil(max([(0 if self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p] == 0 else (periodic_demand / (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p] * (1 - (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].degradation_rate[v] * (t_ - t)))))) for t_ in self.allSubperiods for p, periodic_demand in enumerate(demand[t_]) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].lifetime[v]]))
                    model.addConstr(self.v_Plus[tech.tree.type,v,t] <= ub_v, name = f'N{self.id}_UpperBound_v_plus_{tech.tree.type}_{v}_{t}')
                    for t__ in self.allSubperiods:
                        if t__ <= t < t__ + self.FindAncestorFromDiff(t,t__).productiontechNodeList[i].lifetime[v]:
                            model.addConstr(self.v_Minus[tech.tree.type,v,t__,t] <= ub_v, name = f'N{self.id}_UpperBound_v_minus_{tech.tree.type}_{v}_{t__}_{t}')

class TechnologyTree:
    def __init__(self, type, segment, numSubperiods, numSubterms, lifetime, initialCost, initialEmission, degradation_rate, initialOMcost, OMcostchangebyage, depreciation_rate, initial_salvage_value, OMcostchangebyyear, spatial_requirement, periodic_electricity_production=None, electricity_storage_capacity=None, storage_charging_efficiency=None, storage_discharging_efficiency=None):
        self.type = type
        self.initialCost = initialCost
        self.initialEmission = initialEmission
        self.periodic_electricity_production = periodic_electricity_production
        self.electricity_storage_capacity = electricity_storage_capacity
        self.storage_charging_efficiency = storage_charging_efficiency
        self.storage_discharging_efficiency = storage_discharging_efficiency
        self.nodes = []
        self.degradation_rate = degradation_rate
        self.initialOMcost = initialOMcost
        self.OMcostchangebyage = OMcostchangebyage
        self.depreciation_rate = depreciation_rate
        self.initial_salvage_value = initial_salvage_value
        self.OMcostchangebyyear = OMcostchangebyyear
        self.lifetime = lifetime
        self.segment = segment
        self.spatial_requirement = spatial_requirement
        self.numSubperiods = numSubperiods
        self.numSubterms = numSubterms
        self.versions = len(initialCost)
        self.preroot = TechnologyNode(0, None, 1, self, self.versions, [0 for _ in range(self.versions)], [0 for _ in range(self.versions)], self.periodic_electricity_production, self.electricity_storage_capacity, self.lifetime, self.degradation_rate, self.initialOMcost, self.OMcostchangebyage, self.depreciation_rate, self.initial_salvage_value, self.OMcostchangebyyear, self.spatial_requirement, self.storage_charging_efficiency, self.storage_discharging_efficiency) # preroot is stage-0: the existing situation
        self.root = TechnologyNode(1, self.preroot, 1, self, self.versions, self.initialCost, self.initialEmission, self.periodic_electricity_production, self.electricity_storage_capacity, self.lifetime, self.degradation_rate, self.initialOMcost, self.OMcostchangebyage, self.depreciation_rate, self.initial_salvage_value, self.OMcostchangebyyear, self.spatial_requirement, self.storage_charging_efficiency, self.storage_discharging_efficiency) #root is the inital decision making node.

    def ConstructByMultipliers(self, numStages, probabilities, costMultiplier, efficiencyMultiplier, emissionMultiplier):
        leaves = [self.root]
        numBranches = len(probabilities)
        for n in range(1,numStages):
            nextLeaves = []
            for leaf in leaves:
                for b in range(numBranches):
                    leaf.AddChild(probabilities[b], costMultiplier[b], efficiencyMultiplier[b], emissionMultiplier[b])
                for child in leaf.children:
                    nextLeaves.append(child)
            leaves = []
            for leaf in nextLeaves:
                leaves.append(leaf)

class TechnologyNode:
    def __init__(self, id_In, parent_In, probability_In, tree_In, versionnum_In, cost_In, emission_In, periodic_electricity_In, storage_capacity_In, lifetime_In, degradation_In, OMcost_In, OMcostchangebyage_In, depreciation_In, salvage_value_In, OMcostchangebyyear_In, spatial_requirement_In, storage_charge_efficiency_In, storage_discharge_efficiency_In):
        self.id = id_In
        self.parent = parent_In

        self.tree = tree_In
        self.tree.nodes.append(self)
        self.children = []

        if self.parent is None:
            self.stage = 0
            self.probability = 1
            self.numSubperiods = 1
        else:
            self.stage = self.parent.stage + 1
            self.probability = self.parent.probability * probability_In
            self.numSubperiods = self.tree.numSubperiods

        self.NumVersion = versionnum_In
        self.cost = cost_In
        self.lifetime = lifetime_In
        self.emission = emission_In
        self.periodic_electricity = periodic_electricity_In
        self.storage_capacity = storage_capacity_In
        self.degradation_rate = degradation_In
        self.OMcost = OMcost_In
        self.OMcostchangebyage = OMcostchangebyage_In
        self.OMcostchangebyyear = OMcostchangebyyear_In
        self.depreciation_rate = depreciation_In
        self.salvage_value = salvage_value_In
        self.spatial_requirement = spatial_requirement_In
        self.storage_charge_efficiency = storage_charge_efficiency_In
        self.storage_discharge_efficiency = storage_discharge_efficiency_In

    def AddChild(self, prob, costMult, effMult, emisMult):
        child = TechnologyNode(len(self.tree.nodes), self, prob, self.tree, self.NumVersion, [i*costMult for i in self.cost], [i*emisMult for i in self.emission], ([[x*effMult for x in i] for i in self.periodic_electricity] if self.periodic_electricity is not None else None), ([i*effMult for i in self.storage_capacity] if self.storage_capacity is not None else None), self.lifetime, self.degradation_rate, self.OMcost, self.OMcostchangebyage, self.depreciation_rate, [i*costMult for i in self.salvage_value], self.OMcostchangebyyear, self.spatial_requirement, self.storage_charge_efficiency, self.storage_discharge_efficiency)
        self.children.append(child)

def Output(m):
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'} 
    status = m.status

    print('The optimization status is ' + status_code[status])
    if status == 2:    
        print('Optimal solution:')
        for v in m.getVars():
            if v.x > 0:
                print(str(v.varName) + " = " + str(v.x))    
        print('Optimal objective value: ' + str(m.objVal) + "\n")

def OutputProductionResults(model, scenarioTree, discount_factor, demand, numStages, numSubperiods, numSubterms, numMultipliers, results_directory, grid_electricity_cost, tolerance=1e-6):
    grid_purchase_results = []
    discharging_results = []
    installation_results = []
    salvaging_results = []
    operating_results = []

    for node in scenarioTree.nodes:
        for key, var in node.g_Purchase.items():
            if var.X > tolerance:
                t, p = key
                grid_purchase_results.append({'Node': node, 'NodeID': node.id, 't': t, 'p': p, 'value': var.X})

        for key, var in node.i_Discharging.items():
            if var.X > tolerance:
                t, p = key
                discharging_results.append({'Node': node, 'NodeID': node.id, 't': t, 'p': p, 'value': var.X})

        for key, var in node.v_Plus.items():
            if var.X > tolerance:
                tech_type, v, t = key
                installation_results.append({'Node': node, 'NodeID': node.id, 'tech': node.techNodeList[node.tech_types.index(tech_type)], 'tech_type': tech_type, 'v': v, 't': t, 'value': var.X})

        for key, var in node.v_Minus.items():
            if var.X > tolerance:
                tech_type, v, t, t_ = key
                salvaging_results.append({'Node': node, 'NodeID': node.id, 'tech': node.techNodeList[node.tech_types.index(tech_type)], 'tech_type': tech_type, 'v': v, 't': t, 't_': t_, 'value': var.X})

        for key, var in node.v_Existing.items():
            if var.X > tolerance:
                tech_type, v, t, t_ = key
                operating_results.append({'Node': node, 'NodeID': node.id, 'tech': node.FindAncestorFromDiff(t, t_).techNodeList[node.FindAncestorFromDiff(t, t_).tech_types.index(tech_type)], 'tech_type': tech_type, 'v': v, 't': t, 't_': t_, 'value': var.X})

    purchase_and_sales_results = []
    for node in scenarioTree.nodes:
        for each_period in node.stageSubperiods:
            annual_grid_purchase = sum([each_grid_purchase_result['value'] for each_grid_purchase_result in grid_purchase_results if (each_grid_purchase_result['NodeID'] == node.id and each_grid_purchase_result['t'] == each_period)])
            if annual_grid_purchase > tolerance:
                purchase_and_sales_results.append({'NodeID': node.id, 't': each_period, 'variable': 'grid', 'event': 'purchase', 'quantity': annual_grid_purchase, 'cost_per_unit': grid_electricity_cost[each_period]*(discount_factor**each_period), 'total_cost': annual_grid_purchase*grid_electricity_cost[each_period]*(discount_factor**each_period)})
            for each_installation_result in installation_results:
                if (each_installation_result['t'] == each_period and each_installation_result['NodeID'] == node.id and each_installation_result['value'] > tolerance):
                    purchase_and_sales_results.append({'NodeID': node.id, 't': each_period, 'variable': each_installation_result['tech_type'] + 'V' + str(each_installation_result['v']), 'event': 'installation', 'quantity': each_installation_result['value'], 'cost_per_unit': each_installation_result['tech'].cost[each_installation_result['v']]*(discount_factor**each_period), 'total_cost': each_installation_result['value']*each_installation_result['tech'].cost[each_installation_result['v']]*(discount_factor**each_period)})
            for each_salvaging_result in salvaging_results:
                if (each_salvaging_result['t_'] == each_period and each_salvaging_result['NodeID'] == node.id and each_salvaging_result['value'] > tolerance):
                    purchase_and_sales_results.append({'NodeID': node.id, 't': each_salvaging_result['t'], 't_': each_salvaging_result['t_'], 'variable': each_salvaging_result['tech_type'] + 'V' + str(each_salvaging_result['v']), 'event': 'salvaging', 'quantity': each_salvaging_result['value'], 'cost_per_unit': each_salvaging_result['tech'].salvage_value[each_salvaging_result['v']]*(1 - (each_salvaging_result['tech'].depreciation_rate[each_salvaging_result['v']] * (each_salvaging_result['t_'] - each_salvaging_result['t'])))*(discount_factor**each_salvaging_result['t_']), 'total_cost': each_salvaging_result['value']*each_salvaging_result['tech'].salvage_value[each_salvaging_result['v']]*(1 - (each_salvaging_result['tech'].depreciation_rate[each_salvaging_result['v']] * (each_salvaging_result['t_'] - each_salvaging_result['t'])))*(discount_factor**each_salvaging_result['t_'])})

    node_summary_installation_dict = defaultdict(float)
    node_summary_installation_cost_dict = defaultdict(float)
    node_summary_salvage_dict = defaultdict(float)
    node_summary_salvage_cost_dict = defaultdict(float)
    node_summary_grid_purchase_dict = defaultdict(float)
    node_summary_grid_purchase_cost_dict = defaultdict(float)
    node_summary_operating_cost_dict = defaultdict(float)

    for each_installation_result in installation_results:
        if each_installation_result['value'] > tolerance:
            node_summary_installation_dict[(each_installation_result['NodeID'], each_installation_result['tech_type'], each_installation_result['v'])] += each_installation_result['value']
            node_summary_installation_cost_dict[(each_installation_result['NodeID'], each_installation_result['tech_type'], each_installation_result['v'])] += 0.000001 * each_installation_result['value']*each_installation_result['tech'].cost[each_installation_result['v']]*(discount_factor**each_installation_result['t'])

    for each_salvaging_result in salvaging_results:
        if each_salvaging_result['value'] > tolerance:
            node_summary_salvage_dict[(each_salvaging_result['NodeID'], each_salvaging_result['tech_type'], each_salvaging_result['v'])] += each_salvaging_result['value']
            node_summary_salvage_cost_dict[(each_salvaging_result['NodeID'], each_salvaging_result['tech_type'], each_salvaging_result['v'])] += 0.000001 * each_salvaging_result['value']*each_salvaging_result['tech'].salvage_value[each_salvaging_result['v']]*(1 - (each_salvaging_result['tech'].depreciation_rate[each_salvaging_result['v']] * (each_salvaging_result['t_'] - each_salvaging_result['t'])))*(discount_factor**each_salvaging_result['t_'])

    for each_grid_result in grid_purchase_results:
        if each_grid_result['value'] > tolerance:
            node_summary_grid_purchase_dict[each_grid_result['NodeID']] += each_grid_result['value']
            node_summary_grid_purchase_cost_dict[each_grid_result['NodeID']] += 0.000001 * each_grid_result['value']*grid_electricity_cost[each_grid_result['t']]*(discount_factor**each_grid_result['t'])

    for each_operating_result in operating_results:
        if each_operating_result['value'] > tolerance:
            node_summary_operating_cost_dict[(each_operating_result['NodeID'])] += 0.000001 * each_operating_result['value']*each_operating_result['tech'].OMcost[each_operating_result['v']] * ((each_operating_result['tech'].OMcostchangebyage[each_operating_result['v']])**(each_operating_result['t_'] - each_operating_result['t'])) * ((each_operating_result['tech'].OMcostchangebyyear[each_operating_result['v']])**(each_operating_result['t'])) * (discount_factor**each_operating_result['t_'])

    node_summary_results = []

    for (node_id, tech_type, v), total_value in node_summary_installation_dict.items():
        node_summary_results.append({'NodeID': node_id, f'Installation{tech_type}_V{v}': total_value})

    total_installation_cost = {}
    for (node_id, tech_type, v), total_value in node_summary_installation_cost_dict.items():
        node_summary_results.append({'NodeID': node_id, f'InstallationCost{tech_type}_V{v}': total_value})
        total_installation_cost[node_id] = total_installation_cost.get(node_id, 0) + total_value

    for (node_id, tech_type, v), total_value in node_summary_salvage_dict.items():
        node_summary_results.append({'NodeID': node_id, f'Salvage{tech_type}_V{v}': total_value})

    total_salvage_cost = {}
    for (node_id, tech_type, v), total_value in node_summary_salvage_cost_dict.items():
        node_summary_results.append({'NodeID': node_id, f'SalvageCost{tech_type}_V{v}': total_value})
        total_salvage_cost[node_id] = total_salvage_cost.get(node_id, 0) + total_value

    for node_id, total_value in node_summary_grid_purchase_dict.items():
        node_summary_results.append({'NodeID': node_id, 'GridPurchase': total_value})

    for node_id, total_value in node_summary_grid_purchase_cost_dict.items():
        node_summary_results.append({'NodeID': node_id, 'GridPurchaseCost': total_value})

    for node_id, total_value in node_summary_operating_cost_dict.items():
        node_summary_results.append({'NodeID': node_id, 'TotalO&MCost': total_value})
    
    for node_id, total_value in total_installation_cost.items():
        node_summary_results.append({'NodeID': node_id, 'TotalInstallationCost': total_value})

    for node_id, total_value in total_salvage_cost.items():
        node_summary_results.append({'NodeID': node_id, 'TotalSalvageCost': total_value})

    combined_node_summary = {}

    for entry in node_summary_results:
        node_id = entry['NodeID']
        if node_id not in combined_node_summary:
            combined_node_summary[node_id] = {'NodeID': node_id}
        for key, value in entry.items():
            if key != 'NodeID':
                combined_node_summary[node_id][key] = value

    for node_id, summary in combined_node_summary.items():
        summary['TotalNodeCost'] = summary.get('TotalInstallationCost', 0) + summary.get('GridPurchaseCost', 0) + summary.get('TotalO&MCost', 0) - summary.get('TotalSalvageCost', 0)

    node_summary_list = list(combined_node_summary.values())

    for leaf_node in [node for node in scenarioTree.nodes if not node.children]:
        path_ids = []
        current_node = leaf_node
        while current_node is not None:
            path_ids.append(current_node.id)
            current_node = current_node.parent

        for entry in node_summary_list:
            if entry["NodeID"] == leaf_node.id:
                entry["PathGridPurchaseQuantity"] = sum(rec.get("GridPurchase", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)                
                entry["PathBatteryInstallationQuantity"] = sum(rec.get("Installationbattery_V0", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)  
                entry["PathO&MCost"] = sum(rec.get("TotalO&MCost", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)
                entry["PathSalvageCost"] = sum(rec.get("TotalSalvageCost", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)
                entry["PathGridPurchaseCost"] = sum(rec.get("GridPurchaseCost", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)
                entry["PathInstallationCost"] = sum(rec.get("TotalInstallationCost", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)
                entry["PathTotalCost"] = sum(rec.get("TotalNodeCost", 0) for rec in node_summary_list if rec["NodeID"] in path_ids)              
                break

    annual_summary_results = []
    annual_production_results_dict = defaultdict(float)
    annual_grid_purchase_results_dict = defaultdict(float)
    annual_grid_cost_results_dict = defaultdict(float)
    annual_om_cost_results_dict = defaultdict(float)
    annual_installation_cost_results_dict = defaultdict(float)
    annual_salvaging_cost_results_dict = defaultdict(float)
    annual_spatial_usage_results_dict = defaultdict(float)
    annual_discharging_results_dict = defaultdict(float)

    operating_dict = {}
    for each_operating_result in operating_results:
        if (each_operating_result['value'] > tolerance and each_operating_result['tech'].tree.segment == 'production'):
            annual_production_results_dict[(each_operating_result['NodeID'], each_operating_result['t_'], each_operating_result['tech_type'])] += each_operating_result['value'] * sum([each_operating_result['tech'].periodic_electricity[each_operating_result['v']][p] for p in range(numSubterms)]) * (1 - (each_operating_result['tech'].degradation_rate[each_operating_result['v']] * (each_operating_result['t_'] - each_operating_result['t'])))
            for p in range(numSubterms):
                operating_dict[(each_operating_result['NodeID'], each_operating_result['t_'], p)] = operating_dict.get((each_operating_result['NodeID'], each_operating_result['t_'], p), 0) + each_operating_result['value'] * each_operating_result['tech'].periodic_electricity[each_operating_result['v']][p] * (1 - (each_operating_result['tech'].degradation_rate[each_operating_result['v']] * (each_operating_result['t_'] - each_operating_result['t'])))           

        if each_operating_result['value'] > tolerance:
            annual_om_cost_results_dict[(each_operating_result['NodeID'], each_operating_result['t_'])] += 0.000001 * each_operating_result['value'] * each_operating_result['tech'].OMcost[each_operating_result['v']] * ((each_operating_result['tech'].OMcostchangebyage[each_operating_result['v']])**(each_operating_result['t_'] - each_operating_result['t'])) * ((each_operating_result['tech'].OMcostchangebyyear[each_operating_result['v']])**(each_operating_result['t'])) * (discount_factor**each_operating_result['t_'])
            annual_spatial_usage_results_dict[(each_operating_result['NodeID'], each_operating_result['t_'])] += each_operating_result['value'] * each_operating_result['tech'].spatial_requirement[each_operating_result['v']]

    renewable_met_dict = {}
    for (node_id, t_, p), total_value in operating_dict.items():
        renewable_met_dict[(node_id, t_)] = renewable_met_dict.get((node_id, t_), 0) + min(demand[t_][p], total_value)

    for each_grid_result in grid_purchase_results:
        if each_grid_result['value'] > tolerance:
            annual_grid_purchase_results_dict[(each_grid_result['NodeID'], each_grid_result['t'])] += each_grid_result['value']
            annual_grid_cost_results_dict[(each_grid_result['NodeID'], each_grid_result['t'])] += 0.000001 * each_grid_result['value']*grid_electricity_cost[each_grid_result['t']]*(discount_factor**each_grid_result['t'])

    for each_installation_result in installation_results:
        if each_installation_result['value'] > tolerance:
            annual_installation_cost_results_dict[(each_installation_result['NodeID'], each_installation_result['t'])] += 0.000001 * each_installation_result['value']*each_installation_result['tech'].cost[each_installation_result['v']]*(discount_factor**each_installation_result['t'])

    for each_salvaging_result in salvaging_results:
        if each_salvaging_result['value'] > tolerance:
            annual_salvaging_cost_results_dict[(each_salvaging_result['NodeID'], each_salvaging_result['t_'])] += 0.000001 * each_salvaging_result['value']*each_salvaging_result['tech'].salvage_value[each_salvaging_result['v']]*(1 - (each_salvaging_result['tech'].depreciation_rate[each_salvaging_result['v']] * (each_salvaging_result['t_'] - each_salvaging_result['t'])))*(discount_factor**each_salvaging_result['t_'])

    for (node_id, t_, tech_type), total_value in annual_production_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t_, f'{tech_type}_Production': total_value})

    for (node_id, t_), total_value in annual_spatial_usage_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t_, 'SpatialUsage': total_value})

    for (node_id, t), total_value in annual_grid_cost_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'GridPurchaseCost': total_value})

    for (node_id, t), total_value in annual_installation_cost_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'InstallationCost': total_value})

    for (node_id, t), total_value in annual_salvaging_cost_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'SalvageCost': total_value})

    for (node_id, t), total_value in annual_om_cost_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'O&MCost': total_value})

    for (node_id, t), total_value in annual_grid_purchase_results_dict.items():
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'GridPurchaseQuantity': total_value})
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'GridPercentage': 100*total_value/sum(demand[t])})

    for result in discharging_results:
        if result['value'] > tolerance:
            key = (result['NodeID'], result['t'])
            annual_discharging_results_dict[key] = annual_discharging_results_dict.get(key, 0) + result['value']

    for (node_id, t), total_value in annual_discharging_results_dict.items(): #InventoryPercentage and usage results are not reliable due multiple optima. Use 100 - generation percentage - grid percentage instead.
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'InventoryUsage': total_value})
        annual_summary_results.append({'NodeID': node_id, 't_': t, 'InventoryPercentage': 100*total_value / sum(demand[t]) if sum(demand[t]) > tolerance else 0})

    combined_annual_summary_results = {}
    for entry in annual_summary_results:
        key = (entry["NodeID"], entry["t_"])
        if key not in combined_annual_summary_results:
            combined_annual_summary_results[key] = {"NodeID": entry["NodeID"], "t": entry["t_"]}
        for k, v in entry.items():
            if k not in ("NodeID", "t_"):
                combined_annual_summary_results[key][k] = v

    combined_annual_summary_results_list = list(combined_annual_summary_results.values())
    for entry in combined_annual_summary_results_list:
        entry["RenewablePercentage"] = 100 * renewable_met_dict.get((entry["NodeID"], entry["t"]), 0) / sum(demand[entry["t"]]) if sum(demand[entry["t"]]) > tolerance else 0
        entry["TotalCost"] = entry.get("InstallationCost", 0) + entry.get("GridPurchaseCost", 0) + entry.get("O&MCost", 0) - entry.get("SalvageCost", 0)

    node_summary_df = pd.DataFrame(node_summary_list)
    annual_summary_df = pd.DataFrame(combined_annual_summary_results_list)
    purchase_sales_df = pd.DataFrame(purchase_and_sales_results)

    node_summary_df.to_csv(os.path.join(results_directory, f'NodeSummary_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv'), index=False)
    annual_summary_df.to_csv(os.path.join(results_directory, f'AnnualSummary_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv'), index=False)
    purchase_sales_df.to_csv(os.path.join(results_directory, f'PurchaseAndSales_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv'), index=False)

def OptimizationModel(scenarioTree, emission_limits, demand, numStages, numSubperiods, numMultipliers, numSubterms, initial_tech, budget, grid_electricity_cost, safety_stock, discount_factor = 0.97):
    model = Model('MachineReplacement')
    model.setParam('OutputFlag', True)

    for node in scenarioTree.nodes:
        node.AddVariables(model)
        node.AddObjectiveCoefficients(model, grid_electricity_cost, discount_factor)
        node.AddTechnologyBalanceConstraints(model)
        node.AddDemandConstraints(model, demand)
        node.AddInventoryBalanceConstraints(model)
        node.AddBatteryCapacityConstraints(model)
        node.AddBudgetConstraints(model, budget)
        node.AddEmissionConstraints(model, emission_limits)
        node.InitializeCurrentTech(initial_tech)
        node.AddUpperBoundsForIP(model, demand)
        node.AddSpatialConstraints(model, spatial_limit=None)

    results_directory = f'Results_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    model.setParam('MIPGap', 0.01)
    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', 86400)
    model.setParam('Threads', 20)
    model.setParam('NodefileStart', 0.95)
    model.setParam('NodefileDir', '.')
    model.setParam('LogFile', os.path.join(results_directory, f'GurobiLog_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.txt'))

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    optimization_time = end_time - start_time

    lp_filename = os.path.join(results_directory, 'MachineReplacement.lp')
    sol_filename = os.path.join(results_directory, 'MachineReplacement.sol')
    #model.write(lp_filename)
    model.write(sol_filename)

    OutputProductionResults(model, scenarioTree, discount_factor, demand, numStages, numSubperiods, numSubterms, numMultipliers, results_directory, grid_electricity_cost)
    Output(model)

    model_results = {
        'numStages': numStages,
        'numSubperiods': numSubperiods,
        'numSubterms': numSubterms,
        'numMultipliers': numMultipliers,
        'Number of Nodes': len(scenarioTree.nodes),
        'Number of Variables': model.NumVars,
        'Number of Constraints': model.NumConstrs,
        'Optimization Time (s)': optimization_time,
        'Objective Function Value': model.objVal
    }

    return model_results

def clustering_n_consecutive_data_points(values, n):
    result = []
    for i in range(0, len(values), n):
        cluster_sum = sum(values[i:i+n])
        result.append(cluster_sum)
    return result

electricity_demand_2023 = pd.read_excel(os.path.join('Data', 'Demand.xlsx'), sheet_name='2023 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
electricity_demand_2023 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2023[j] for j in range(i - 1, -1, -1) if electricity_demand_2023[j] >= 100), None), next((electricity_demand_2023[j] for j in range(i + 1, len(electricity_demand_2023)) if electricity_demand_2023[j] >= 100), None))) for i, x in enumerate(electricity_demand_2023)]

electricity_demand_2024 = pd.read_excel(os.path.join('Data', 'Demand.xlsx'), sheet_name='2024 Hourly Electricity Demand')['Consumption (kWh/h)'].tolist()
electricity_demand_2024 = [x if x >= 100 else ((lambda lv, rv: (lv + rv) / 2 if lv is not None and rv is not None else x)(next((electricity_demand_2024[j] for j in range(i - 1, -1, -1) if electricity_demand_2024[j] >= 100), None), next((electricity_demand_2024[j] for j in range(i + 1, len(electricity_demand_2024)) if electricity_demand_2024[j] >= 100), None))) for i, x in enumerate(electricity_demand_2024)]

electricity_demand_2023 = electricity_demand_2023[24:]
electricity_demand_2024 = electricity_demand_2024[:(8760-24)]

base_electricity_demand = [(val_2023 + val_2024) / 2 for val_2023, val_2024 in zip(electricity_demand_2023, electricity_demand_2024)]

solar_initial = pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Initial values')
solar_advancements = {1: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements1'),
                      2: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements2'),
                      3: pd.read_excel(os.path.join('Data', 'Solar Power.xlsx'), sheet_name='Advancements3')}
base_solar_periodic_production = pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='solar').T.values.tolist()[:(8760-24)]

wind_initial = pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Initial values')
wind_advancements = {1: pd.read_excel(os.path.join('Data', 'Wind Power.xlsx'), sheet_name='Advancements1')}
base_wind_periodic_production = pd.read_excel(os.path.join('Data', 'Technology Subterm Data.xlsx'), sheet_name='wind').T.values.tolist()[:((8760-24))]

battery_initial = pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Initial values')
battery_advancements = {1: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements1'),
                        2: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements2'),
                        3: pd.read_excel(os.path.join('Data', 'Electricity Storage.xlsx'), sheet_name='Advancements3')}

num_Stages_list = [3]
num_Subperiods_list = [5]
num_Subterms_list = [4368]
num_Multipliers_list = [2]

results = {}

for numStages in num_Stages_list:
    for numSubperiods in num_Subperiods_list:
        for numSubterms in num_Subterms_list:

            electricity_demand = clustering_n_consecutive_data_points(base_electricity_demand, int((8760-24)/numSubterms))
            solar_periodic_production = [clustering_n_consecutive_data_points(version_production, int((8760-24)/numSubterms)) for version_production in base_solar_periodic_production]
            wind_periodic_production = [clustering_n_consecutive_data_points(version_production, int((8760-24)/numSubterms)) for version_production in base_wind_periodic_production]

            grid_electricity_cost = [0.144 for _ in range(numStages*numSubperiods+1)]
            emission_limits = [None for _ in range(numStages*numSubperiods)] + [0] # [0.01 * sum(electricity_demand)]
            safety_stock = [None for _ in range(numStages*numSubperiods+1)]
            budget = [0, 5000000, 5000000, 5000000, 5000000, 5000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000]

            for numMultipliers in num_Multipliers_list:
                solar = TechnologyTree('solar', 'production', numSubperiods, numSubterms, 
                    lifetime=[solar_initial.iloc[2, i] for i in range(1, solar_initial.shape[1])], 
                    initialCost=[solar_initial.iloc[0, i] for i in range(1, solar_initial.shape[1])], 
                    initialEmission=[solar_initial.iloc[1, i] for i in range(1, solar_initial.shape[1])], 
                    degradation_rate=[solar_initial.iloc[3, i] for i in range(1, solar_initial.shape[1])], 
                    initialOMcost=[solar_initial.iloc[4, i] for i in range(1, solar_initial.shape[1])], 
                    OMcostchangebyage=[solar_initial.iloc[5, i] for i in range(1, solar_initial.shape[1])], 
                    depreciation_rate=[solar_initial.iloc[6, i] for i in range(1, solar_initial.shape[1])],
                    initial_salvage_value=[solar_initial.iloc[7, i] for i in range(1, solar_initial.shape[1])],
                    OMcostchangebyyear=[solar_initial.iloc[9, i] for i in range(1, solar_initial.shape[1])],
                    spatial_requirement=[solar_initial.iloc[10, i] for i in range(1, solar_initial.shape[1])],
                    periodic_electricity_production = [[max(0, x) for x in sublist[:numSubterms]] for sublist in solar_periodic_production])
                solar.ConstructByMultipliers(numStages, probabilities=[solar_advancements[numMultipliers][col][0] for col in solar_advancements[numMultipliers].columns if col != "Metrics"], costMultiplier=[solar_advancements[numMultipliers][col][3] for col in solar_advancements[numMultipliers].columns if col != "Metrics"], efficiencyMultiplier=[solar_advancements[numMultipliers][col][4] for col in solar_advancements[numMultipliers].columns if col != "Metrics"], emissionMultiplier=[solar_advancements[numMultipliers][col][5] for col in solar_advancements[numMultipliers].columns if col != "Metrics"])

                wind = TechnologyTree('wind', 'production', numSubperiods, numSubterms, 
                    lifetime=[wind_initial.iloc[2, i] for i in range(1, wind_initial.shape[1])], 
                    initialCost=[wind_initial.iloc[0, i] for i in range(1, wind_initial.shape[1])], 
                    initialEmission=[wind_initial.iloc[1, i] for i in range(1, wind_initial.shape[1])], 
                    degradation_rate=[wind_initial.iloc[3, i] for i in range(1, wind_initial.shape[1])], 
                    initialOMcost=[wind_initial.iloc[4, i] for i in range(1, wind_initial.shape[1])], 
                    OMcostchangebyage=[wind_initial.iloc[5, i] for i in range(1, wind_initial.shape[1])],
                    depreciation_rate=[wind_initial.iloc[6, i] for i in range(1, wind_initial.shape[1])], 
                    initial_salvage_value=[wind_initial.iloc[7, i] for i in range(1, wind_initial.shape[1])],
                    OMcostchangebyyear=[wind_initial.iloc[9, i] for i in range(1, wind_initial.shape[1])],
                    spatial_requirement=[wind_initial.iloc[10, i] for i in range(1, wind_initial.shape[1])],
                    periodic_electricity_production = [[max(0, x) for x in sublist[:numSubterms]] for sublist in wind_periodic_production])
                wind.ConstructByMultipliers(numStages, probabilities=[wind_advancements[1][col][0] for col in wind_advancements[1].columns if col != "Metrics"], costMultiplier=[wind_advancements[1][col][3] for col in wind_advancements[1].columns if col != "Metrics"], efficiencyMultiplier=[wind_advancements[1][col][4] for col in wind_advancements[1].columns if col != "Metrics"], emissionMultiplier=[wind_advancements[1][col][5] for col in wind_advancements[1].columns if col != "Metrics"])

                battery = TechnologyTree('battery', 'storage', numSubperiods, numSubterms, 
                    lifetime=[battery_initial.iloc[2, i] for i in range(1, battery_initial.shape[1])], 
                    initialCost=[battery_initial.iloc[0, i] for i in range(1, battery_initial.shape[1])], 
                    initialEmission=[battery_initial.iloc[1, i] for i in range(1, battery_initial.shape[1])], 
                    degradation_rate=[battery_initial.iloc[3, i] for i in range(1, battery_initial.shape[1])], 
                    initialOMcost=[battery_initial.iloc[4, i] for i in range(1, battery_initial.shape[1])], 
                    OMcostchangebyage=[battery_initial.iloc[5, i] for i in range(1, battery_initial.shape[1])],
                    depreciation_rate=[battery_initial.iloc[6, i] for i in range(1, battery_initial.shape[1])], 
                    initial_salvage_value=[battery_initial.iloc[7, i] for i in range(1, battery_initial.shape[1])],
                    OMcostchangebyyear=[battery_initial.iloc[9, i] for i in range(1, battery_initial.shape[1])],
                    spatial_requirement=[battery_initial.iloc[10, i] for i in range(1, battery_initial.shape[1])],
                    electricity_storage_capacity=[battery_initial.iloc[11, i] for i in range(1, battery_initial.shape[1])],
                    storage_charging_efficiency=[battery_initial.iloc[12, i] for i in range(1, battery_initial.shape[1])],
                    storage_discharging_efficiency=[battery_initial.iloc[13, i] for i in range(1, battery_initial.shape[1])])
                battery.ConstructByMultipliers(numStages, probabilities=[battery_advancements[numMultipliers][col][0] for col in battery_advancements[numMultipliers].columns if col != "Metrics"], costMultiplier=[battery_advancements[numMultipliers][col][3] for col in battery_advancements[numMultipliers].columns if col != "Metrics"], efficiencyMultiplier=[battery_advancements[numMultipliers][col][4] for col in battery_advancements[numMultipliers].columns if col != "Metrics"], emissionMultiplier=[battery_advancements[numMultipliers][col][5] for col in battery_advancements[numMultipliers].columns if col != "Metrics"])

                scenarioTree = ScenarioTree([solar, wind, battery])

                initial_tech = [[solar_initial.iloc[8, i] for i in range(1, solar_initial.shape[1])],
                                [wind_initial.iloc[8, i] for i in range(1, wind_initial.shape[1])],
                                [battery_initial.iloc[8, i] for i in range(1, battery_initial.shape[1])]]

                key = f's{numStages}_p{numSubperiods}_t{numSubterms}_n{numMultipliers}'
                results[key] = OptimizationModel(scenarioTree, emission_limits, demand=[electricity_demand[:numSubterms]]*(numStages*numSubperiods + 1), numStages=numStages, numSubperiods=numSubperiods, numMultipliers=numMultipliers, numSubterms=numSubterms, initial_tech=initial_tech, budget=budget, grid_electricity_cost=grid_electricity_cost, safety_stock=safety_stock)

                #df_results = pd.DataFrame.from_dict(results, orient='index')
                #df_results.reset_index(inplace=True)
                #df_results.rename(columns={'index': 'Scenario'}, inplace=True)
                #excel_filename = 'comparison_results.xlsx'
                #df_results.to_excel(excel_filename, index=False)
                #print(f"Results saved to {excel_filename}")