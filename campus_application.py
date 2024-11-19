from gurobipy import GRB,Model,quicksum
import itertools
import time
import pandas as pd
import math
from functools import reduce

class ScenarioTree:
    def __init__(self, technologyTrees):
        self.numStages = technologyTrees[0].nodes[-1].stage
        self.numSubperiods = technologyTrees[0].nodes[-1].numSubperiods
        self.numSubterms = technologyTrees[0].numSubterms
        self.nodes = []

        prerootTechNodeList = []
        for tech in technologyTrees:
            prerootTechNodeList.append(tech.nodes[0])
        self.preroot = ScenarioNode(0, None, 1, self, prerootTechNodeList)

        rootTechNodeList = []
        for tech in technologyTrees:
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

    def Print(self):
        for node in self.nodes:
            node.Print_()

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
        self.productiontechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == "Production"]
        self.storagetechNodeList = [tech for tech in self.techNodeList if tech.tree.segment == "Storage"]

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

    def Print_(self):
        techNodeIDs = ""
        for tech in self.techNodeList:
            techNodeIDs = techNodeIDs + str(tech.cost) + ","

        str_ = ""
        for s in range(self.stage-1):
            str_ = str_ + "\t"
        prt="()"
        if self.parent is not None:
            prt= str("(")+str(self.parent.id)+str(")")
            print(str_ + str(self.id) + str(prt) + ";" + str(round(self.probability,5)) + "; " + techNodeIDs)

    def FindAncestor(self, t):
        ancestor = self
        while t not in ancestor.stageSubperiods:
            ancestor = ancestor.parent
        return ancestor

    def AddVariables(self, model):
        self.v_Plus = {}
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.productiontechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], vtype=GRB.INTEGER, name="plus_"+str(self.id))) # purchase
        self.v_Plus.update(model.addVars([(tech.tree.type, v, t)  for tech in self.storagetechNodeList for v in range(tech.NumVersion) for t in self.stageSubperiods], vtype=GRB.CONTINUOUS, name="plus_"+str(self.id))) # purchase
        self.v_Minus = {}
        self.v_Minus.update(model.addVars([(tech.tree.type, v, t, t_)  for tech in self.productiontechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.INTEGER, name="minus_"+str(self.id))) # retired
        self.v_Minus.update(model.addVars([(tech.tree.type, v, t, t_)  for tech in self.storagetechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, name="minus_"+str(self.id))) # retired
        self.v_Existing = model.addVars([(tech.tree.type, v, t, t_) for tech in self.techNodeList for v in range(tech.NumVersion) for t in self.allSubperiods for t_ in self.stageSubperiods if t <= t_ < t + tech.lifetime[v]], vtype=GRB.CONTINUOUS, name="exist_"+str(self.id)) # exist
        self.g_Purchase = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="grid_purchase_"+str(self.id)) # grid purchase amount
        self.i_Carrying = model.addVars([(t, p) for t in self.stageSubperiods for p in self.stageSubterms], vtype=GRB.CONTINUOUS, name="carry_"+str(self.id)) # inventory carriage amount

    def AddObjectiveCoefficients(self, model, grid_electricity_cost, discount_factor):
        for i, tech in enumerate(self.techNodeList):
            for v in range(tech.NumVersion):
                for t_ in self.stageSubperiods:
                    self.v_Plus[tech.tree.type,v,t_].obj = self.probability * tech.cost[v] * (discount_factor**(t_))
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            self.v_Minus[tech.tree.type,v,t,t_].obj = -self.probability * tech.salvage_value[v] * (tech.depreciation_rate[v] * tech.lifetime[v] - (tech.depreciation_rate[v] * (t_ - t))) * (discount_factor**(t_))
                            self.v_Existing[tech.tree.type,v,t,t_].obj = self.probability * self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcost[v] * ((self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcostchangebyage[v])**(t_ - t)) * ((self.FindAncestorFromDiff(t,t_).techNodeList[i].OMcostchangebyyear[v])**(t)) * (discount_factor**(t_))

        for t in self.stageSubperiods:
            for p in self.stageSubterms:
                self.g_Purchase[t,p].obj = self.probability * grid_electricity_cost[t] * (discount_factor**(t))

    def AddBalanceConstraints(self, model):
        for tech in self.techNodeList:
            for v in range(tech.NumVersion):
                for t_ in self.stageSubperiods:
                    for t in self.allSubperiods:
                        if t <= t_ < t + tech.lifetime[v]:
                            model.addConstr(self.v_Existing[tech.tree.type,v,t,t_] == self.FindAncestorFromDiff(t,t_).v_Plus[tech.tree.type,v,t]
                                - quicksum(self.FindAncestorFromDiff(t__, t_).v_Minus[tech.tree.type,v,t,t__] for t__ in range(t,t_+1)), name = f'N{self.id}_Balance_{tech.tree.type}_{v}_{t}_{t_}')

    def SetMinustoZeroConstraints(self, model):
        for t in self.stageSubperiods:
            for tech in self.techNodeList:
                for v in range(tech.NumVersion):
                    model.addConstr(self.v_Minus[tech.tree.type,v,t,t] == 0, name = f'N{self.id}_MinusTechZero_{tech.tree.type}_{v}_{t}')

    def FindAncestorFromDiff(self, t, t_):
        ancestor = self
        amount_subperiods = len(ancestor.stageSubperiods) 
        node_no_1 = (t-1) // amount_subperiods
        node_no_2 = (t_-1) // amount_subperiods
        how_many_more_ancestors = node_no_2 - node_no_1
        for _ in range(how_many_more_ancestors):
            ancestor = ancestor.parent
        return ancestor

    def InitializeCurrentTech(self, model, initial_tech):
        if self.parent == None:
            for i, tech in enumerate(self.techNodeList):
                for v in range(tech.NumVersion):
                    if initial_tech[i][v] != 0:
                        self.v_Plus[tech.tree.type, v, 0].lb = initial_tech[i][v]
                    else:
                        self.v_Plus[tech.tree.type, v, 0].ub = 0

    def AddEmissionConstraints(self, model, emission_limits):
        for t in self.stageSubperiods:
            if emission_limits[t] != None:
                model.addConstr(quicksum(self.g_Purchase[t,p] for p in self.stageSubterms) <= emission_limits[t], name = f'N{self.id}_Emission_{t}')

    def AddBudgetConstraints(self, model, budget): # Give budget list without discount factor or add discount factor to both sides in following code
        for t in self.stageSubperiods:
            if budget[t] != None:
                model.addConstr(quicksum(tech.cost[v] * self.v_Plus[tech.tree.type,v,t] for tech in self.techNodeList for v in range(tech.NumVersion)) <= budget[t], name = f'N{self.id}_Budget_{t}')

    def AddDemandConstraints(self, model, demand):
        if self.id != 0:
            for t_ in self.stageSubperiods:
                for p, periodic_demand in enumerate(demand[t_]):
                    if p == 0:
                        model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p]*self.v_Existing[tech.tree.type,v,t,t_]*(1 - (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.productiontechNodeList) for v in range(self.productiontechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].lifetime[v]) + self.g_Purchase[t_, p+1] + self.FindAncestorFromDiff(t_-1,t_).i_Carrying[t_-1, len(demand[t_])] - self.i_Carrying[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')
                    else:
                        model.addConstr(quicksum((self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p]*self.v_Existing[tech.tree.type,v,t,t_]*(1 - (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].degradation_rate[v] * (t_ - t)))) for i, tech in enumerate(self.productiontechNodeList) for v in range(self.productiontechNodeList[i].NumVersion) for t in range(0,t_+1) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].lifetime[v]) + self.g_Purchase[t_, p+1] + self.i_Carrying[t_, p] - self.i_Carrying[t_, p+1] >= periodic_demand, name = f'N{self.id}_Demand_Electricity_{t_}_{p}')

    def AddBatteryCapacityConstraints(self, model):
        for t_ in self.stageSubperiods:
            for p in self.stageSubterms:
                model.addConstr(self.i_Carrying[t_,p] <= quicksum(self.v_Existing[tech.tree.type,v,t,t_] for tech in self.storagetechNodeList for v in range(tech.NumVersion) for t in self.allSubperiods if t <= t_ < t+tech.lifetime[v]), name = f'N{self.id}_BatteryCapacity_{t_}_{p}')

    def AddUpperBoundsForIP(self, model, demand):
        for i, tech in enumerate(self.productiontechNodeList):
            for v in range(tech.NumVersion):
                for t in self.stageSubperiods:
                    ub_v = math.ceil(max([(0 if self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p] == 0 else (periodic_demand / (self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].periodic_electricity[v][p] * ((1 - self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].degradation_rate[v])**(t_ - t))))) for t_ in self.allSubperiods for p, periodic_demand in enumerate(demand[t_]) if t <= t_ < t + self.FindAncestorFromDiff(t,t_).productiontechNodeList[i].lifetime[v]]))
                    model.addConstr(self.v_Plus[tech.tree.type,v,t] <= ub_v, name = f'N{self.id}_UpperBound_v_plus_{tech.tree.type}_{v}_{t}')
                    for t__ in self.allSubperiods:
                        if t__ <= t < t__ + self.FindAncestorFromDiff(t,t__).productiontechNodeList[i].lifetime[v]:
                            model.addConstr(self.v_Minus[tech.tree.type,v,t__,t] <= ub_v, name = f'N{self.id}_UpperBound_v_minus_{tech.tree.type}_{v}_{t__}_{t}')
                            model.addConstr(self.v_Existing[tech.tree.type,v,t__,t] <= ub_v, name = f'N{self.id}_UpperBound_v_exist_{tech.tree.type}_{v}_{t__}_{t}')

    def PrintEmissionConsumption(self, model):
        if len(self.children) == 0:
            print("Total emission_"+str(self.id)+ ": " + str(self.emissionExpr.getValue()))

class TechnologyTree: # preroot'ta techamounta self.initialTechAmount koydum, 0 konulursa node_0 constraintinden dolayı infeasible sonuç alınıyor
    def __init__(self, type, numSubperiods, numSubterms, lifetime, segment, initialCost, initialEfficiency, initialEmission, periodic_electricity_production, degradation_rate, initialOMcost, OMcostchangebyage, depreciation_rate, initial_salvage_value, OMcostchangebyyear):
        self.type = type
        self.initialCost = initialCost
        self.initialEfficiency = initialEfficiency
        self.initialEmission = initialEmission
        self.periodic_electricity_production = periodic_electricity_production
        self.nodes = []
        self.degradation_rate = degradation_rate
        self.initialOMcost = initialOMcost
        self.OMcostchangebyage = OMcostchangebyage
        self.OMcostchangebyyear = OMcostchangebyyear
        self.depreciation_rate = depreciation_rate
        self.lifetime = lifetime
        self.segment = segment[0]
        self.initial_salvage_value = initial_salvage_value
        self.numSubperiods = numSubperiods
        self.numSubterms = numSubterms
        self.versions = len(initialCost)
        self.preroot = TechnologyNode(0, None, 1, self, self.versions, [0 for _ in range(self.versions)], [0 for _ in range(self.versions)], [0 for _ in range(self.versions)], self.periodic_electricity_production, self.lifetime, self.degradation_rate, self.initialOMcost, self.OMcostchangebyage, self.depreciation_rate, self.initial_salvage_value, self.OMcostchangebyyear) # preroot is stage-0: the existing situation
        self.root = TechnologyNode(1, self.preroot, 1, self, self.versions, self.initialCost, self.initialEfficiency, self.initialEmission, self.periodic_electricity_production, self.lifetime, self.degradation_rate, self.initialOMcost, self.OMcostchangebyage, self.depreciation_rate, self.initial_salvage_value, self.OMcostchangebyyear) #root is the inital decision making node.

    def ConstructByMultipliers(self, numStages, probabilities, costMultiplier, efficiencyMultiplier, emissionMultiplier):
        leaves = [self.root] # root is stage 1
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

    def Print(self):
        print()
        print(self.type)
        for node in self.nodes:
            node.Print_()

class TechnologyNode:
    def __init__(self, id_In, parent_In, probability_In, tree_In, versionnum_In, cost_In, efficiency_In, emission_In, periodic_electricity_In, lifetime_In, degradation_In, OMcost_In, OMcostchangebyage_In, depreciation_In, salvage_value_In, OMcostchangebyyear_In):
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
        self.efficiency = efficiency_In
        self.emission = emission_In
        self.periodic_electricity = periodic_electricity_In
        self.degradation_rate = degradation_In
        self.OMcost = OMcost_In
        self.OMcostchangebyage = OMcostchangebyage_In
        self.OMcostchangebyyear = OMcostchangebyyear_In
        self.depreciation_rate = depreciation_In
        self.salvage_value = salvage_value_In

    def Print(self):
        print(self.id)
        if self.parent is None:
            print("none")
        else:
            print(self.parent.id)
        print(self.stage)
        print(self.probability)
        print(self.cost)
        print(self.efficiency)
        print(self.emission)
        print(self.periodic_electricity)
        print(self.lifetime)
        print(self.degradation_rate)
        print(self.OMcost)
        print(self.OMcostchangebyage)
        print(self.OMcostchangebyyear)
        print(self.depreciation_rate)
        print(self.salvage_value)
        print()

    def Print_(self):                     #This function will error when it is called since versions are added
        str_ = ""
        for s in range(self.stage-1):
            str_ = str_ + "\t"
        prt="()"
        if self.parent is not None:
            prt= str("(")+str(self.parent.id)+str(")")
            print(str_ + str(self.id) + str(prt) + ";" + str(round(self.probability,5)) + "; " + str(round(self.cost,3)) + " " + str(round(self.efficiency,3)) + " " + str(round(self.emission,3)) + " " + str(round(self.periodic_electricity,3)))

    def AddChild(self, prob, costMult, effMult, emisMult):
        child = TechnologyNode(len(self.tree.nodes), self, prob, self.tree, self.NumVersion, [i*costMult for i in self.cost], [i*effMult for i in self.efficiency], [i*emisMult for i in self.emission], [[x*effMult for x in i] for i in self.periodic_electricity], self.lifetime, self.degradation_rate, self.OMcost, self.OMcostchangebyage, self.depreciation_rate, [i*costMult for i in self.salvage_value], self.OMcostchangebyyear)
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

def OutputProductionResults(model, scenarioTree, discount_factor, demand, numStages, numSubperiods, numSubterms, numMultipliers):
    results = []
    purchase_sale_results = []
    for node in scenarioTree.nodes:
        for key, var in node.v_Active.items():
            var_value = var.X
            if var_value > 0:
                tech_type, v, t, t_, p = key
                tech = node.FindAncestorFromDiff(t, t_).techNodeList[node.FindAncestorFromDiff(t, t_).tech_types.index(tech_type)]
                periodic_electricity = tech.periodic_electricity[v][p-1] * (1 - (tech.degradation_rate[v] * (t_ - t)))
                #periodic_thermal = tech.periodic_thermal[v][p-1] * (1 - (tech.degradation_rate[v] * (t_ - t)))
                total_electricity = var_value * periodic_electricity
                #total_thermal = var_value * periodic_thermal
                om_cost_per_unit = node.probability * tech.OMcost[v] * (tech.OMcostchangebyage[v] ** (t_ - t)) * (tech.OMcostchangebyyear[v] ** t) * (discount_factor ** t_)
                result = {
                    'NodeID': node.id,
                    'VariableName': var.VarName,
                    'TechType': tech_type,
                    'Version': v,
                    't': t,
                    't_': t_,
                    'p': p,
                    'VariableValue': var_value,
                    'PeriodicElectricityProductionPerUnit': periodic_electricity,
                    # 'PeriodicThermalProductionPerUnit': periodic_thermal,
                    'TotalElectricityProduction': total_electricity,
                    # 'TotalThermalProduction': total_thermal,
                    'OMCostPerUnit': om_cost_per_unit,
                    'OMCost': var_value * om_cost_per_unit
                }
                results.append(result)

        for key, var in node.v_Existing.items():
            var_value = var.X
            if var_value > 0:
                tech_type, v, t, t_ = key
                tech = node.FindAncestorFromDiff(t, t_).techNodeList[node.FindAncestorFromDiff(t, t_).tech_types.index(tech_type)]

                for p in node.stageSubterms:
                    if tech_type in ['solar', 'wind']:
                        periodic_electricity = tech.periodic_electricity[v][p-1] * (1 - (tech.degradation_rate[v] * (t_ - t)))
                        total_electricity = var_value * periodic_electricity
                    else:
                        periodic_electricity = 0
                        total_electricity = 0

                    om_cost_per_unit = node.probability * tech.OMcost[v] * (1 / tech.tree.numSubterms) * (tech.OMcostchangebyage[v] ** (t_ - t)) * (tech.OMcostchangebyyear[v] ** t) * (discount_factor ** t_)
                    result = {
                        'NodeID': node.id,
                        'VariableName': var.VarName,
                        'TechType': tech_type,
                        'Version': v,
                        't': t,
                        't_': t_,
                        'p': p,
                        'VariableValue': var_value,
                        'PeriodicElectricityProductionPerUnit': periodic_electricity,
                        'TotalElectricityProduction': total_electricity,
                        'OMCostPerUnit': om_cost_per_unit,
                        'OMCost': var_value * om_cost_per_unit
                    }
                    results.append(result)

        for key, var in node.v_Plus.items():
            var_value = var.X
            if var_value > 0:
                tech_type, v, t_ = key
                tech = node.techNodeList[node.tech_types.index(tech_type)]
                purchase_cost_per_unit = node.probability * tech.cost[v] * (discount_factor ** t_)
                purchase_result = {
                    'NodeID': node.id,
                    'VariableName': var.VarName,
                    'TechType': tech_type,
                    'Version': v,
                    't_': t_,
                    'VariableValue': var_value,
                    'CostPerUnit': purchase_cost_per_unit,
                    'TotalCost': var_value * purchase_cost_per_unit
                }
                purchase_sale_results.append(purchase_result)

        for key, var in node.v_Minus.items():
            var_value = var.X
            if var_value > 0:
                tech_type, v, t, t_ = key
                tech = node.FindAncestorFromDiff(t, t_).techNodeList[node.FindAncestorFromDiff(t, t_).tech_types.index(tech_type)]
                salvage_value_per_unit = -node.probability * tech.salvage_value[v] * (tech.depreciation_rate[v] * tech.lifetime[v] - (tech.depreciation_rate[v] * (t_ - t))) * (discount_factor ** t_)
                sale_result = {
                    'NodeID': node.id,
                    'VariableName': var.VarName,
                    'TechType': tech_type,
                    'Version': v,
                    't': t,
                    't_': t_,
                    'VariableValue': var_value,
                    'CostPerUnit': salvage_value_per_unit,
                    'TotalCost': var_value * salvage_value_per_unit
                }
                purchase_sale_results.append(sale_result)

    df_results = pd.DataFrame(results)
    df_purchase_sale = pd.DataFrame(purchase_sale_results)
    total_production_by_tech = df_results.groupby(['NodeID', 't_', 'TechType']).agg({'TotalElectricityProduction': 'sum', 'OMCost': 'sum'}).reset_index()
    summary_data = []
    node_t_combinations = total_production_by_tech[['NodeID', 't_']].drop_duplicates()

    for _, row in node_t_combinations.iterrows():
        summary_entry = {'NodeID': row['NodeID'], 't_': row['t_']}
        total_electricity_demand = sum(demand['electricity'][row['t_']])
        production_data = total_production_by_tech[(total_production_by_tech['NodeID'] == row['NodeID']) & (total_production_by_tech['t_'] == row['t_'])]

        for tech_type in ['solar', 'wind', 'natural_gas', 'battery']:
            tech_data = production_data[production_data['TechType'] == tech_type]
            total_electricity_production = tech_data['TotalElectricityProduction'].sum() if not tech_data.empty else 0
            total_om_cost = tech_data['OMCost'].sum() if not tech_data.empty else 0
            electricity_percentage = (total_electricity_production / total_electricity_demand * 100) if total_electricity_demand > 0 else 0
            summary_entry[f'{tech_type}_ElectricityPercentage'] = electricity_percentage
            summary_entry[f'{tech_type}_OMCost'] = total_om_cost

        total_om_cost = production_data['OMCost'].sum() if not production_data.empty else 0
        summary_entry['Total_OMCost'] = total_om_cost
        purchase_data = df_purchase_sale[(df_purchase_sale['NodeID'] == row['NodeID']) & (df_purchase_sale['t_'] == row['t_'])]
        total_purchase_cost = purchase_data[purchase_data['TotalCost'] > 0]['TotalCost'].sum() if not purchase_data.empty else 0
        summary_entry['Total_PurchaseCost'] = total_purchase_cost
        total_salvage_value = -purchase_data[purchase_data['TotalCost'] < 0]['TotalCost'].sum() if not purchase_data.empty else 0
        summary_entry['Total_SalvageValue'] = total_salvage_value
        summary_data.append(summary_entry)

    df_summary = pd.DataFrame(summary_data)
    purchase_data = df_purchase_sale[df_purchase_sale['TotalCost'] > 0]
    sale_data = df_purchase_sale[df_purchase_sale['TotalCost'] < 0]
    purchase_summary = purchase_data[purchase_data['TechType'].isin(['solar', 'wind', 'battery'])].groupby(['NodeID', 'TechType', 'Version']).agg({'VariableValue': 'sum', 'TotalCost': 'sum'}).reset_index()
    purchase_summary.rename(columns={'VariableValue': 'PurchasedQuantity', 'TotalCost': 'PurchaseCost'}, inplace=True)

    sale_summary = sale_data[sale_data['TechType'].isin(['solar', 'wind', 'battery'])].groupby(['NodeID', 'TechType', 'Version']).agg({'VariableValue': 'sum', 'TotalCost': 'sum'}).reset_index()
    sale_summary['SalvageValue'] = -sale_summary['TotalCost']
    sale_summary.rename(columns={'VariableValue': 'SoldQuantity'}, inplace=True)
    sale_summary = sale_summary[['NodeID', 'TechType', 'Version', 'SoldQuantity', 'SalvageValue']]
    purchase_sale_summary = pd.merge(purchase_summary, sale_summary, on=['NodeID', 'TechType', 'Version'], how='outer')
    purchase_sale_summary.fillna(0, inplace=True)
    purchase_sale_summary['Tech_Version'] = purchase_sale_summary['TechType'] + '_V' + purchase_sale_summary['Version'].astype(str)
    purchase = purchase_sale_summary.pivot(index='NodeID', columns='Tech_Version', values='PurchasedQuantity').fillna(0)
    purchase.columns = ['Purchased_' + col for col in purchase.columns]
    purchase_cost_pivot = purchase_sale_summary.pivot(index='NodeID', columns='Tech_Version', values='PurchaseCost').fillna(0)
    purchase_cost_pivot.columns = ['PurchaseCost_' + col for col in purchase_cost_pivot.columns]
    sales = purchase_sale_summary.pivot(index='NodeID', columns='Tech_Version', values='SoldQuantity').fillna(0)
    sales.columns = ['Sold_' + col for col in sales.columns]
    salvage_value = purchase_sale_summary.pivot(index='NodeID', columns='Tech_Version', values='SalvageValue').fillna(0)
    salvage_value.columns = ['SalvageValue_' + col for col in salvage_value.columns]

    dfs = [purchase, purchase_cost_pivot, sales, salvage_value]
    purchase_sale = reduce(lambda left, right: pd.merge(left, right, on='NodeID', how='outer'), dfs)
    purchase_sale.fillna(0, inplace=True)
    total_node_costs = df_summary.groupby('NodeID').agg({'Total_OMCost': 'sum', 'Total_PurchaseCost': 'sum', 'Total_SalvageValue': 'sum'}).reset_index()
    total_node_costs['Total_NodeCost'] = total_node_costs['Total_OMCost'] + total_node_costs['Total_PurchaseCost'] - total_node_costs['Total_SalvageValue']
    purchase_sale = pd.merge(purchase_sale, total_node_costs, on='NodeID', how='left')

    #with pd.ExcelWriter(f'Results.xlsx_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}') as writer:
    #    df_results.to_excel(writer, sheet_name='ProductionResults', index=False)
    #    total_production_by_tech.to_excel(writer, sheet_name='TotalProductionByTechType', index=False)
    #    df_purchase_sale.to_excel(writer, sheet_name='PurchaseAndSales', index=False)
    #    df_summary.to_excel(writer, sheet_name='Summary', index=False)
    #    purchase_sale.to_excel(writer, sheet_name='PurchaseSaleSummary', index=False)

    df_results.to_csv(f'ProductionResults_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv', index=False)
    total_production_by_tech.to_csv(f'TotalProductionByTechType_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv', index=False)
    df_purchase_sale.to_csv(f'PurchaseAndSales_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv', index=False)
    df_summary.to_csv(f'Summary_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv', index=False)
    purchase_sale.to_csv(f'PurchaseSaleSummary_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.csv', index=False)

def OptimizationModel(scenarioTree, emission_limits, demand, numStages, numSubperiods, numMultipliers, numSubterms, initial_tech, budget, grid_electricity_cost, discount_factor = 0.99):
    model = Model('MachineReplacement')
    model.setParam('OutputFlag', True)

    for node in scenarioTree.nodes:
        node.AddVariables(model)
        node.AddObjectiveCoefficients(model, grid_electricity_cost, discount_factor)
        node.InitializeCurrentTech(model, initial_tech)
        node.AddEmissionConstraints(model, emission_limits)
        node.AddBudgetConstraints(model, budget)
        node.AddDemandConstraints(model, demand)
        node.SetMinustoZeroConstraints(model)
        node.AddBalanceConstraints(model)
        node.AddBatteryCapacityConstraints(model)
        node.AddUpperBoundsForIP(model, demand)

    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 86400)
    #model.setParam('LogFile', f'Gurobi_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.txt')

    start_time = time.time()
    model.optimize()
    end_time = time.time()
    optimization_time = end_time - start_time

    lp_filename = f'MachineReplacement_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.lp'
    sol_filename = f'MachineReplacement_{numStages}_{numSubperiods}_{numSubterms}_{numMultipliers}.sol'
    #model.write(lp_filename)
    #model.write(sol_filename)

    #OutputProductionResults(model, scenarioTree, discount_factor, demand, numStages, numSubperiods, numSubterms, numMultipliers)
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
        'Objective Function Value':   model.objVal
    }

    return model_results

num_Stages_list = [2]
num_Subperiods_list = [2]
num_Subterms_list = [2000]
num_Multipliers_list = [2]

results = {}

for numStages in num_Stages_list:
    for numSubperiods in num_Subperiods_list:
        for numSubterms in num_Subterms_list:

            grid_electricity_cost = [0.132 for _ in range(numStages*numSubperiods+1)]
            emission_limits = [None for _ in range(numStages*numSubperiods)] + [0]
            budget = [None for _ in range(numStages*numSubperiods+1)]

            electricity_demand = [pd.read_excel('Demands.xlsx', sheet_name='Electricity Demand')['Demands'].tolist()[:numSubterms]]*(numStages*numSubperiods + 1)

            for numMultipliers in num_Multipliers_list:
                solar_initial = pd.read_excel('Solar Power.xlsx', sheet_name='Initial values')
                solar_advancements = pd.read_excel('Solar Power.xlsx', sheet_name=f'Advancements{numMultipliers}')
                solar_hourly_production = pd.read_excel('productiondata.xlsx', sheet_name='solar')
                solar_hourly_production_list = solar_hourly_production.T.values.tolist()
                solar_hourly_production_list = [sublist[:numSubterms] for sublist in solar_hourly_production_list]
                solar_hourly_production_list = solar_hourly_production_list[:2] + solar_hourly_production_list[4:]
                solar_hourly_production_list = [[max(0, x) for x in sublist] for sublist in solar_hourly_production_list]

                solar = TechnologyTree("solar", numSubperiods, numSubterms, 
                    lifetime=[solar_initial.iloc[4, i] for i in range(1, solar_initial.shape[1])], 
                    segment=[solar_initial.iloc[5, i] for i in range(1, solar_initial.shape[1])], 
                    initialCost=[solar_initial.iloc[0, i] for i in range(1, solar_initial.shape[1])], 
                    initialEfficiency=[solar_initial.iloc[1, i] for i in range(1, solar_initial.shape[1])], 
                    initialEmission=[solar_initial.iloc[2, i] for i in range(1, solar_initial.shape[1])], 
                    #periodic_electricity_production = [[float(num or '0') for num in each_version.split(',')[:numSubterms]] for each_version in [solar_initial.iloc[3, i] for i in range(1, solar_initial.shape[1])]],
                    periodic_electricity_production = solar_hourly_production_list,
                    degradation_rate=[solar_initial.iloc[6, i] for i in range(1, solar_initial.shape[1])], 
                    initialOMcost=[solar_initial.iloc[7, i] for i in range(1, solar_initial.shape[1])], 
                    OMcostchangebyage=[solar_initial.iloc[8, i] for i in range(1, solar_initial.shape[1])], 
                    depreciation_rate=[solar_initial.iloc[9, i] for i in range(1, solar_initial.shape[1])], 
                    initial_salvage_value=[solar_initial.iloc[10, i] for i in range(1, solar_initial.shape[1])],
                    OMcostchangebyyear=[solar_initial.iloc[12, i] for i in range(1, solar_initial.shape[1])])
                solar.ConstructByMultipliers(numStages, probabilities=[solar_advancements[col][0] for col in solar_advancements.columns if col != "Metrics"], costMultiplier=[solar_advancements[col][3] for col in solar_advancements.columns if col != "Metrics"], efficiencyMultiplier=[solar_advancements[col][4] for col in solar_advancements.columns if col != "Metrics"], emissionMultiplier=[solar_advancements[col][5] for col in solar_advancements.columns if col != "Metrics"])

                wind_initial = pd.read_excel('Wind Power.xlsx', sheet_name='Initial values')
                wind_advancements = pd.read_excel('Wind Power.xlsx', sheet_name=f'Advancements{1}')
                wind_hourly_production = pd.read_excel('productiondata.xlsx', sheet_name='wind')
                wind_hourly_production_list = wind_hourly_production.T.values.tolist()
                wind_hourly_production_list = [sublist[:numSubterms] for sublist in wind_hourly_production_list]

                wind = TechnologyTree("wind", numSubperiods, numSubterms, 
                    lifetime=[wind_initial.iloc[4, i] for i in range(1, wind_initial.shape[1])], 
                    segment=[wind_initial.iloc[5, i] for i in range(1, wind_initial.shape[1])], 
                    initialCost=[wind_initial.iloc[0, i] for i in range(1, wind_initial.shape[1])], 
                    initialEfficiency=[wind_initial.iloc[1, i] for i in range(1, wind_initial.shape[1])], 
                    initialEmission=[wind_initial.iloc[2, i] for i in range(1, wind_initial.shape[1])], 
                    #periodic_electricity_production = [[float(num or '0') for num in each_version.split(',')[:numSubterms]] for each_version in [wind_initial.iloc[3, i] for i in range(1, wind_initial.shape[1])]],
                    periodic_electricity_production = wind_hourly_production_list,
                    degradation_rate=[wind_initial.iloc[6, i] for i in range(1, wind_initial.shape[1])], 
                    initialOMcost=[wind_initial.iloc[7, i] for i in range(1, wind_initial.shape[1])], 
                    OMcostchangebyage=[wind_initial.iloc[8, i] for i in range(1, wind_initial.shape[1])], 
                    depreciation_rate=[wind_initial.iloc[9, i] for i in range(1, wind_initial.shape[1])], 
                    initial_salvage_value=[wind_initial.iloc[10, i] for i in range(1, wind_initial.shape[1])],
                    OMcostchangebyyear=[wind_initial.iloc[12, i] for i in range(1, wind_initial.shape[1])])
                wind.ConstructByMultipliers(numStages, probabilities=[wind_advancements[col][0] for col in wind_advancements.columns if col != "Metrics"], costMultiplier=[wind_advancements[col][3] for col in wind_advancements.columns if col != "Metrics"], efficiencyMultiplier=[wind_advancements[col][4] for col in wind_advancements.columns if col != "Metrics"], emissionMultiplier=[wind_advancements[col][5] for col in wind_advancements.columns if col != "Metrics"])

                battery_initial = pd.read_excel('Battery.xlsx', sheet_name='Initial values')
                battery_advancements = pd.read_excel('Battery.xlsx', sheet_name=f'Advancements{numMultipliers}')

                battery = TechnologyTree("battery", numSubperiods, numSubterms, 
                    lifetime=[battery_initial.iloc[4, i] for i in range(1, battery_initial.shape[1])], 
                    segment=[battery_initial.iloc[5, i] for i in range(1, battery_initial.shape[1])], 
                    initialCost=[battery_initial.iloc[0, i] for i in range(1, battery_initial.shape[1])], 
                    initialEfficiency=[battery_initial.iloc[1, i] for i in range(1, battery_initial.shape[1])], 
                    initialEmission=[battery_initial.iloc[2, i] for i in range(1, battery_initial.shape[1])], 
                    periodic_electricity_production = [[float(num or '0') for num in each_version.split(',')[:numSubterms]] for each_version in [battery_initial.iloc[3, i] for i in range(1, battery_initial.shape[1])]],
                    degradation_rate=[battery_initial.iloc[6, i] for i in range(1, battery_initial.shape[1])], 
                    initialOMcost=[battery_initial.iloc[7, i] for i in range(1, battery_initial.shape[1])], 
                    OMcostchangebyage=[battery_initial.iloc[8, i] for i in range(1, battery_initial.shape[1])], 
                    depreciation_rate=[battery_initial.iloc[9, i] for i in range(1, battery_initial.shape[1])], 
                    initial_salvage_value=[battery_initial.iloc[10, i] for i in range(1, battery_initial.shape[1])],
                    OMcostchangebyyear=[battery_initial.iloc[12, i] for i in range(1, battery_initial.shape[1])])
                battery.ConstructByMultipliers(numStages, probabilities=[battery_advancements[col][0] for col in battery_advancements.columns if col != "Metrics"], costMultiplier=[battery_advancements[col][3] for col in battery_advancements.columns if col != "Metrics"], efficiencyMultiplier=[battery_advancements[col][4] for col in battery_advancements.columns if col != "Metrics"], emissionMultiplier=[battery_advancements[col][5] for col in battery_advancements.columns if col != "Metrics"])

                scenarioTree = ScenarioTree([solar, wind, battery])

                initial_tech = [[solar_initial.iloc[11, i] for i in range(1, solar_initial.shape[1])],
                                [wind_initial.iloc[11, i] for i in range(1, wind_initial.shape[1])],
                                [battery_initial.iloc[11, i] for i in range(1, battery_initial.shape[1])]]

                key = f's{numStages}_p{numSubperiods}_t{numSubterms}_n{numMultipliers}'
                results[key] = OptimizationModel(scenarioTree, emission_limits, demand=electricity_demand, numStages=numStages, numSubperiods=numSubperiods, numMultipliers=numMultipliers, numSubterms=numSubterms, initial_tech=initial_tech, budget=budget, grid_electricity_cost=grid_electricity_cost)

                #df_results = pd.DataFrame.from_dict(results, orient='index')            #Output to an Excel file
                #df_results.reset_index(inplace=True)
                #df_results.rename(columns={'index': 'Scenario'}, inplace=True)
                #excel_filename = 'comparison_results.xlsx'
                #df_results.to_excel(excel_filename, index=False)
                #print(f"Results saved to {excel_filename}")