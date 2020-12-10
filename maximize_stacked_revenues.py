# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:44:13 2020

@author: konsteriotis
"""

class battery_portfolio:
    def __init__(self,ns,dam_participation,rm_participation,fm_participation,bm_participation,dam_prices,rm_up_prices,rm_dn_prices,fmp_prices,fmq_prices,bm_up_prices,bm_dn_prices,
                 S_max, E_max, roundtrip_eff, initial_soc):
        # User Input
        self.ns = ns  # Number of BSUs (Battery Storage Units)
        self.dam_participation = dam_participation # (1 if the BSUs participate in the DAM, 0 otherwise)
        self.rm_participation = rm_participation # (1 if the BSUs participate in the RM, 0 otherwise)
        self.fm_participation = fm_participation # (1 if the BSUs participate in the FM, 0 otherwise)
        self.bm_participation = bm_participation # (1 if the BSUs participate in the BM, 0 otherwise)
        self.dam_prices = dam_prices # DAM prices in MW/euro
        self.rm_up_prices = rm_up_prices # RM up prices in MW/euro
        self.rm_dn_prices = rm_dn_prices # RM dn prices in MW/euro
        self.fmp_prices = fmp_prices # FM active power prices in MW/euro
        self.fmq_prices = fmq_prices # FM reactive power prices in MVAr/euro
        self.bm_up_prices = bm_up_prices # BM upward prices in MW/euro
        self.bm_dn_prices = bm_dn_prices # BM downward prices in MW/euro
        self.S_max = S_max # BSUs' active power capacity in kW
        self.E_max = E_max # BSUs' energy capacity in kWh
        self.roundtrip_eff = roundtrip_eff # BSUs' rountrip efficiencies (0 to 1, unitless)
        self.E0 = initial_soc # BSUs' initial state of energy as a percentage of energy capacity (0 to 1, unitless)
    
    def maximize_stacked_revenues(self):
        from IPython import get_ipython
        get_ipython().magic('reset -sf')

        import pyomo.environ as pyo
        from pyomo.environ import value
        # from pyomo.opt import SolverFactory
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.dam_prices = np.array(self.dam_prices)
        self.rm_up_prices = np.array(self.rm_up_prices)
        self.rm_dn_prices = np.array(self.rm_dn_prices)
        self.fmp_prices = np.array(self.fmp_prices)
        self.fmq_prices = np.array(self.fmq_prices)
        self.bm_up_prices = np.array(self.bm_up_prices)
        self.bm_dn_prices = np.array(self.bm_dn_prices)
        self.S_max = 0.001*np.array(self.S_max)
        self.E_max = 0.001*np.array(self.E_max)
        self.roundtrip_eff = np.array(self.roundtrip_eff)
        self.E0 = np.array(self.E0)
        
        nt = len(self.dam_prices) # Number of timeslots
        ch_eff = np.sqrt(self.roundtrip_eff) # Charging efficiency
        dis_eff = np.sqrt(self.roundtrip_eff) # Discharging efficiency
        
        
        # if self.bm_participation==0:
            # self.bm_up_prices = np.zeros([nt])
            # self.bm_dn_prices = np.zeros([nt])
            
        MM = 4
        A_theta = np.zeros((MM,self.ns))
        B_theta = np.zeros((MM,self.ns))
        C_theta = np.zeros((MM,self.ns))


        for k in range(0,MM):
            for i in range(0,self.ns):
                A_theta[k,i] = np.cos(((-1+2*(k+1))*np.pi)/MM) 
                B_theta[k,i] = np.sin(((-1+2*(k+1))*np.pi)/MM) 
                C_theta[k,i] = self.S_max[i]*np.cos(np.pi/MM)

        # Create Model
        model = pyo.ConcreteModel()

        # Sets
        T = np.array([t for t in range(0,nt)])
        S = np.array([i for i in range(0,self.ns)])
        M = np.array([k for k in range(0,MM)])
        model.T = pyo.Set(initialize=T)
        model.S = pyo.Set(initialize=S)
        model.M = pyo.Set(initialize=M)
        
        # Variables
        model.dis = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.ch = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.S, model.T, domain=pyo.Binary)
        model.rup = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.rdn = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.pup = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.pdn = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.soe = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
        model.pess = pyo.Var(model.S, model.T)
        model.qess = pyo.Var(model.S, model.T)
        
        # Constraints
        model.soeUB = pyo.Constraint(model.S, model.T, rule = lambda model, i, t: model.soe[i,t]<=self.E_max[i])
        model.qessUB = pyo.Constraint(model.S, model.T,  rule = lambda model, i, t: model.qess[i,t]<=self.S_max[i])
        model.qessLB = pyo.Constraint(model.S, model.T,  rule = lambda model, i, t: model.qess[i,t]>=-self.S_max[i])
        
        # model.pupBound = pyo.Constraint(model.S, model.T,  rule = lambda model, i, t: model.pup[i,t]<=self.S_max[i])
        # model.pdnBound = pyo.Constraint(model.S, model.T,  rule = lambda model, i, t: model.pdn[i,t]<=self.S_max[i])
        model.disUB = pyo.Constraint(model.S, model.T, rule=lambda model,i,t:model.dis[i,t]<=self.S_max[i]*model.u[i,t])

        model.chUB =  pyo.Constraint(model.S, model.T, rule = lambda model, i, t: model.ch[i,t] <= self.S_max[i]*(1-model.u[i,t]))

        model.rupUB = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.rup[i,t]<=self.S_max[i] + model.ch[i,t] - model.dis[i,t])

        model.rdnUB = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.rdn[i,t] <=self.S_max[i] - model.ch[i,t] + model.dis[i,t])

        model.pupUB = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.pup[i,t] <=self.S_max[i] + model.ch[i,t] - model.dis[i,t] - model.rup[i,t])

        model.pdnUB = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.pdn[i,t] <=self.S_max[i] + model.dis[i,t] - model.ch[i,t] - model.rdn[i,t])

        def SOE_rule(model, i, t):
            if t==0:
                return model.soe[i,t] == self.E0[i]*self.E_max[i] + ch_eff[i]*(model.ch[i,t]+model.pdn[i,t]) - (model.dis[i,t]+model.pup[i,t])*(1/dis_eff[i])
            else:
                return model.soe[i,t] == model.soe[i, t-1]+ch_eff[i]*(model.ch[i,t]+model.pdn[i,t]) - (model.dis[i,t]+model.pup[i,t])*(1/dis_eff[i])


        model.socDynamics = pyo.Constraint(model.S, model.T, rule=SOE_rule)

        model.rdnCapability = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.soe[i,t] + model.rdn[i,t]*ch_eff[i] <= self.E_max[i])

        model.rupCapability = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.soe[i,t] - model.rup[i,t]*(1/dis_eff[i]) >= 0)

        model.lastTimeslot = pyo.Constraint(model.S, rule=lambda model, i: model.soe[i,nt-1] >= self.E0[i]*self.E_max[i])

        model.pessDefinition = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.pess[i,t] == model.dis[i,t] - model.ch[i,t] + model.pup[i,t] - model.pdn[i,t])

        model.lineCircleLinearization = pyo.Constraint(model.S, model.T, model.M, rule=lambda model, i, t, k: A_theta[k,i]*model.pess[i,t] + B_theta[k,i]*model.qess[i,t]<=C_theta[k,i])

        # Market Participation
        if self.dam_participation==0:
            model.fix_dis = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.dis[i,t]==0)
            model.fix_ch = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.ch[i,t]==0)
    
        if self.rm_participation==0:
            model.fix_rup = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.rup[i,t]==0)
            model.fix_rdn = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.rdn[i,t]==0)
    
        if self.fm_participation==0:
            model.fix_qess = pyo.Constraint(model.S, model.T, rule=lambda model, i, t:model.qess[i,t]==0)

        if self.fm_participation==0 and self.bm_participation==0:     
            model.fix_pup = pyo.Constraint(model.S, model.T, rule=lambda model, i, t:model.pup[i,t]==0)
            model.fix_pdn = pyo.Constraint(model.S, model.T, rule=lambda model, i, t:model.pdn[i,t]==0)
    
        # Objective
        if self.dam_participation==1:        
            dam_profits = sum(self.dam_prices[t]*sum((model.dis[i,t]-model.ch[i,t]) for i in model.S) for t in model.T)
        else:
            dam_profits = 0

        if self.rm_participation==1:
            rm_profits = sum(self.rm_up_prices[t]*sum(model.rup[i,t] for i in model.S) for t in model.T)+sum(self.rm_dn_prices[t]*sum(model.rdn[i,t] for i in model.S) for t in model.T)
        else:
            rm_profits = 0
    
        if self.fm_participation==1:
            fm_profits = sum(sum(self.fmp_prices[i,t]*(model.pup[i,t]-model.pdn[i,t]) for i in model.S) for t in model.T)+sum(sum(self.fmq_prices[i,t]*model.qess[i,t] for i in model.S) for t in model.T)
        else:
            fm_profits = 0
    
        if self.bm_participation==1:
            bm_profits = sum(self.bm_up_prices[t]*sum(model.pup[i,t] for i in model.S) for t in model.T)-sum(self.bm_dn_prices[t]*sum(model.pdn[i,t] for i in model.S) for t in model.T)
        else:
            bm_profits = 0    
            
        # DIS = [[0.5, 0, 0.5],[0.5, 0, 0.5]] 
        # CH = [[0, 0.5, 0],[0, 0.5, 0]]
        # model.check_dis = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.dis[i,t]==DIS[i][t])
        # model.check_ch = pyo.Constraint(model.S, model.T, rule=lambda model, i, t: model.ch[i,t]==CH[i][t])
        model.obj = pyo.Objective(expr = dam_profits + rm_profits + fm_profits + bm_profits, sense=pyo.maximize)    
        
        opt = pyo.SolverFactory('cplex')
        
        result = opt.solve(model, tee=True)
        
        Profits = value(model.obj)
        
        discharge = [[1000*value(model.dis[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        charge = [[1000*value(model.ch[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        
        # dam_schedule = np.zeros([self.ns,nt])

        dam_schedule = [[discharge[i][t]-charge[i][t] for t in range(0,nt)] for i in range(0,self.ns)]
        rup_commitment = [[1000*value(model.rup[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        rdn_commitment = [[1000*value(model.rdn[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        
        pup = [[1000*value(model.pup[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        pdn = [[1000*value(model.pdn[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        
        pflexibility = [[pup[i][t]-pdn[i][t] for t in range(0,nt)] for i in range(0,self.ns)]
        
        qflexibility = [[1000*value(model.qess[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        
        soc = [[1000*value(model.soe[i,t]) for t in range(0,nt)] for i in range(0,self.ns)]
        DAM_profits =  sum(self.dam_prices[t]*sum(0.001*dam_schedule[i][t] for i in model.S) for t in model.T)
        RM_profits = sum(self.rm_up_prices[t]*sum(0.001*rup_commitment[i][t] for i in model.S) for t in model.T)+sum(self.rm_dn_prices[t]*0.001*sum(rdn_commitment[i][t] for i in model.S) for t in model.T)
        FM_profits = sum(sum(self.fmp_prices[i][t]*0.001*pflexibility[i][t] for i in model.S) for t in model.T)+sum(sum(self.fmq_prices[i][t]*0.001*qflexibility[i][t] for i in model.S) for t in model.T)
        BM_profits = sum(self.bm_up_prices[t]*sum(0.001*pup[i][t] for i in model.S) for t in model.T)-sum(self.bm_dn_prices[t]*sum(0.001*pdn[i][t] for i in model.S) for t in model.T)
        
       
        x_data = np.arange(1,nt+1)

        A1 = plt.figure(1) 
        y1_data = np.asarray(dam_schedule)
        for j in range(0,self.ns):
            plt.plot(x_data,y1_data[j], label="Battery "+str(j+1))
        plt.title("DAM Scehdule")
        plt.xlabel("Time (h)")
        plt.ylabel("Power (kW)")
        plt.xlim([1,nt])
        plt.ylim([-1000*self.E_max[j],1000*self.E_max[j]])
        lgd = plt.legend(frameon=False)
        plt.grid()
        A1.show()               

        A2 = plt.figure(2)
        y2_data = np.asarray(rup_commitment)
        for j in range(0,self.ns):
            plt.plot(x_data,y2_data[j], label="Battery "+str(j+1))
        plt.title("Up Reserve Commitment")
        plt.xlabel("Time (h)")
        plt.ylabel("Power (kW)")
        plt.xlim([1,nt])
        plt.ylim([0,1000*self.E_max[j]])
        lgd = plt.legend(frameon=False)
        plt.grid()
        A2.show
        
        A3 = plt.figure(3)
        y3_data = np.asarray(rdn_commitment)
        for j in range(0,self.ns):
            plt.plot(x_data,y3_data[j], label="Battery "+str(j+1))
        
        plt.title("Down Reserve Commitment")
        plt.xlabel("Time (h)")
        plt.ylabel("Power (kW)")
        plt.xlim([1,nt])
        plt.ylim([0,1000*self.E_max[j]])
        lgd = plt.legend(frameon =False)
        plt.grid()
        A3.show()
        
        A4 = plt.figure(4)
        y4_data = np.asarray(pflexibility)
        for j in range(0,self.ns):
            plt.plot(x_data,y4_data[j], label = "Battery "+str(j+1))
        
        plt.title("Active Power Flexibility Provision")
        plt.xlabel("Time (h)")
        plt.ylabel("Power (kW)")
        plt.xlim([1,nt])
        plt.ylim([-1000*self.E_max[j],1000*self.E_max[j]])
        lgd = plt.legend(frameon=False)
        plt.grid()
        A4.show()
        
        A5 = plt.figure(5)
        y5_data = np.asarray(qflexibility)
        for j in range(0,self.ns):
            plt.plot(x_data,y5_data[j], label = "Battery "+str(j+1))
        
        plt.title("Reactive Power Flexibility Provision")
        plt.xlabel("Time (h)")
        plt.ylabel("Power (kW")
        plt.xlim([1,nt])
        plt.ylim([-1000*self.E_max[j],1000*self.E_max[j]])
        lgd = plt.legend(frameon=False)
        plt.grid()
        A5.show()
        
        A6 = plt.figure(6)
        xm = np.arange(1,5)
        market_profits = [DAM_profits, RM_profits, FM_profits, BM_profits]
        plt.bar(xm, market_profits)
        plt.title("Profits per Market")
        plt.ylabel("Financial Balance (\u20ac)")
        plt.xticks(xm, ["DAM", "RM", "FM", "BM"])
        plt.grid()
        A6.show()
        
        # y2_data = np.asarray(rup_commitment)
        # B = plt.figure(2)
        # plt.plot(x_data,y2_data.T)
        # B.show()
        return [Profits,pup,pdn,dam_schedule,rup_commitment,rdn_commitment,pflexibility,qflexibility,soc,DAM_profits,RM_profits,FM_profits,BM_profits]