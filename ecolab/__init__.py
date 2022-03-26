import numpy as np
import numba
import matplotlib
from ecolab.agents import Rabbit, RHD_Status, Gender, AgentType
import matplotlib.pyplot as plt

@numba.jit  
def run_ecolab(env, agents, Niteration=360, earlystop=True):
    
    
    preg_prob = {11: 11/365,
                 0: 35/365,
                 1: 59/365,
                 2: 82/365,
                 3: 59/365,
                 4: 35/365,
                 5: 11/365,
                 6: 5/365,
                 7: 5/365,
                 8: 5/365,
                 9: 5/365,
                 10: 5/365}

    record=[] #TODO
    sus = []
    infected = []
    immune = []
    #male = []
    #female = []
    infant = []
    total =[]
    days=[]
    for it in range(Niteration):
        print("iteration: %g" %it)
        month = (it / 30) % 12  ## 0~11 ->  Jan~Dec
        for agent in agents:
            agent.other_daily_grow()
            if month > 3 and agent.type == AgentType.Adults and agent.gender == Gender.Female and agent.pregnancy_days == -1:
                agent.reproduct(agents, int(month), preg_prob)
            agent.move(env)
            agent.die()
            
        agents = [a for a in agents if a.death == False]
        for agent in agents:
            if agent.rhd_status == RHD_Status.Infected and agent.infected_days > 0:
                agent.infection(agents)
            newborn = agent.born_new_rabbit(agents, env)
            if newborn is not None:
                agents += newborn
        
        
        record.append({'susceptible agents': np.array([a.summary_vector() for a in agents if a.rhd_status == RHD_Status.Susceptible]),
                      'infected agents': np.array([a.summary_vector() for a in agents if a.rhd_status == RHD_Status.Infected]),
                      'immune agents': np.array([a.summary_vector() for a in agents if a.rhd_status == RHD_Status.Recoverd_Immune])})   
        
        #print("the number of whole agents: ", len(agents))
        #print("the number of Female adults", len([a for a in agents if a.type == AgentType.Adults and a.gender == Gender.Female]))
        #print("the number of Male adults", len([a for a in agents if a.type == AgentType.Adults and a.gender == Gender.Male]))
        #print("the infants of the agents", len([a for a in agents if a.type == AgentType.Infants]))
        # print("the adults of the agents", len([a for a in agents if a.type == AgentType.Adults]))
        #print("the number of susceptible agents: ", len(record[it]['susceptible agents']))   
        #print("the number of infected agents: ", len(record[it]['infected agents']))
        #print("the number of immune agents:", len(record[it]['immune agents']))
        #print("===========================================")    
        days.append(it)
        sus.append(len(record[it]['susceptible agents']))
        infected.append(len(record[it]['infected agents']))
        immune.append(len(record[it]['immune agents']))
        total.append(len(agents))
        infant.append(len([a for a in agents if a.type == AgentType.Infants]))                
        
        if earlystop:
            if len(agents)==0: break
    
    return record, sus, infected, immune, total, infant       

    