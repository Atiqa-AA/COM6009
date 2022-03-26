
"""
@author: Eurus.T
"""
from enum import Enum
import numpy as np
import numba
import random



INFANT_NATURE_MOTALITY_RATE = 0.0002
ADULT_NATURE_MOTALITY_RATE = 0.0013
#RHD_INFECTION_PROB = 1.27 * 10e-4
#PREGNANT_PROB = 0.4
MAX_CAPCITY = 40 # max capcity for each grid

class RHD_Status(Enum):
    """
    RHD_Status
    """
    Susceptible = 'susceptible'
    Infected = 'infected'
    Recoverd_Immune = 'immune'
    
 
class AgentType(Enum):
     """
     The type of rabbit according to different age
     """   
     Infants = 'infants'
     Adults = 'adults'
     
class Gender(Enum):
    """
    """
    Male = 1
    Female = 0
    
class Rabbit:
    """
    Base class for all types of rabbit
    """
    pregnancy_days = -1
    infected_days = -1
    maxage = 3650
    def __init__(self,position,age,infected=False):
        self.position = position
        self.age = age
        if np.random.randint(0,2) == 1: ## 1 is male, 0 is female
            self.gender = Gender.Male
        else:
            self.gender = Gender.Female
        self.death = False
        if not infected :
            self.rhd_status = RHD_Status.Susceptible
        else :
            self.rhd_status = RHD_Status.Infected
            self.infected_days = 1
        if age < 90:
            self.speed = 0
            self.type = AgentType.Infants
        else :
            self.speed = 5
            self.type = AgentType.Adults
            # if self.gender == Gender.Female:
            #     self.reproduct_ablity = True
    @numba.jit         
    def try_move(self, newposition, env):
        if env.check_position(newposition):
            self.position = newposition
    
    @numba.jit        
    def move(self, env):
        if self.speed > 0:
            d = np.random.rand()*2*np.pi
            delta = np.round(np.array([np.cos(d),np.sin(d)]) * np.random.randint(0, self.speed + 1)) 
            self.try_move(self.position + delta, env)   
    
    @numba.jit    
    def die(self):
        
        if self.rhd_status == RHD_Status.Infected and self.infected_days > 0: ##infected death begins on the 2nd infected day
            # self.death = (np.random.rand() > 0.1 * self.infected_days)
            self.death = (np.random.rand() < np.exp(-self.infected_days))
        if self.death == True: return 
        # nature death
        if self.age > self.maxage: 
            self.death = True
            return 
        if self.type == AgentType.Infants:
            self.death =  (np.random.rand() < INFANT_NATURE_MOTALITY_RATE)
        else:
            self.death = (np.random.rand() < ADULT_NATURE_MOTALITY_RATE)
    
    @numba.jit         
    def get_nearby_rabbit(self, position, agents):
        return [a for a in agents if (np.abs(a.position[0] - position[0]) < 2 and np.abs(a.position[1] - position[1]) < 2)]
    
    @numba.jit 
    def infection(self, agents):
        cnt = 0
        for a in agents:
            if (a.type == AgentType.Adults and a.rhd_status == RHD_Status.Susceptible and (a.position == self.position).all()):
                print("one rabbbit get infected")
                a.infected_days = 0
                a.rhd_status = RHD_Status.Infected
                cnt += 1
            if cnt == 2: break
        # nearby_susceptible_agents = [a for a in agents if (a.rhd_status == RHD_Status.Susceptible and (a.position == self.position).all())]
        # # nearby_infected_agents = [a for a in agents if a.rhd_status == RHD_Status.Infected and a.infected_days > 0 and np.abs(a.position[0] - self.position[0]) < 2 and np.abs(a.position[1] - self.position[1]) < 2]
        # if len(nearby_susceptible_agents) > 0:
        #     chosen_agent = random.sample(nearby_susceptible_agents, 2)
        #     for a in chosen_agent:
        #         a.infected_days = 0   ## initial day of infection
        #         a.rhd_status = RHD_Status.Infected
            
            # print("one rabbit get infected")
            #np.abs(a.position[0] - self.position[0]) < 2 and np.abs(a.position[1] - self.position[1]) < 2)

    # @numba.jit 
    def reproduct(self, agents, month, preg_prob):
        
        same_grid_male = [a for a in agents if (a.death == False and a.type == AgentType.Adults and a.gender == Gender.Male and (a.position == self.position).all())]
        #prob = np.random.uniform(0.06, 0.44)
        prob = preg_prob.get(month)
        if len(same_grid_male) > 0 and np.random.rand() <= prob:
            self.pregnancy_days = 0
            # print("one female adult rabbit get pregant")
    
    
    @numba.jit             
    def born_new_rabbit(self,agents,env):
        if self.pregnancy_days > 30:
            self.pregnancy_days = -1
            if len(agents) / env.shape[0] < MAX_CAPCITY:
                litter_num = np.random.randint(2, 8)
                newborn =[]
                i = 0
                for i in range(litter_num):
                    newborn.append(Rabbit(position = self.position, age = 1))
                    i += 1
                 # print("here is newborn , number:", len(newborn))
                return newborn
        
        return None
            
    @numba.jit 
    def other_daily_grow(self): # run this function first in each day loop
        self.age +=1
        if self.pregnancy_days > -1:
            self.pregnancy_days += 1
        
        if self.rhd_status == RHD_Status.Infected:
            self.infected_days += 1
        
        if self.infected_days > 10:
            # print("here is a new immune rabbit!")
            self.rhd_status = RHD_Status.Recoverd_Immune
            self.infected_days = -1
            
            
        if self.age > 90:
            self.speed = 5
            self.type = AgentType.Adults
            
    @numba.jit        
    def summary_vector(self):
        return [self.position[0], self.position[1], self.type, self.rhd_status]