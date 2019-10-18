import Presslight_agent as model
import observe_presslight as ob
import torch
import numpy as np
import traci
from flow_generate import Flow_generator as flow
import matplotlib.pyplot as plt
import random

# Global parameters
random.seed(1915)
MEMORY_SIZE = 1000
N_STATES = 9
N_ACTIONS = 2
EPSILON = 0.5
TARGET_UPDATE_INTERVAL = 2
BATCH_SIZE = 100
EPOCH = 30
MAX_STEP = 1000
DUR = 5

#Initial settings of sumo
sumobinary = PATH_SUMO
sumoConfig= PATH_CONFIG
sumoCmd=[sumobinary,"-c",sumoConfig,'--start']


#Main Part
if __name__ == "__main__":
    traci.start(sumoCmd)
    TLlist = traci.trafficlight.getIDList()
    print(TLlist)
    Agents = ['agent'+'%s'%a for a in range(len(TLlist))]
    reward_each_epoch = [[0] for i in range(len(TLlist))]
    global_reward = []
    num_agents = len(Agents)
    for agent in Agents:
        locals()[agent] = model.DQN(MEMORY_SIZE,N_STATES,N_ACTIONS,EPSILON,TARGET_UPDATE_INTERVAL,BATCH_SIZE)
    travel_dic = dict()
    traci.close()
    
    for ep in range(EPOCH):
        traci.start(sumoCmd)
        print('iteration %d'%ep)
        if vars()[Agents[0]].memory_counter >= MEMORY_SIZE:
            print('loss : %f'%(vars()[Agents[0]].loss_records)[-1])
        for step in range(0,MAX_STEP,DUR):
            for s in range(DUR):
                traci.simulationStep()
            travel_dic = ob.update_travel_time(travel_dic)
            for agent in Agents:
                LightID = TLlist[int(agent[-1])]
                state = ob.get_state(LightID)
                action = vars()[agent].choose_action(torch.Tensor(state).view(1,N_STATES))
                traci.trafficlight.setPhase(LightID,action*2)
                state_n = ob.get_state(LightID)
                reward = ob.cal_reward_presslight(LightID)
                vars()[agent].reward.append(reward)
                #store the experience
                vars()[agent].store_memory(state,action,state_n,reward)
                if vars()[agent].memory_counter >= MEMORY_SIZE:
                    for k in range(DUR):
                        vars()[agent].update_network()
                if vars()[agent].loss_records and vars()[agent].loss_records[-1] <= 0.05 or vars()[agent].memory_counter>=MEMORY_SIZE*2:
                    vars()[agent].epsilon = 0.9
                reward_each_epoch[int(agent[-1])][-1] += reward
                state = state_n
                vars()[agent].waitingtime.append(ob.get_all_waitingtime(LightID)[0])
        for agent in Agents:
            vars()[agent].ac_reward.append(sum(vars()[agent].reward))
            vars()[agent].reward = []    
        traci.close()
        if (ep+1)%5 == 0:
            np.save(r'travel_dic.npy',travel_dic)#save the travel time for data analysis
            
for i in range(3*num_agents):
    plt.subplot(num_agents,3,i//3*3+1)
    plt.plot(vars()[Agents[i//3]].loss_records)
    plt.subplot(num_agents,3,i//3*3+2)
    plt.plot(vars()[Agents[i//3]].ac_reward)
    plt.subplot(num_agents,3,i//3*3+3)
    plt.plot(vars()[Agents[i//3]].waitingtime)
plt.show()
