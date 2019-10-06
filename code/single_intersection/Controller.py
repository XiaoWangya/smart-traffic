#组成整体的交通控制器，包括具体的交通信号等控制细节
import RLmind as RL
import observe as ob
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt

# Global parameters
MEMORY_SIZE = 200
N_STATES = 8
N_ACTIONS = 2
EPSILON = 0.9
TARGET_UPDATE_INTERVAL = 10
BATCH_SIZE = 20
EPOCH = 10
MAX_STEP = 120
#Initial settings of sumo
sumobinary='YOUR_PATH/sumo.exe'
sumoConfig= 'YOUR_PATH.sumocfg'
sumoCmd=[sumobinary,"-c",sumoConfig,'--start']
#Main Part
T = RL.DQN(MEMORY_SIZE,N_STATES,N_ACTIONS,EPSILON,TARGET_UPDATE_INTERVAL,BATCH_SIZE)
reward_each_epoch = np.zeros(EPOCH)
for ep in range(EPOCH):
    traci.start(sumoCmd)
    TLID = traci.trafficlight.getIDList()
    #take a observation and make a decision
    traci.simulationStep()
    state = ob.get_state(TLID[0])
    for j in range(MAX_STEP):
        action = T.choose_action(torch.Tensor(state).view(1,8))
        #process the decision
        traci.trafficlight.setPhase(TLID[0],action*2)
        traci.simulationStep()
        #take a observation again
        state_n = ob.get_state(TLID[0])
        #recieve the reward
        reward = ob.cal_reward(state,state_n)
        #store the experience
        T.store_memory(state,action,state_n,reward)

        if T.memory_counter >= MEMORY_SIZE:
            T.update_network()
        
        reward_each_epoch[ep] += reward
        state = state_n
    traci.close()
plt.plot(reward_each_epoch)
plt.show()
