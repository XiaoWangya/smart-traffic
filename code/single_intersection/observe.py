#这里定义所有用于获取状态的函数
import traci

def cal_reward(state,state_n):
    '''
        return the difference of sum of state and new state
    '''
    reward = sum(state[0:4])*sum(state[4:8])-sum(state_n[0:4])*sum(state_n[4:8])
    return reward

def get_incoming_edge(TrafficLightID):
    '''
        return the incoming edges of the given traffic light
        listed with [south,east,north,west]
    '''
    incoming_edge = []
    for link in range(len(traci.trafficlight.getControlledLinks(TrafficLightID))):
        if len(traci.trafficlight.getControlledLinks(TrafficLightID)[link]):
            incoming_edge.append(traci.lane.getEdgeID(traci.trafficlight.getControlledLinks(TrafficLightID)[link][0][0]))
        incoming_edge = sorted(set(incoming_edge),key=incoming_edge.index)#不损失顺序排序并消去重复元素    
    return incoming_edge 

def get_state(TrafficLightID):
    '''
        return the observation state of the given intersection
        It's a list with 8 elements [queue_length_from_4_directions, waiting_time_from_4_directions]
    '''
    incoming_edge = get_incoming_edge(TrafficLightID)
    state_queue_length = []
    state_waiting_time = []
    for edge in incoming_edge:
        if traci.edge.getWaitingTime(edge):
            state_waiting_time.append(traci.edge.getWaitingTime(edge))
        else:
            state_waiting_time.append(0)
        if traci.edge.getLastStepHaltingNumber(edge):
            state_queue_length.append(traci.edge.getLastStepHaltingNumber(edge))
        else:
            state_queue_length.append(0)
    state = state_queue_length+state_waiting_time
    return state
