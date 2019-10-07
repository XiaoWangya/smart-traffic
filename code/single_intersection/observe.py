#这里定义所有用于获取状态的函数
import traci
import numpy as np

def cal_reward(state,state_n):
    '''
        return the difference of sum of state and new state
    '''
    reward = sum(state[0:4])*sum(state[4:8])-sum(state_n[0:4])*sum(state_n[4:8])
    return reward

def cal_reward_presslight(pressure_intersection):
    '''
        return the reward defined in paper 'presslight'
    '''
    return -pressure_intersection

def get_state(TrafficLightID):
    '''
        return the observation state of the given intersection
        It's a list with 8 elements [queue_length_from_4_directions, waiting_time_from_4_directions]
    '''
    incoming_edge = get_edges(TrafficLightID)[1]
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

def get_pressure(TrafficLightID):
    '''
        return the pressure of the movement in the given intersection
        As defined in paper "PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network"
        the pressure of a movement is defined as the difference of vehicle density between the incoming and outgoing lanes
        the pressure of a movement (l,m) is denoted by 
            w(l,m) = dens(l)-dens(m)
        where den(-) is the density of the given lane, which is the remainder of the observed number of vehicle and the maximum 
        permissible vehicles.
    '''
    incoming_edge, outgoing_edge = get_edges(TrafficLightID)[1:]
    
    if len(incoming_edge) and len(incoming_edge) == len(outgoing_edge):
        pressure_movement = []
        for i in range(2):
            pressure_movement.append(traci.edge.getLastStepOccupancy(incoming_edge[i])-traci.edge.getLastStepOccupancy(outgoing_edge[1-i]))
            pressure_movement.append(traci.edge.getLastStepOccupancy(incoming_edge[i+2])-traci.edge.getLastStepOccupancy(outgoing_edge[3-i]))
    else:
        print("error")
        return 0
    pressure_intersection = np.abs(sum(pressure_movement))
    return pressure_intersection

def get_edges(TrafficLightID):
    '''
    return a list of edges for the given intersection, incoming edges and outgoing edges
    listed with [west,east,south,north]_incoming + [west,east,south,north]_outgoing
    '''
    TLIDList = traci.trafficlight.getIDList()
    if TrafficLightID in TLIDList:
        edge_list = []
        incoming_edge = []
        outgoing_edge = []
        for link in range(len(traci.trafficlight.getControlledLinks(TrafficLightID))):
            if len(traci.trafficlight.getControlledLinks(TrafficLightID)[link]):
                incoming_edge.append(traci.lane.getEdgeID(traci.trafficlight.getControlledLinks(TrafficLightID)[link][0][0]))
                outgoing_edge.append(traci.lane.getEdgeID(traci.trafficlight.getControlledLinks(TrafficLightID)[link][0][1]))
        #排序并消去重复元素
        #
        incoming_edge = sorted(set(incoming_edge))
        outgoing_edge = sorted(set(outgoing_edge))
        edge_list = incoming_edge + outgoing_edge
    else:
        print('please enter the correct intersection id')
    return edge_list,incoming_edge,outgoing_edge

def lane_segment(lane_id,n_parts):
    '''
        return a dictionary of id-class pairs
        classify the vehicles into {1,2,3} 3 classes
        1 -> closest to the intersection
        3 -> farthest to the intersection
    '''
    VehicleID = traci.lane.getLastStepVehicleIDs(lane_id)
    segment_state = dict()
    lane_length = traci.lane.getLength(lane_id)
    for id in VehicleID:
        current_position = traci.vehicle.getLanePosition(id)/lane_length
        if 1 - current_position:
            dict[id] = n_parts-np.floor(n_parts*current_position)#Closer to the intersection, rank higher.
        else:
            dict[id] = 1.0# in case that the current_position is at the start position
    return segment_state

def get_lane_seg_number(dic_distribution,n_parts):
    '''
        return the number of vehicles in each segmented part
    '''
    number_seg = []
    if n_parts:
        for i in range(n_parts):
            number_seg.append(list(dic_distribution.values()).count(i+1))
        
    else:print('error')
    return number_seg#ranked from 1 to n.
