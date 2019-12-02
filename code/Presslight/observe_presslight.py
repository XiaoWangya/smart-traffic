#这里定义所有用于获取状态的函数
import traci
import numpy as np
#import sklearn.preprocessing as skp

#enc = skp.OneHotEncoder()

def cal_reward(state,state_n):
    '''
        return the difference of sum of state and new state
    '''
    reward = sum(state[0:4])*sum(state[4:8])-sum(state_n[0:4])*sum(state_n[4:8])
    return reward

def cal_reward_presslight(TrafficLightID):
    '''
        return the reward defined in paper 'presslight'
    '''
    pressure_intersection = get_pressure(TrafficLightID)
    return -pressure_intersection

def get_state(TrafficLightID,segment_number = 1):
    '''
        return the observation state of the given intersection
        It's a list with 8 elements [queue_length_from_4_directions, waiting_time_from_4_directions]
    '''
    
    All_links = [i[0][:2] for i in traci.trafficlight.getControlledLinks(TrafficLightID) if len(i)]
    state = []
    if segment_number ==1:
        for links in All_links:
            upper_stream = traci.lane.getLastStepVehicleNumber(links[0])
            down_stream = traci.lane.getLastStepVehicleNumber(links[1])
            state.append(upper_stream)
            state.append(down_stream)
    else:
        for links in All_links:
            seg_pdf = lane_segment(links[0],segment_number)
            upper_stream = get_lane_seg_number(seg_pdf,segment_number)
            down_stream = traci.lane.getLastStepVehicleNumber(links[1])
            state = state+upper_stream
            state.append(down_stream)
    trafficlight_phase = traci.trafficlight.getPhase(TrafficLightID)
    state.append(trafficlight_phase)
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
    All_links = [i[0][:2] for i in traci.trafficlight.getControlledLinks(TrafficLightID) if len(i)]
    pressure_movement = []
    for link in All_links:
        incoming_edge = traci.lane.getEdgeID(link[0])
        outgoing_edge = traci.lane.getEdgeID(link[1])
        pressure_movement.append(traci.edge.getLastStepOccupancy(incoming_edge)-traci.edge.getLastStepOccupancy(outgoing_edge))
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
        incoming_edge = list(sorted(set(incoming_edge)))
        outgoing_edge = list(sorted(set(outgoing_edge)))
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
    if n_parts>1:
        for id in VehicleID:
            current_position = traci.vehicle.getLanePosition(id)/lane_length
            if 1 - current_position:
                dict[id] = n_parts-np.floor(n_parts*current_position)#Closer to the intersection, rank higher.
            else:
                dict[id] = 1.0# in case that the current_position is at the start position
    else:
        for id in VehicleID:
            dict[id] = 1.0
    return segment_state

def get_lane_seg_number(dic_distribution,n_parts):
    '''
        return a list. the number of vehicles in each segmented part
    '''
    number_seg = []
    if n_parts:
        for i in range(n_parts):
            number_seg.append(list(dic_distribution.values()).count(i+1))
        
    else:print('error')
    return number_seg#ranked from 1 to n.

def get_all_waiting(TrafficLightID):
    '''
        return the waiting time at the given intersection
    '''
    EdgeList = get_edges(TrafficLightID)[0]
    #print(EdgeList)
    waiting_time = []
    for item in EdgeList:
        #print(traci.edge.getWaitingTime(item))
        waiting_time.append(traci.edge.getWaitingTime(item))
    return sum(waiting_time),waiting_time

def update_travel_time(dic_travel = dict()):
    '''
        return a dic of all vehicles's travel time till now
    '''
    All_vehicles = traci.vehicle.getIDList()
    for vehicle in All_vehicles:
        if vehicle not in dic_travel:
            dic_travel[vehicle] = [0,0,0]
            dic_travel[vehicle][0] = traci.simulation.getCurrentTime()
        dic_travel[vehicle][1] = traci.simulation.getCurrentTime()-dic_travel[vehicle][0]
        dic_travel[vehicle][2] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
    return dic_travel
    
def get_neighbor(TrafficLightID):
    '''
        return a dictionary consisting of neighbors id and the direction vector
    '''
    Position = traci.junction.getPosition(TrafficLightID)
    ALL_TL = traci.trafficlight.getIDList()
    Neighbor = dict()
    for id in ALL_TL:
        P1 = traci.junction.getPosition(id)
        vec = (np.asarray(P1)-np.asarray(Position))/200
        if np.linalg.norm(vec) == 1:
            Neighbor[id] = vec
    return Neighbor

def get_lane(TrafficLightID):
    '''
        Return a list of lanes for given traffic light ID
        [NS,EW,SN,WE] or [up,right,down,left]
    '''
    EdgeList = get_edges(TrafficLightID)[1]
    All_lanes = list(traci.lane.getIDList())
    LaneList_r = []
    LaneList = [0,0,0,0]
    for lane in All_lanes:
        if traci.lane.getEdgeID(lane) in EdgeList:
            LaneList_r.append(lane)
    for lane in LaneList_r:
        dire = np.asarray(traci.lane.getShape(lane)[0])-np.asarray(traci.lane.getShape(lane)[1])
        dire[dire<0]=-1
        dire[dire>0] = 1
        dire = list(dire)
        INDEX = np.int(dire.index(0)*1+1-dire[1-dire.index(0)])
        LaneList[INDEX] = lane
    return LaneList


def get_map_matrix(TrafficLightID):
    LaneList = get_lane(TrafficLightID)
    print(LaneList)
    VeID = traci.vehicle.getIDList()
    if LaneList:
        Num_Lanes = len(LaneList)
        Length_Lanes = np.int(np.ceil(np.max([traci.lane.getLength(LaneList[i]) for i in range(Num_Lanes)])))
        Block_lengths = 3
        Vmax = traci.lane.getMaxSpeed(LaneList[0])
        Map = np.zeros((Num_Lanes,Length_Lanes//Block_lengths+1,2))
        for vehicle in VeID:
            if traci.vehicle.getLaneID(vehicle) in LaneList:
                Index = LaneList.index(traci.vehicle.getLaneID(vehicle))
                Lane_Pos = np.int(traci.vehicle.getLanePosition(vehicle)//Block_lengths)
                Map[Index,Lane_Pos,0] = 1
                Map[Index,Lane_Pos,1] = traci.vehicle.getSpeed(vehicle)/Vmax
    return Map



