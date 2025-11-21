from swarm_rescue.solutions.my_drone_motionless import MyDroneMotionless
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.solutions.my_drone_lidar_communication import MyDroneLidarCommunication
from typing import Optional, Tuple, Dict, Any, List

from numpy import pi, cos, sin

from swarm_rescue.solutions.my_drone_driver import MyDroneDriver
from typing import Optional, Tuple, Any


class MyDroneEval(MyDroneLidarCommunication):
    def define_message_for_all(self) -> Tuple[Optional[int], Tuple[Any, Any]]:
        """
        Define the message, the drone will send to and receive from other surrounding drones.

        Returns:
            Tuple[Optional[int], Tuple[Any, Any]]: The message data.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        print(self.gps_values())
        #print(self.lidar_possible_paths())
        return msg_data 

class MyDroneEval1(MyDroneMotionless):
    """
    Evaluation drone class that inherits from MyDroneRandom.

    This class can be extended to implement custom evaluation logic.
    """
    #LiDAR
    def lidar_possible_paths(self) -> List:
        '''This function's purpose is to collect Lidar data, analyse it, and output a list containing potential areas to explore at that timestep. all rays from -180 to -135 and 135 to 180 degrees are discarded because, in theory, the drone is coming from behind himself, so this is not a possible path. the list of possible paths is list containing tuples of the angle to explore, along with a boolean value corresponding to wether this node has been explored or not. All booleans will be False in this functin but should be changed later (DFS algorithm) '''
        list_possible_area=[]
        #list_test=[] test list that stores the min and max rays that help populate the list_possible_area list
        min_ray=-3/4*pi,0
        max_ray=0,0
        ray_ini=False
        minimal_distance=285
        coords=self.gps_values()
        angle=self.measured_compass_angle()
        step_forward=50

        if not self.lidar_is_disabled():
            lidar_data=self.lidar_values()
            ray_angles = self.lidar_rays_angles()
            for i in range (22,len(lidar_data)-22):
                if lidar_data[i]>minimal_distance: 
                    if lidar_data[i-1]<=minimal_distance:
                        if i==22:
                            ray_ini=True
                        min_ray=ray_angles[i],i
                else:
                    if i!=0 and lidar_data[i-1]>minimal_distance:
                        max_ray=ray_angles[i-1],i-1
                        if max_ray!=min_ray and min_ray[1]+3<max_ray[1]:
                            #list_test.append((min_ray,max_ray))
                            list_possible_area.append(((coords[0]+step_forward*(cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(sin(angle+(min_ray[0]+max_ray[0])/2))),False))
                if i==len(lidar_data)-23 and min_ray[1]>max_ray[1]:
                    if ray_ini:
                        boolean=True

                        for k in range(min_ray[1],len(lidar_data)+22):
                            if boolean:
                                if lidar_data[i%181]<=minimal_distance:
                                    boolean=False

                        if boolean:
                            del list_possible_area[0]
                            #list_test.append((min_ray,max_ray))
                            list_possible_area.append(((coords[0]+step_forward*(cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(sin(angle+(min_ray[0]+max_ray[0])/2))),False))
                            return list_possible_area

                    max_ray=ray_angles[i],i
                    #list_test.append((min_ray,max_ray))
                    list_possible_area.append(((coords[0]+step_forward*(cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(sin(angle+(min_ray[0]+max_ray[0])/2))),False))
        
        return list_possible_area
    


    def define_message_for_all(self) -> Tuple[Optional[int], Tuple[Any, Any]]:
        """
        Define the message, the drone will send to and receive from other surrounding drones.

        Returns:
            Tuple[Optional[int], Tuple[Any, Any]]: The message data.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        print(self.lidar_possible_paths())
        return msg_data 
    

       
    

