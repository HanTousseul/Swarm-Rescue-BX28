import math
from swarm_rescue.simulation.drone.controller import CommandsDict

class MyDroneDriver:
    SAFE_DISTANCE = 20    
    Kp_ROTATION = 1.0     
    Kp_FORWARD = 0.5       

    def __init__(self):
        self.estimated_pos = [0.0, 0.0]
        self.estimated_angle = 0.0
        self.current_target = [0.0, 0.0]
        self.lidar_values_list = []       
        self.current_grasper_state = 0    

    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    def move_to_target(self) -> CommandsDict:
        """
        Generates movement commands to reach self.current_target from self.estimated_pos.
        Uses P-controller for rotation and optional forward scaling.
        Includes basic LiDAR collision avoidance.
        """
        delta_x = self.current_target[0] - self.estimated_pos[0]
        delta_y = self.current_target[1] - self.estimated_pos[1]

        target_angle = math.atan2(delta_y, delta_x)
        angle_error = self.normalize_angle(target_angle - self.estimated_angle)

        rotation_command = max(-1.0, min(1.0, self.Kp_ROTATION * angle_error))

        distance_to_target = math.hypot(delta_x, delta_y)
        forward_command = max(-1.0, min(1.0, self.Kp_FORWARD * distance_to_target))

        if self.lidar_values_list:
            lidar_front_index = len(self.lidar_values_list) // 2
            if self.lidar_values_list[lidar_front_index] < self.SAFE_DISTANCE:
                forward_command = 0.0 
        commands: CommandsDict = {
            "forward": forward_command,
            "lateral": 0.0,  
            "rotation": rotation_command,
            "grasper": self.current_grasper_state
        }

        return commands
