import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any


# Import necessary modules from framework
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


# --- CONFIGURATION ---
SAFE_DISTANCE = 40      # Safe distance (pixels) to avoid collisions
KP_ROTATION = 2.0       # P coefficient for rotation
KP_FORWARD = 0.5        # P coefficient for forward movement
MAX_LIDAR_RANGE = 150   # Threshold to consider as "frontier"
REACH_THRESHOLD = 25.0  # Distance to consider as reached destination


class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # --- 1. NAVIGATOR VARIABLES (Quoc Viet & Anhad) ---
        # Estimated position and angle (More reliable than raw GPS)
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None # To check when GPS is lost/recovered


        # --- 2. MAPPER VARIABLES (Marc) ---
        self.edge = {}
        self.visited_node = []
        
        # --- 3. COMMANDER VARIABLES (Van Khue) ---
        self.state = "EXPLORING" # EXPLORING, RESCUING, RETURNING, DROPPING
        self.path_history = {}
        self.current_target = None # Current target point (np.array)
        self.rescue_center_pos = None # Rescue center position (save when found)

        self.position_before_rescue = None
        self.initial_position = None
        self.cnt_timestep = 0
        
        self.temporary_store = [] # Temporarily stores visited nodes in no-com zone
        self.comm_active = True # Whether communication is activated
        self.consecutive_silence_steps = 0 # Number of consecutive timesteps the drone haven't received any messages

    def reset(self):
        # --- 1. NAVIGATOR VARIABLES (Quoc Viet & Anhad) ---
        # Estimated position and angle (More reliable than raw GPS)
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None # To check when GPS is lost/recovered


        # --- 2. MAPPER VARIABLES (Marc) ---
        self.edge = {}
        self.visited_node = []
        
        # --- 3. COMMANDER VARIABLES (Van Khue) ---
        self.state = "EXPLORING" # EXPLORING, RESCUING, RETURNING, DROPPING
        self.path_history = {}
        self.current_target = None # Current target point (np.array)
        self.rescue_center_pos = None # Rescue center position (save when found)

        self.position_before_rescue = None
        
        self.temporary_store = [] # Temporarily stores visited nodes in no-com zone
        self.comm_active = True # Whether communication is activated
        self.consecutive_silence_steps = 0 # Number of consecutive timesteps the drone haven't received any messages


    def update_navigator(self):
        """Update estimated position based on GPS (if available) or Odometer."""
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            # GOOD GPS: Trust GPS
            self.estimated_pos = gps_pos
            self.estimated_angle = compass_angle
            self.gps_last_known = gps_pos
        
        else:
            # GPS LOST: Use Odometer for accumulation (Dead Reckoning)
            odom = self.odometer_values() # [dist, alpha, theta]
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                
                # Quoc Viet's logic:
                # Alpha is the movement direction relative to OLD orientation
                move_angle = self.estimated_angle + alpha
                
                self.estimated_pos[0] += dist * math.cos(move_angle)
                self.estimated_pos[1] += dist * math.sin(move_angle)
                
                # Update new angle
                self.estimated_angle = normalize_angle(self.estimated_angle + theta)
        if self.initial_position is None: self.initial_position = self.estimated_pos


    def lidar_possible_paths(self) -> List:
        '''
        Collect Lidar data, analyze and return a list of potential areas (Frontiers).
        Modified: Sort the list to prioritize points directly IN FRONT of the drone.
        Returns an empty list if GPS is not working and there is no self.estimated_pos
        '''
        list_possible_area = []
        min_ray = -3/4 * math.pi, 0
        max_ray = 0, 0
        ray_ini = False
        minimal_distance = 250
        step_forward = 132
        
        # Note: Should use estimated_pos instead of gps_values to avoid errors when GPS is lost
        coords = self.estimated_pos
        angle = self.estimated_angle

        if coords is None: return [] # Avoid crash if GPS is lost and estimated_pos is not set


        # Helper function to calculate angle deviation (used for sorting)
        def sort_key_by_angle(item):
            # item structure: ((x, y), visited)
            target_pos = item[0]
            dx = target_pos[0] - coords[0]
            dy = target_pos[1] - coords[1]
            
            # Angle of the vector from drone to target point
            target_vector_angle = math.atan2(dy, dx)
            # Angle deviation from drone's heading (normalized to -pi to pi)
            diff = normalize_angle(target_vector_angle - angle, False)
            
            # Return absolute value (closer to 0 is better)
            return abs(diff)

        if not self.lidar_is_disabled():
            lidar_data = self.lidar_values()
            ray_angles = self.lidar_rays_angles()
            
            for i in range(22, len(lidar_data) - 22):
                if lidar_data[i] > minimal_distance:
                    if lidar_data[i - 1] <= minimal_distance:
                        if i == 22:
                            ray_ini = True
                        min_ray = ray_angles[i], i
                else:
                    if i != 0 and lidar_data[i - 1] > minimal_distance:
                        max_ray = ray_angles[i - 1], i - 1
                        if max_ray != min_ray and min_ray[1] + 3 < max_ray[1]:
                            # Calculate coordinates
                            avg_angle = (min_ray[0] + max_ray[0]) / 2
                            tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                            ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                            list_possible_area.append(((tx, ty), False))
                
                # Handle edge case (circular wrap-around)
                if i == len(lidar_data) - 23 and min_ray[1] > max_ray[1]:
                    if ray_ini:
                        boolean = True
                        for k in range(min_ray[1], len(lidar_data) + 22):
                            if boolean:
                                if lidar_data[i % 181] <= minimal_distance:
                                    boolean = False

                        if boolean:
                            del list_possible_area[0]
                            
                            # Calculate last point
                            avg_angle = (min_ray[0] + max_ray[0]) / 2
                            tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                            ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                            list_possible_area.append(((tx, ty), False))
                            
                            # --- SORT BEFORE RETURNING ---
                            list_possible_area.sort(key=sort_key_by_angle)
                            return list_possible_area

                    max_ray = ray_angles[i], i
                    # Calculate last point (no loop)
                    avg_angle = (min_ray[0] + max_ray[0]) / 2
                    tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                    ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                    list_possible_area.append(((tx, ty), False))

        # --- SORT BEFORE RETURNING (Normal case) ---
        list_possible_area.sort(key=sort_key_by_angle, reverse=True)
        
        return list_possible_area

    def update_mapper(self):
        """Scan Lidar to find new frontier points."""
        list_possible_area = self.lidar_possible_paths()
        pos_key = tuple(self.estimated_pos)
        if pos_key not in self.edge:
            self.edge[pos_key] = [] 
        for val in list_possible_area:
            x = val[0][0]
            y = val[0][1]
            visited = False
            for node in self.visited_node:
                delta_x = x - node[0]
                delta_y = y - node[1]
                dist_to_target = math.hypot(delta_x, delta_y)
                if dist_to_target < 30: visited = True
            if not visited: 
                self.edge[pos_key].append((x,y))
                # print(f'Add new target {x}, {y}')



    def move_to_target(self) -> CommandsDict:
        """
        Control the drone to move PRECISELY to the target.
        Strategy: Go slow, rotate accurately, decelerate early.
        """
        # # print(f'Going to {self.current_target}') # Debug if needed
        
        if self.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        delta_x = self.current_target[0] - self.estimated_pos[0]
        delta_y = self.current_target[1] - self.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # 1. Rotate towards the target
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.estimated_angle)
        
        # Increase rotation force to correct heading faster
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # 2. Move forward (NEW LOGIC: SLOW & STEADY)
        
        # Speed configuration
        MAX_SPEED = 0.6
        BRAKING_DIST = 150.0
        STOP_DIST = 15.0 # Increase stopping distance slightly for safety

        if dist_to_target > BRAKING_DIST:
            forward_cmd = MAX_SPEED
        elif dist_to_target > STOP_DIST:
            # Linear deceleration
            # Removed max(0.1, ...) line to allow it to reduce close to 0
            forward_cmd = (dist_to_target / BRAKING_DIST) * MAX_SPEED
            forward_cmd = max(0.1, forward_cmd)
        else:
            # Very close (< 15px): Cut throttle completely
            forward_cmd = 0.05

        # 3. Rotation Discipline (Strict Rotation)
        # Only allow movement if heading is accurate (deviation < 0.2 rad ~ 11 degrees)
        # Old code was 0.5 (30 degrees) -> Too loose
        if abs(angle_error) > 0.2:
            forward_cmd = 0.0 # Stop to finish rotating

        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # 4. Collision Avoidance (Lidar Safety) - Kept as is
        # if self.lidar_using_state:
        # lidar_vals = self.lidar_values()
        # if lidar_vals is not None:
        #     if lidar_vals[90] < SAFE_DISTANCE:
        #         forward_cmd = 0.0 
        #         rotation_cmd = 1.0 

        # --- SPECIAL LOGIC FOR RETURNING (Carrying person) ---
        if self.state == "RETURNING":
            forward_cmd = 0.7
        # else:
        #     # print(f'Spec of moving, forward: {forward_cmd}, rotation: {rotation_cmd}')

        grasper_val = 1 if (self.state == "RESCUING" or self.state == "RETURNING") else 0


        # if not self.lidar_using_state:
        #     return {
        #         "forward": forward_cmd*1.5, 
        #         "lateral": 0.0, 
        #         "rotation": rotation_cmd, 
        #         "grasper": grasper_val
        #     }
        # else:
        # print(f'Spec of moving, forward: {forward_cmd}, rotation: {rotation_cmd}, dist: {dist_to_target}')
        return {
            "forward": forward_cmd, 
            "lateral": 0.0, 
            "rotation": rotation_cmd, 
            "grasper": grasper_val
        }
    
    def visit(self, pos):
        if pos is not None:
            pos_key = tuple(pos) if isinstance(pos, np.ndarray) else pos
            if pos_key not in self.visited_node: 
                # print(f'Add {pos_key} to visited nodes')
                self.visited_node.append(pos_key)


    def control(self) -> CommandsDict:
        self.cnt_timestep += 1
        check_center = False
        
        # 1. Update Navigator (Always run first)
        self.update_navigator()
        
        # 2. Process Communications (Update knowledge from other drones)
        self.comm_visited()
        self.update_comm_status()
        
        # 3. Process Semantic Sensor (Find person / Station)
        semantic_data = self.semantic_values()
        found_person_pos = None
        found_rescue_pos = None
        
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    # Calculate person position based on relative angle/distance
                    angle_global = self.estimated_angle + data.angle
                    px = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    py = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    found_person_pos = np.array([px, py])
                
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    # Save station position for later use
                    check_center = True
                    if self.rescue_center_pos is None:
                        angle_global = self.estimated_angle + data.angle
                        rx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                        ry = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                        self.rescue_center_pos = np.array([rx, ry])
                    found_rescue_pos = True


        # 4. STATE MACHINE
        
        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            if self.cnt_timestep == 2400: self.state = "RETURNING"
            # If person found -> Switch to rescue
            if found_person_pos is not None:
                self.state = "RESCUING"
                # print(f'Going to rescue from {self.current_target} to {found_person_pos}')
                self.position_before_rescue = self.current_target
                self.current_target = found_person_pos
                # child_key = tuple(found_person_pos)
                # self.path_history[child_key] = self.current_target
                # self.current_target = found_person_pos
                # cur_key = tuple(self.current_target)
                # # print(f'Check key valid: {cur_key in self.path_history}')
            
            # If no target or reached old target
            elif self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                self.visit(self.current_target)
                # print(f'Arrived {self.current_target}')
                
                # 2. Update Mapper (To find new paths)
                if self.current_target is None: self.visit(self.estimated_pos)
                self.update_mapper()


                pos_key = tuple(self.estimated_pos)
                if pos_key in self.edge and len(self.edge[pos_key]):
                    next_target = self.edge[pos_key].pop()
                    
                    if self.current_target is None: self.path_history[next_target] = self.estimated_pos
                    else: self.path_history[next_target] = self.current_target
                    
                    self.current_target = np.array(next_target)
                else:
                    # No more exploration paths -> Return to previous node
                    current_key = tuple(self.current_target) if self.current_target is not None else None
                    if current_key and current_key in self.path_history:
                         self.current_target = self.path_history[current_key]
                         # print('Goint to parent node')
                    else:
                        # Handle when there's no return path
                        # print("No parent node found, staying at current position")
                        self.current_target = self.estimated_pos.copy()
                # print(f'Choose next target: {self.current_target}')


        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":

            if found_person_pos is not None and not self.grasped_wounded_persons():
                # print(f'Going to rescue from {self.current_target} to {found_person_pos}')
                self.current_target = found_person_pos
                # child_key = tuple(found_person_pos)
                # parent_node = self.current_target if self.current_target is not None else self.estimated_pos
                # self.path_history[child_key] = parent_node.copy()
                # # print(f'Save parent: parent: {parent_node}, child: {child_key}')

            # Check if already grasped
            if self.grasped_wounded_persons():
                self.state = "RETURNING"
                self.lidar_using_state = False
                self.current_target = self.position_before_rescue
                # print(f'Graped person at target {self.current_target} and go back to {self.current_target}')


        # --- STATE: RETURNING ---
        elif self.state == "RETURNING":
            if check_center:
                # print(f'See rescue center at {self.rescue_center_pos} with dist {np.linalg.norm(self.estimated_pos - self.rescue_center_pos)}')
                self.current_target = self.rescue_center_pos
                if found_rescue_pos and np.linalg.norm(self.estimated_pos - self.current_target) < 15:
                    # print(f'Start dropping person at {self.estimated_pos}')
                    self.state = "DROPPING"
            else:

                # print(f'Return, dist: {np.linalg.norm(self.estimated_pos - self.current_target)}')
                if np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:

                    current_key = tuple(self.current_target) if self.current_target is not None else None
                    # print(f'Check key {current_key} and {current_key in self.path_history}')
                    if current_key and current_key in self.path_history: 
                        # print(f'Check parent: {self.path_history[current_key]}')
                        self.current_target = self.path_history[current_key]
                        # print(f'Going back to parent')
                    else:
                        # If rescue center found -> go straight to it
                        if self.rescue_center_pos is not None:
                            self.current_target = self.rescue_center_pos
                        
                        # If reached destination (station)
                        if found_rescue_pos and np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                            self.state = "DROPPING"
                # print(f'Going back to {self.current_target}, at {self.estimated_pos}')


        # --- STATE: DROPPING ---
        elif self.state == "DROPPING":
            # Stop and release
            # print(f'Finish dropped')
            # self.reset()
            # self.state = "EXPLORING"
            self.current_target = self.initial_position

        # 5. Execute movement
        return self.move_to_target()


    def define_message_for_all(self):
        """
        Send message to other drones
        """
        message = {'id': self.identifier, 'position': tuple(self.estimated_pos)}
        message['temporary'] = self.temporary_store if self.temporary_store else []
        return message
    
    def comm_visited(self):
        """Add the visited nodes from other drones to self.visited_node, 
        using self.visit(pos). Additionally, update visited nodes based on the 
        temporary list received from the drone that was in no-com zone, and clear
        out temporary_store of that drone
        """
        if not self.communicator:
            return
        
        if self.communicator.received_messages:
            self.temporary_store = []
        
        for msg in self.communicator.received_messages:
            if not msg:
                continue
            
            if 'position' in msg:
                self.visit(msg['position'])
            
            if 'temporary' in msg and msg['temporary']:
                for p in msg['temporary']:
                    self.visit(p)
        
    def update_comm_status(self):
        """Update the communication status of the drone, 
        to determine whether the drone is in No-Com Zone.
        """
        if self.communicator and len(self.communicator.received_messages) > 0:
            self.consecutive_silence_steps = 0
            self.comm_active = True
        else:
            self.consecutive_silence_steps += 1
        
        # Threshold: If silence persists for 10 steps, assume No-Com
        if self.consecutive_silence_steps > 10:
            self.comm_active = False
            
    def visit_no_comm(self): 
        """Add the nodes to a temporary list and visited_node if the drone is in no-com zone
        """
        if self.comm_active == False:
            pos = tuple(self.estimated_pos)
            self.visit(pos)
            self.temporary_store.append(pos) 
        
     
