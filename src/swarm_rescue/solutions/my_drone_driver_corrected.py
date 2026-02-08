import math
import numpy as np
from typing import *
from scipy.stats import circmean


# Import necessary modules from framework
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.mapping.example_mapping import OccupancyGrid



# --- CONFIGURATION ---
SAFE_DISTANCE = 30      # Safe distance (pixels) to avoid collisions
KP_ROTATION = 2.0       # P coefficient for rotation
KP_FORWARD = 0.5        # P coefficient for forward movement
MAX_LIDAR_RANGE = 150   # Threshold to consider as "frontier"
REACH_THRESHOLD = 25.0  # Distance to consider as reached destination
MAP_RESOLUTION = 8      # Resolution of the map generated (might be helpful for pathfinding)


class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # --- 1. NAVIGATOR VARIABLES (Quoc Viet & Anhad) ---
        # Estimated position and angle (More reliable than raw GPS)
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None # To check when GPS is lost/recovered
        self.iteration = 0

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

        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=MAP_RESOLUTION,
                                  lidar=self.lidar())   


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
        Collect Lidar data, analyze and return a list of potential areas (Frontiers), sorted from the position highest
        to lowest difference between the actual angle of the drone (to get the minimum angle difference, take the last one),
        and the required angle to get to the potential area. 
        Returns None if there is no self.estimated_pos
        '''
        lidar_data = self.lidar_values()
        ray_angles = self.lidar_rays_angles()
        lidar_possible_angles = [] #Allows us to sort the possible paths by absolute value of angle
        minimal_distance_coefficient = 1.2 #coefficient by which we multiply the average length of lidar rays to find the minimal distance
        minimal_distance = min(np.mean(np.array(lidar_data)) * minimal_distance_coefficient, 190) #Distance above which the ray is considered not to hit an obstacle anymore. min 190 because semantic sensor rays have range 200, we choose a value slightly smaller
        step_forward = 132 #Distance that the drone will move forward from his actual position towards the possible path it chose
        angle_ignore=0 #angle centered in Pi that the drone will not consider as a possible path. Prevents the drone from counting as a possible path the path from where it came from
        edge_length=0.7 #The difference of length of two consecutive rays to consider as an opening in the wall. Given as multiple of the length of the bigger ray
        Same_possible_path = 50 #Maximum distance between two possible paths for them to be considered as the same 
        already_visited_path = 50 #Distance above which a new possible path will be considered valid if its distance with an already visited path is greather than already visited path

        coords = self.estimated_pos
        angle = self.estimated_angle
    
        if coords is None: return [] # Avoid crash if GPS is lost and estimated_pos is not set (which should not happen)

        begin_loop = False # if the last consecutive rays hit a wall
        end_of_loop = False # if the first consecutive rays hit a wall
        min_ray = 0,False
        max_ray = 180, False
        edge_begin = None
        edge_end = None
        extra_rays = 20 # we take the new possible path of an edge as the middle of extra rays after the edge
        correct_position_nb_rays:int = 5 #(used in correct position helper function) number of rays sweeped centered around the possible path that are checked for minimum length 

        #print('position', coords, angle)
        def is_visited(position: Tuple) -> bool:
            '''
            function that returns True if the position is worth adding to list_possible_paths, False otherwise
            checks for proximity with visited nodes, current position and all possible paths
                        
            :param position: (x,y) coordinates of the new possible path considered
            :type position: Tuple
            :return: whether or not the position is worth adding to the list
            :rtype: bool
            '''
            visited = False
            for elt in self.visited_node:

                node = np.array(elt)

                if not visited and np.linalg.norm(position-node)<already_visited_path:
                    visited = True

            if not(visited):

                for elt in lidar_possible_angles:
                    node = np.array(elt[0])
                    
                    if not(visited) and np.linalg.norm(position-node)<Same_possible_path:
                        visited = True

            if not(visited):

                if np.linalg.norm(position-coords)<Same_possible_path:
                    visited = True
            return visited

        def correct_position(mean_angle:float) -> Tuple:
            '''
            Takes in a new possible path candidate, makes sure that there isn't a wall between it and the drone, and corrects it if so
            
            :param mean_angle: float of the angle of this new position
            :type mean_angle: float
            :return: ((x,y),mean_angle) corrected new possible path
            :rtype: Tuple
            '''
            min_dist: float = step_forward
            index: int = (round(np.rad2deg(mean_angle)) // 2 + 90) % 180
            is_first_index: bool = False
            #print('correct_position_call', mean_angle)
            for ray in range(index - correct_position_nb_rays, index + correct_position_nb_rays + 1):

                if lidar_data[ray % 180] < step_forward + SAFE_DISTANCE:

                    #print('first_needs_correction',ray, lidar_data[ray % 180])
                    min_dist = lidar_data[ray % 180] - SAFE_DISTANCE
                    if not(is_first_index):

                        is_first_index = True
                        first_index = ray
                        last_index = ray

                    else:
                        last_index = ray


            #print('needs correction?', is_first_index, min_dist)
            if not(is_first_index): return None

            if first_index == index - correct_position_nb_rays and last_index == index + correct_position_nb_rays + 1: 
                #print('needs correction? No')
                return coords
            
            #print('it does need correction', index, first_index, last_index)
            continuity: bool = True

            for ray in range(first_index, last_index):

                if lidar_data[ray % 180] > step_forward + SAFE_DISTANCE:
                    continuity = False

            if continuity:

                if first_index > index or (last_index > index and abs(first_index - index) <= abs(index - last_index)):

                    interval_correction = first_index - 10, first_index

                else:

                    interval_correction = last_index + 1, last_index + 11

                interval_correction_continuity: bool = True
            
                for ray in range(interval_correction[0], interval_correction[1]):

                    if lidar_data[ray % 180] < step_forward + SAFE_DISTANCE:
                        #print(ray, lidar_data[ray % 180])
                        interval_correction_continuity = False

            if not(continuity) or not(interval_correction_continuity): 

                new_pos = np.array((float(coords[0] + min_dist*np.cos(mean_angle)), 
                                    float(coords[1] + min_dist*np.sin(mean_angle))))
                #print('new_if', continuity, first_index, last_index, new_pos, mean_angle)
                return (new_pos, mean_angle)

            else:

                #print('aaaaaaaaaaaaaaaaaaaaaaaaaa',
                #    (interval_correction[0], ray_angles[interval_correction[0]]),
                #    (interval_correction[1], ray_angles[interval_correction[1]]),
                #    step_forward)
                computed_position = compute_position(
                    (interval_correction[0], ray_angles[interval_correction[0] % 180]),
                    (interval_correction[1], ray_angles[interval_correction[1] % 180]),
                    step_forward = step_forward
                    )
                
                new_pos = computed_position[0], computed_position[1]
                new_mean_angle = computed_position[2]
                
                #print('new_else', continuity, interval_correction, interval_correction_continuity, first_index, last_index, new_pos, new_mean_angle)
                return(new_pos, new_mean_angle)

        def compute_position(Ray1:Tuple, Ray2:Tuple, step_forward: float) -> Tuple:
            '''
            Takes in two rays and outputs the position of the node to be added as well as the mean angle of the two rays.
            
            :param Ray1: The first ray of the position
            :type Ray1: Tuple (index, ray_angle[index])
            :param Ray2: The last ray of the position
            :type Ray2: Tuple (index, ray_angle[index])
            :param step_forward: distance between the drone and the new possible path
            :type step_forward: float
            '''
            mean_angle = normalize_angle(circmean((Ray1[1], Ray2[1])))
            #print('mean_angle', mean_angle, Ray1, Ray2)
            Trueangle = mean_angle + angle


            return (float((coords[0] + step_forward * np.cos(Trueangle))), 
                    float(coords[1] + step_forward * np.sin(Trueangle)), 
                    mean_angle
                    )    

        def add_to_lidar_possible_angles(position_mean_angle: Tuple) -> None:
            '''
            Takes as argument compute position for 2 rays, and inserts the corresponding position in lidar_possible_angle if the node is not yet visited, while sorting the list in decreasing difference between angle of the drone and angle of the position
            
            :param position_mean_angle: Tuple of the form (coords[0], coords[1], mean_angle)
            :type position_mean_angle: Tuple
            '''
            visited:bool = False #if the node has been visited or not
            position:tuple = (position_mean_angle[0], position_mean_angle[1])
            mean_angle:float = position_mean_angle[2]
            visited = is_visited(position)

            if visited: return # we stop if path is already visited

            #print('sending candidate to correct', position, mean_angle)
            # correction refers to setting the node closer to the drone in case it is hidden by a wall for some reason
            needs_correction = correct_position(mean_angle) 
            #print('position, needs correction',position, needs_correction)
            if needs_correction:

                position = needs_correction[0] # we correct if needed
                visited = is_visited(position)
                #print('Needs corrcetion, is visited?', visited)
                if visited: return # if the corrected path is not worth adding

            # if the path is new (theoretically)
            inserted = False
            # Sort the angles in decreasing absolute value of angle order
            if len(lidar_possible_angles)>0:
                rank = 0
                while rank < len(lidar_possible_angles) and abs(lidar_possible_angles[rank][1]) < abs(mean_angle):
                    rank+=1

                if rank != len(lidar_possible_angles):

                    inserted = True
                    lidar_possible_angles.insert(rank, (position,mean_angle))

            if not inserted :

                lidar_possible_angles.append((position, mean_angle))

            #print('computed', (position,mean_angle))
            return

        #In case there is nothing directly in front of the drone
        boolean = True
        for index in range(85,96):
            if lidar_data[index] < minimal_distance: 
                boolean = False
        if boolean:
            Ray1 = 85, ray_angles[85]
            Ray2 = 95, ray_angles[95]
            computed = compute_position(Ray1, Ray2, step_forward)
            add_to_lidar_possible_angles(computed)

        for index in range(round(angle_ignore/2), 181 - round(angle_ignore/2) -1):

            if lidar_data[index] < minimal_distance and lidar_data[index+1] > minimal_distance:
                
                min_ray = index+1, ray_angles[index+1]
                #print('min_ray', min_ray)
            elif lidar_data[index] > minimal_distance and lidar_data[index+1] < minimal_distance:

                max_ray = index, ray_angles[index]
                #print('max_ray', max_ray)
                if min_ray == (0,False):

                    end_of_loop = max_ray # if the "hole" that the drone is seeing with its first lidar ray starts in the zone he's not looking in
                elif min_ray[0]+4<max_ray[0]: # only add possible path if a couple of consecutive rays 'see' it
                    #print('call1')
                    #print('min_ray,max_ray', min_ray,max_ray)
                    computed = compute_position(min_ray,max_ray,step_forward)
                    add_to_lidar_possible_angles(computed)
            
            if lidar_data[index+1]*edge_length > lidar_data[index]: #imagine a little room with a door flush in a wall. Might not be deep but we can use the fact that there will be two consecutive rays with a big gap
                edge_begin = index+1, ray_angles[index+1]
                #print('edge_begin', edge_begin)

            elif lidar_data[index+1] < lidar_data[index] * edge_length:
                 
                edge_end = index, ray_angles[index]
                #print('edge_end',edge_end)
                if edge_begin != None: 

                    #print('call2, edge_begin and edge_end', edge_begin, edge_end)
                    computed = compute_position(edge_begin,edge_end,step_forward)
                    #print('computed',computed)
                    add_to_lidar_possible_angles(computed)

                    edge_begin = None
                    edge_end = None

            if edge_begin != None:

                if index > edge_begin[0]+22: # if 45 degrees have passed and we still haven't found an edge_end

                    edge_end = edge_begin[0] + extra_rays, ray_angles[edge_begin[0]+extra_rays]
                    #print('call2, edge_begin')
                    computed = compute_position(edge_begin,edge_end,step_forward)
                    #print('computed',computed)
                    add_to_lidar_possible_angles(computed)

                    edge_begin = None
                    edge_end = None

            elif edge_end != None: # we found an edge_end but no edge_begin, we set edge_begin to be the 10th ray before edge_end.
                edge_begin = (edge_end[0]-extra_rays) % 181, ray_angles[(edge_end[0]-extra_rays) % 181]
            
                #print('call2, edge_end')
                computed = compute_position(edge_begin,edge_end,step_forward)
                add_to_lidar_possible_angles(computed)

                edge_begin = None
                edge_end = None                

        if min_ray!= (0,False) and max_ray[0]<min_ray[0]:

            begin_loop=min_ray

        # we now take care of the begin and end loops
        if begin_loop != False and end_of_loop != False:
            
            end_of_loop_bool = False
            index = begin_loop[0]



            while index - 180 < end_of_loop[0] and not(end_of_loop_bool) and(lidar_data[index % 180] > minimal_distance):

                index +=1

                if index % 180 == end_of_loop[0]:

                    end_of_loop_bool = True

            #print('call3')
            #print('end_of_loop', end_of_loop)
            #print('call3 param', begin_loop, (index, ray_angles[index%180]))
            computed=compute_position(begin_loop, (index, ray_angles[index % 180]), step_forward)
            add_to_lidar_possible_angles(computed)

            if not(end_of_loop_bool):

                index = end_of_loop[0]

                while index + 180 > begin_loop[0] and lidar_data[index % 180] > minimal_distance:

                    index += 1

                #print('call4')
                #print('call4 params',(index % 180, ray_angles[index % 180]), end_of_loop )
                computed=compute_position((index % 180, ray_angles[index % 180]), end_of_loop, step_forward)
                add_to_lidar_possible_angles(computed)

        elif begin_loop != False and end_of_loop == False:

            index = begin_loop[0]

            while index < 180 + round(angle_ignore/2) and lidar_data[index % 180] > minimal_distance:

                index +=1

            #print('call5')
            #print(begin_loop, (index, ray_angles[index % 180]))
            computed=compute_position(begin_loop, (index, ray_angles[index % 180]), step_forward)
            add_to_lidar_possible_angles(computed)

        elif begin_loop == False and end_of_loop != False:

            index = end_of_loop[0]

            while index > - round(angle_ignore/2) and lidar_data[index % 180] > minimal_distance:

                index -=1

            index+=1
            #print('call6')
            computed=compute_position((index % 180, ray_angles[index % 180]), end_of_loop, step_forward)
            add_to_lidar_possible_angles(computed)
        
        lidar_possible_paths = [tuple((a[0],a[1])) for a in lidar_possible_angles ]
        #print('list',lidar_possible_angles)
        print(lidar_possible_angles)
        lidar_possible_angles.reverse()
        return lidar_possible_angles

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
        #print(f'Going to {self.current_target}') # Debug if needed
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
        #     # #print(f'Spec of moving, forward: {forward_cmd}, rotation: {rotation_cmd}')

        grasper_val = 1 if (self.state == "RESCUING" or self.state == "RETURNING") else 0


        # if not self.lidar_using_state:
        #     return {
        #         "forward": forward_cmd*1.5, 
        #         "lateral": 0.0, 
        #         "rotation": rotation_cmd, 
        #         "grasper": grasper_val
        #     }
        # else:
        # #print(f'Spec of moving, forward: {forward_cmd}, rotation: {rotation_cmd}, dist: {dist_to_target}')
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
        # Mapping stuff

        #self.estimated_pose = Pose(np.asarray(self.estimated_pos),
        #                           self.estimated_angle)
        #self.grid.update_grid(pose=self.estimated_pose)

        #self.iteration +=1
        #if self.iteration % 5 == 0:
        #    self.grid.display(self.grid.grid,
        #                      self.estimated_pose,
        #                      title="occupancy grid")
        #    self.grid.display(self.grid.zoomed_grid,
        #                      self.estimated_pose,
        #                      title="zoomed occupancy grid")
        #lidar_paths = self.lidar_possible_paths()
        #elt = ((210,40),0)
        #print('aaaaaaaaaaaaaaaaaaaaa',elt)
        #self.grid.add_points(round(elt[0][0]/ MAP_RESOLUTION), round(elt[0][1] / MAP_RESOLUTION), 1000)

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
                if self.position_before_rescue is None: self.position_before_rescue = self.estimated_pos
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
                if self.current_target is None: self.current_target = self.position_before_rescue
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
        return



    def define_message_for_all(self):
        pass