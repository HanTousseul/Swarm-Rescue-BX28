import math
import numpy as np
from typing import List, Tuple
from swarm_rescue.simulation.utils.utils import normalize_angle
from scipy.stats import circmean

# Configuration from original file
MAX_LIDAR_RANGE = 150   # Threshold to consider as "frontier"
SAFE_DISTANCE = 30      # Safe distance (pixels) to avoid collisions

class Navigator:
    def __init__(self, drone):
        self.drone = drone
        
        # --- MAPPER VARIABLES ---
        self.edge = {}
        self.visited_node = []
        self.path_history = {}
        self.waypoint_stack = [] # Stack to store return path
        
        # GPS/Odometer variables
        self.gps_last_known = None

    def update_navigator(self):
        """Update estimated position from GPS or Odometer (Dead Reckoning)."""
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()

        if gps_pos is not None and compass_angle is not None:
            self.drone.estimated_pos = gps_pos
            self.drone.estimated_angle = compass_angle
            self.gps_last_known = gps_pos
        else:
            # GPS Lost -> Use Odometer accumulation
            odom = self.drone.odometer_values() # [dist, alpha, theta]
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                move_angle = self.drone.estimated_angle + alpha
                self.drone.estimated_pos[0] += dist * math.cos(move_angle)
                self.drone.estimated_pos[1] += dist * math.sin(move_angle)
                self.drone.estimated_angle = normalize_angle(self.drone.estimated_angle + theta)
                
        if self.drone.initial_position is None: 
            self.drone.initial_position = self.drone.estimated_pos

    def availability_gps(self):
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()
        return gps_pos is not None or compass_angle is not None

    def lidar_possible_paths(self) -> List:
        '''
        Collect Lidar data, analyze and return a list of potential areas (Frontiers), sorted from the position highest
        to lowest difference between the actual angle of the drone (to get the minimum angle difference, take the last one),
        and the required angle to get to the potential area. 
        Returns None if there is no self.estimated_pos
        '''
        lidar_data = self.drone.lidar_values()
        ray_angles = self.drone.lidar_rays_angles()
        lidar_possible_angles = [] #Allows us to sort the possible paths by absolute value of angle
        minimal_distance_coefficient = 1.2 #coefficient by which we multiply the average length of lidar rays to find the minimal distance
        minimal_distance = min(np.mean(np.array(lidar_data)) * minimal_distance_coefficient, 190) #Distance above which the ray is considered not to hit an obstacle anymore. min 190 because semantic sensor rays have range 200, we choose a value slightly smaller
        step_forward = 132 #Distance that the drone will move forward from his actual position towards the possible path it chose
        angle_ignore=0 #angle centered in Pi that the drone will not consider as a possible path. Prevents the drone from counting as a possible path the path from where it came from
        edge_length=0.7 #The difference of length of two consecutive rays to consider as an opening in the wall. Given as multiple of the length of the bigger ray
        Same_possible_path = 50 #Maximum distance between two possible paths for them to be considered as the same 
        already_visited_path = 75 #Distance above which a new possible path will be considered valid if its distance with an already visited path is greather than already visited path
        rescuing_bool:bool = (self.drone.state == 'RESCUING' or self.drone.state == 'RETURNING')

        coords = self.drone.estimated_pos
        angle = self.drone.estimated_angle
    
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

            if not(rescuing_bool): 
                visited = is_visited(position)

            else: visited = True

            if visited: return # we stop if path is already visited

            #print('sending candidate to correct', position, mean_angle)
            # correction refers to setting the node closer to the drone in case it is hidden by a wall for some reason
            needs_correction = correct_position(mean_angle) 
            #print('position, needs correction',position, needs_correction)
            if needs_correction:

                position = needs_correction[0] # we correct if needed
                if not(rescuing_bool):
                    visited = is_visited(position)
                else: visited = True
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
        #print(lidar_possible_angles)
        lidar_possible_angles.reverse()
        
        if self.drone.state == 'RESCUING': 
            print(coords, lidar_possible_angles)
        return lidar_possible_angles

    def update_mapper(self):
        """Build a map of visited points (Graph Building)."""
        list_possible_area = self.lidar_possible_paths()
        # Use Int Key to avoid float precision errors
        pos_key = (int(self.drone.estimated_pos[0]), int(self.drone.estimated_pos[1]))
        
        if pos_key not in self.edge:
            self.edge[pos_key] = [] 
            
        for val in list_possible_area:
            x = val[0][0]
            y = val[0][1]
            visited = False
            for node in self.visited_node:
                if math.hypot(x - node[0], y - node[1]) < 70.0:
                    visited = True
                    break
            if not visited: 
                self.edge[pos_key].append((x,y))

    def visit(self, pos):
        if pos is not None:
            pos_key = tuple(pos) if isinstance(pos, np.ndarray) else pos
            if pos_key not in self.visited_node: 
                self.visited_node.append(pos_key)

    def is_path_blocked(self, target_pos, safety_margin=20):
        """
        Check if the straight line from current position to target_pos is blocked.
        """
        if target_pos is None: return False

        rel_pos = target_pos - self.drone.estimated_pos
        dist = np.linalg.norm(rel_pos)
        target_angle = math.atan2(rel_pos[1], rel_pos[0])
        
        angle_diff = normalize_angle(target_angle - self.drone.estimated_angle)
        
        lidar_data = self.drone.lidar_values()
        ray_angles = self.drone.lidar_rays_angles()

        if lidar_data is None or ray_angles is None:
            return False
        
        min_diff = float('inf')
        closest_ray_idx = -1
        
        for i, ray_angle in enumerate(ray_angles):
            diff = abs(normalize_angle(ray_angle - angle_diff))
            if diff < min_diff:
                min_diff = diff
                closest_ray_idx = i
                
        if closest_ray_idx != -1:
            measured_dist = lidar_data[closest_ray_idx]
            if measured_dist < (dist - safety_margin): 
                return True
                
        return False
    
    def find_best_bypass(self, original_target):
        """
        Find an intermediate point (frontier) that is closest in direction to original_target.
        """
        possible_nodes = self.lidar_possible_paths()
        if not possible_nodes:
            return None
            
        rel_pos = original_target - self.drone.estimated_pos
        target_angle = math.atan2(rel_pos[1], rel_pos[0])
        
        best_node = None
        min_angle_diff = float('inf')
        
        for node_info in possible_nodes:
            node_pos = np.array(node_info[0])
            node_rel = node_pos - self.drone.estimated_pos
            node_angle = math.atan2(node_rel[1], node_rel[0])
            
            diff = abs(normalize_angle(node_angle - target_angle))
            
            if diff < min_angle_diff:
                min_angle_diff = diff
                best_node = node_pos
                
        return best_node

    def find_shortcut_target(self):
        """
        Find the furthest ancestor that the drone can fly straight to (without wall blocking).
        Helps the drone return home faster instead of step-by-step.
        """
        if self.drone.current_target is None: return None
        
        # 1. Retrieve Ancestors Chain
        # Look back max 8 steps to save computation
        ancestors = []
        curr_key = (int(self.drone.current_target[0]), int(self.drone.current_target[1]))
        
        temp_key = curr_key
        for _ in range(8):
            if temp_key in self.path_history:
                parent_pos = self.path_history[temp_key]
                ancestors.append(parent_pos)
                temp_key = (int(parent_pos[0]), int(parent_pos[1]))
            else:
                break
        
        if not ancestors: return None

        # 2. Greedy Check (Furthest to Nearest)
        for target_pos in reversed(ancestors):
            # Distance check: If too far (> 300px), Lidar can't verify wall
            dist = np.linalg.norm(target_pos - self.drone.estimated_pos)
            if dist > 300.0: continue 

            # Check wall block
            # Note: Need larger safety_margin (30px) for shortcuts
            if not self.is_path_blocked(target_pos, safety_margin=30):
                return target_pos
                
        return None