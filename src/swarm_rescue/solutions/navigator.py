import math
import numpy as np
from typing import List, Tuple
from swarm_rescue.simulation.utils.utils import normalize_angle

# Configuration from original file
MAX_LIDAR_RANGE = 150   # Threshold to consider as "frontier"

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
        Collect Lidar data, analyze and return a list of potential areas (Frontiers).
        Modified: Sort the list to prioritize points directly IN FRONT of the drone.
        '''
        list_possible_area = []
        min_ray = -3/4 * math.pi, 0
        max_ray = 0, 0
        ray_ini = False
        minimal_distance = 170
        step_forward = 135  
        
        coords = self.drone.estimated_pos
        angle = self.drone.estimated_angle

        if coords is None: return [] 

        # Helper function to calculate angle deviation (used for sorting)
        def sort_key_by_angle(item):
            # item structure: ((x, y), visited)
            target_pos = item[0]
            dx = target_pos[0] - coords[0]
            dy = target_pos[1] - coords[1]
            target_vector_angle = math.atan2(dy, dx)
            diff = normalize_angle(target_vector_angle - angle, False)
            return abs(diff)

        if not self.drone.lidar_is_disabled():
            lidar_data = self.drone.lidar_values()
            # [FIX CRASH] Check for None
            if lidar_data is None: 
                return []
            ray_angles = self.drone.lidar_rays_angles()
            
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
                            avg_angle = (min_ray[0] + max_ray[0]) / 2
                            tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                            ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                            list_possible_area.append(((tx, ty), False))
                            list_possible_area.sort(key=sort_key_by_angle)
                            return list_possible_area

                    max_ray = ray_angles[i], i
                    avg_angle = (min_ray[0] + max_ray[0]) / 2
                    tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                    ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                    list_possible_area.append(((tx, ty), False))

        list_possible_area.sort(key=sort_key_by_angle, reverse=True)
        return list_possible_area

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
        [UPDATE]: Tự động tăng safety_margin nếu đang cầm người để tránh va chạm "đuôi".
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