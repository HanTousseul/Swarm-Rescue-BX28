import math
import numpy as np
from typing import List, Tuple, Optional
from swarm_rescue.simulation.utils.utils import normalize_angle

# [NEW] Import the new mapping classes
try:
    from .mapping import GridMap, VictimHeatmap
except ImportError:
    from mapping import GridMap, VictimHeatmap

class Navigator:
    def __init__(self, drone):
        self.drone = drone
        
        # --- GPS/Odometer variables ---
        self.gps_last_known = None
        
        # --- [NEW] MAPPING SYSTEM ---
        # Initialize maps (Obstacles & Victims)
        # Note: We assume drone.map_size is set. Default to 800x600 if not.
        map_size = getattr(self.drone, 'map_size', (100, 100))
        self.obstacle_map = GridMap(map_size=map_size)
        self.victim_map = VictimHeatmap(map_size=map_size)
        
        # --- [NEW] PATHFINDING STATE ---
        self.current_astar_path = [] # List of waypoints from A*
        self.last_astar_target = None # To check if target changed

    def update_navigator(self):
        """
        1. Update Pose (GPS/Odom)
        2. Update Maps (Lidar/Semantic)
        """
        # --- 1. Update Position (Dead Reckoning) ---
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

       # --- [NEW] 2. Update Maps ---
        # Update Obstacle Map with Lidar
        lidar_data = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        if lidar_data is not None:
            self.obstacle_map.update_from_lidar(
                self.drone.estimated_pos, 
                self.drone.estimated_angle, # [FIX] Truyền góc Drone vào đây
                lidar_data, 
                lidar_angles
            )
            
        # Update Victim Map (Giữ nguyên)
        semantic_data = self.drone.semantic_values()
        self.victim_map.update_from_semantic(
            self.drone.estimated_pos,
            self.drone.estimated_angle,
            semantic_data
        )
    def availability_gps(self):
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()
        return gps_pos is not None or compass_angle is not None

    # --- [NEW] A* PATHFINDING INTERFACE ---
    
    def get_next_waypoint(self, final_target: np.ndarray) -> Optional[np.ndarray]:
        if final_target is None: return None
            
        # 1. Tính toán lại đường đi nếu cần
        need_replan = False
        if self.last_astar_target is None: need_replan = True
        elif np.linalg.norm(final_target - self.last_astar_target) > 30.0: need_replan = True
        elif len(self.current_astar_path) == 0: need_replan = True
        
        # [NEW] Biến lưu margin đã sử dụng
        used_safety_margin = 2 
            
        if need_replan:
            start_log = self.drone.estimated_pos.astype(int)
            end_log = final_target.astype(int)
            # print(f"[{self.drone.identifier}] 🗺️ PLANNING PATH: {start_log} -> {end_log}")

            # Chiến thuật thử lại (Retry Strategy)
            
            # Lần 1: Đường AN TOÀN (Margin 2 ~ 20px)
            used_safety_margin = 2
            self.current_astar_path = self.obstacle_map.find_path_astar(
                self.drone.estimated_pos, final_target, safety_margin=2
            )
            
            # Lần 2: Đường RỦI RO (Margin 1 ~ 10px)
            if len(self.current_astar_path) == 0:
                used_safety_margin = 1
                self.current_astar_path = self.obstacle_map.find_path_astar(
                    self.drone.estimated_pos, final_target, safety_margin=1
                )

            # Lần 3: Đường CỰC RỦI RO (Margin 0 - Sát tường)
            if len(self.current_astar_path) == 0:
                used_safety_margin = 0
                self.current_astar_path = self.obstacle_map.find_path_astar(
                    self.drone.estimated_pos, final_target, safety_margin=0
                )

            self.last_astar_target = final_target.copy()

            if len(self.current_astar_path) > 0:
                # Lưu margin vào biến instance để dùng ở đoạn dưới
                self.last_used_margin = used_safety_margin
                print(f"[{self.drone.identifier}] ✅ PATH FOUND: {len(self.current_astar_path)} nodes (Margin: {used_safety_margin})")
            else:
                print(f"[{self.drone.identifier}] ❌ PATH FAILED: Target unreachable.")

        # 2. Logic chọn điểm tiếp theo (Smoothing)
        if not self.current_astar_path:
            return final_target

        # --- Logic Smoothing cũ (Chỉ dùng khi Margin >= 2) ---
        LOOKAHEAD_DIST = 100 if self.drone.state == 'EXPLORING' else 20
        best_wp = self.current_astar_path[0]
        
        for wp in self.current_astar_path:
            dist = np.linalg.norm(self.drone.estimated_pos - wp)
            if dist < LOOKAHEAD_DIST:
                if self.obstacle_map.check_line_of_sight(self.drone.estimated_pos, wp):
                    best_wp = wp
                else:
                    break
            else:
                break
        
        while len(self.current_astar_path) > 0:
            dist_to_first = np.linalg.norm(self.drone.estimated_pos - self.current_astar_path[0])
            if dist_to_first < 30.0:
                self.current_astar_path.pop(0)
            else:
                break
                
        return best_wp