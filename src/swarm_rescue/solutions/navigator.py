import math
import numpy as np
from typing import List, Tuple, Optional
from swarm_rescue.simulation.utils.utils import normalize_angle

try:
    from .mapping import GridMap, VictimHeatmap
except ImportError:
    from mapping import GridMap, VictimHeatmap

class Navigator:
    def __init__(self, drone):
        self.drone = drone
        self.gps_last_known = None
        
        map_size = getattr(self.drone, 'map_size', (100, 100))
        self.obstacle_map = GridMap(map_size=map_size)
        self.victim_map = VictimHeatmap(map_size=map_size)
        
        self.current_astar_path = []
        self.last_astar_target = None 
        self.replan_timer = 0
        self.failure_cooldown = 0 
        self.last_used_margin = 2 

        self.cached_nearby_drones = []
        self.dijkstra_update_timer = 0
        self.dijkstra_target_cached = None

    def update_navigator(self, nearby_drones: List[np.ndarray] = []):
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            self.drone.estimated_pos = gps_pos
            self.drone.estimated_angle = compass_angle
            self.gps_last_known = gps_pos
        else:
            odom = self.drone.odometer_values()
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                move_angle = self.drone.estimated_angle + alpha
                self.drone.estimated_pos[0] += dist * math.cos(move_angle)
                self.drone.estimated_pos[1] += dist * math.sin(move_angle)
                self.drone.estimated_angle = normalize_angle(self.drone.estimated_angle + theta)
                
        if self.drone.initial_position is None: 
            self.drone.initial_position = self.drone.estimated_pos

        self.cached_nearby_drones = nearby_drones
        semantic_data = self.drone.semantic_values()
        nearby_victims = []
        if semantic_data:
            from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    angle_global = self.drone.estimated_angle + data.angle
                    vx = self.drone.estimated_pos[0] + data.distance * math.cos(angle_global)
                    vy = self.drone.estimated_pos[1] + data.distance * math.sin(angle_global)
                    nearby_victims.append(np.array([vx, vy]))

        lidar_data = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        if lidar_data is not None:
            self.obstacle_map.update_from_lidar(self.drone.estimated_pos, self.drone.estimated_angle, lidar_data, lidar_angles, nearby_drones_pos=nearby_drones, nearby_victims_pos=nearby_victims)
        self.victim_map.update_from_semantic(self.drone.estimated_pos, self.drone.estimated_angle, semantic_data)

    def find_nearest_walkable(self, target_pos: np.ndarray, search_radius_grid: int = 10) -> Optional[np.ndarray]:
        gx_t, gy_t = self.obstacle_map.world_to_grid(target_pos[0], target_pos[1])
        queue = [(gx_t, gy_t)]
        visited = set([(gx_t, gy_t)])
        SAFE_COST = 400.0 # [TUNE] Chấp nhận điểm hơi nguy hiểm để thoát kẹt
        
        while queue:
            cx, cy = queue.pop(0)
            if 0 <= cx < self.obstacle_map.grid_w and 0 <= cy < self.obstacle_map.grid_h:
                if hasattr(self.obstacle_map, 'cost_map') and self.obstacle_map.cost_map[cy, cx] < SAFE_COST:
                    wx, wy = self.obstacle_map.grid_to_world(cx, cy)
                    return np.array([wx, wy])
            if abs(cx - gx_t) > search_radius_grid or abs(cy - gy_t) > search_radius_grid: continue
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny)); queue.append((nx, ny))
        return None

    def sanitize_target(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        current_cost = self.obstacle_map.get_cost_at(target_pos)
        # [TUNE] Cost < 400 là OK (Cách tường ~20cm)
        if current_cost < 400.0: return target_pos
        safe_pos = self.find_nearest_walkable(target_pos, search_radius_grid=20)
        return safe_pos

    def get_carrot_waypoint(self, path: List[np.ndarray], lookahead_dist: float = 80.0) -> np.ndarray:
        if not path: return self.drone.estimated_pos
        final_pt = path[-1]
        dist_to_final = np.linalg.norm(self.drone.estimated_pos - final_pt)
        if dist_to_final < lookahead_dist: return final_pt
        best_carrot = path[0]
        for wp in path:
            if np.linalg.norm(self.drone.estimated_pos - wp) > lookahead_dist:
                best_carrot = wp; break
        return best_carrot

    def get_next_waypoint(self, final_target: np.ndarray, force_replan: bool = False) -> Optional[np.ndarray]:
        if final_target is None: return None
        
        safe_target = self.sanitize_target(final_target)
        if safe_target is None: return None
        if np.linalg.norm(safe_target - final_target) > 1.0: final_target = safe_target
        
        if self.failure_cooldown > 0:
            self.failure_cooldown -= 1
            if self.current_astar_path: return self.get_carrot_waypoint(self.current_astar_path)
            return None

        if self.drone.state in ['RETURNING', 'DROPPING', 'END_GAME']:
            self.dijkstra_update_timer += 1
            should_update_map = False
            if self.dijkstra_target_cached is None or np.linalg.norm(final_target - self.dijkstra_target_cached) > 20.0: should_update_map = True
            elif self.dijkstra_update_timer > 40:
                if (self.dijkstra_update_timer + self.drone.identifier * 5) % 10 == 0: should_update_map = True

            if should_update_map:
                self.obstacle_map.update_dijkstra_map(final_target)
                self.dijkstra_target_cached = final_target.copy()
                self.dijkstra_update_timer = 0
            
            raw_path = self.obstacle_map.get_dijkstra_path(self.drone.estimated_pos, max_steps=40)
            if len(raw_path) > 0: self.current_astar_path = raw_path
            else: self.current_astar_path = []

        else:
            self.replan_timer += 1
            need_replan = False
            if force_replan: need_replan = True
            elif self.last_astar_target is None: need_replan = True
            elif np.linalg.norm(final_target - self.last_astar_target) > 30.0: need_replan = True
            elif len(self.current_astar_path) == 0: need_replan = True
            elif self.replan_timer > 50: need_replan = True

            if need_replan:
                self.replan_timer = 0
                self.current_astar_path = self.obstacle_map.find_path_astar(self.drone.estimated_pos, final_target)
                if not self.current_astar_path:
                    safe_target = self.find_nearest_walkable(final_target, search_radius_grid=15)
                    if safe_target is not None:
                        self.current_astar_path = self.obstacle_map.find_path_astar(self.drone.estimated_pos, safe_target)
                self.last_astar_target = final_target.copy()
                if not self.current_astar_path: self.failure_cooldown = 30; return None

        # [NEW LOGIC] ADAPTIVE PRUNING (Cắt bỏ điểm thừa)
        # Mục tiêu: Nếu drone đã lướt qua điểm 1, 2 và đang ở gần điểm 3 -> Xóa 1, 2 đi.
        if self.current_astar_path:
            # Chỉ xét 20 điểm đầu tiên (Look window) để tránh nhận nhầm điểm ở đường song song bên kia tường
            search_limit = min(len(self.current_astar_path), 20)
            subset = self.current_astar_path[:search_limit]
            
            # Tìm chỉ số (index) của điểm gần drone nhất hiện tại
            dists = [np.linalg.norm(self.drone.estimated_pos - wp) for wp in subset]
            closest_idx = np.argmin(dists)
            
            # Cắt bỏ toàn bộ các điểm nằm trước điểm gần nhất
            if closest_idx > 0:
                self.current_astar_path = self.current_astar_path[closest_idx:]

        # [LOGIC CŨ] Filter points too close (Giữ lại để dọn nốt điểm hiện tại nếu quá gần)
        while len(self.current_astar_path) > 0:
            if np.linalg.norm(self.drone.estimated_pos - self.current_astar_path[0]) < 20.0:
                self.current_astar_path.pop(0)
            else: break
        
        if not self.current_astar_path: return final_target

        # [TUNE] CARROT
        LOOKAHEAD = 80.0 
        if self.drone.state == 'RESCUING': LOOKAHEAD = 40.0
        
        carrot_wp = self.get_carrot_waypoint(self.current_astar_path, lookahead_dist=LOOKAHEAD)
        
        # [FIX] check_cost=True để đảm bảo không cắt góc tường
        # Cho phép đi qua vùng Cost < 300 (Hơi sát nhưng vẫn đi được nhờ Wall Repulsion của Pilot)
        is_safe_line = self.obstacle_map.check_line_of_sight(
            self.drone.estimated_pos, 
            carrot_wp, 
            safety_radius=1, # Giảm radius check xuống 1 để lọt khe hẹp
            check_cost=True  # BẮT BUỘC CHECK COST
        )

        if not is_safe_line:
            # Nếu đường thẳng tới Cà rốt bị vướng, fallback về điểm gần nhất
            return self.current_astar_path[0] 
            
        return carrot_wp