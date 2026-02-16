import math
import numpy as np
from typing import List, Tuple, Optional
from swarm_rescue.simulation.utils.utils import normalize_angle

try:
    from .mapping import GridMap
except ImportError:
    from mapping import GridMap

class Navigator:
    def __init__(self, drone):
        self.drone = drone
        self.gps_last_known = None
        
        map_size = getattr(self.drone, 'map_size', (100, 100))
        self.obstacle_map = GridMap(map_size=map_size)
        
        self.current_astar_path = []
        self.last_astar_target = None 
        
        # [NEW] Biến ghi nhớ tiến độ trên đường đi
        self.last_path_index = 0 
        
        self.replan_timer = 0
        self.failure_cooldown = 0 
        self.cached_nearby_drones = []
        self.dijkstra_update_timer = 0
        self.dijkstra_target_cached = None

    def update_navigator(self, nearby_drones: List[np.ndarray] = []):
        # ... (Giữ nguyên phần update GPS/Odom/Lidar) ...
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()
        if gps_pos is not None and compass_angle is not None:
            self.drone.estimated_pos = gps_pos
            self.drone.estimated_angle = compass_angle
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
        
        lidar_data = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        if lidar_data is not None:
            self.obstacle_map.update_from_lidar(self.drone.estimated_pos, self.drone.estimated_angle, lidar_data, lidar_angles)

    # ... (Giữ nguyên find_nearest_walkable và sanitize_target) ...
    def find_nearest_walkable(self, target_pos: np.ndarray, search_radius_grid: int = 10) -> Optional[np.ndarray]:
        gx_t, gy_t = self.obstacle_map.world_to_grid(target_pos[0], target_pos[1])
        queue = [(gx_t, gy_t)]; visited = set([(gx_t, gy_t)]); SAFE_COST = 200.0 
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
        if current_cost < 400.0: return target_pos
        safe_pos = self.find_nearest_walkable(target_pos, search_radius_grid=20)
        return safe_pos

    # [UPDATE] Hàm Carrot tìm điểm tiếp theo dựa trên index cũ
    def get_carrot_waypoint(self, path: List[np.ndarray], lookahead_dist: float = 80.0) -> np.ndarray:
        if not path: return self.drone.estimated_pos
        
        # Tìm điểm gần nhất trên path, nhưng CHỈ quét từ vị trí lần trước (last_path_index)
        # Giới hạn cửa sổ tìm kiếm (ví dụ: chỉ tìm trong 10 điểm tiếp theo) để tránh nhảy cóc
        search_range = 20
        start_idx = self.last_path_index
        end_idx = min(len(path), start_idx + search_range)
        
        closest_dist = float('inf')
        current_closest_idx = start_idx

        # Tìm điểm gần nhất trong đoạn cho phép
        for i in range(start_idx, end_idx):
            dist = np.linalg.norm(self.drone.estimated_pos - path[i])
            if dist < closest_dist:
                closest_dist = dist
                current_closest_idx = i
        
        # Cập nhật tiến độ: Không bao giờ lùi lại
        self.last_path_index = current_closest_idx

        # Tìm Cà rốt: Điểm đầu tiên nằm ngoài vòng tròn Lookahead, tính từ vị trí gần nhất
        best_carrot = path[-1]
        for i in range(current_closest_idx, len(path)):
            d_to_carrot = np.linalg.norm(self.drone.estimated_pos - path[i])
            if d_to_carrot > lookahead_dist:
                best_carrot = path[i]
                break
                
        return best_carrot
    
    def get_adaptive_lookahead(self) -> float:
        # Lấy cost tại vị trí hiện tại của drone
        current_cost = self.obstacle_map.get_cost_at(self.drone.estimated_pos)
        
        # Nếu đang ở vùng thoáng (Cost thấp) -> Nhìn xa (100cm)
        if current_cost < 10.0: 
            return 110.0
        # Nếu đang ở gần tường hoặc khe hẹp (Cost cao) -> Nhìn gần (40cm)
        elif current_cost > 100.0:
            return 40.0
        # Nội suy tuyến tính
        else:
            return 110.0 - (current_cost / 100.0) * 70.0

    def get_next_waypoint(self, final_target: np.ndarray, force_replan: bool = False) -> Optional[np.ndarray]:
        if final_target is None: return None
        
        safe_target = self.sanitize_target(final_target)
        if safe_target is None: return None
        if np.linalg.norm(safe_target - final_target) > 1.0: final_target = safe_target
        
        if self.failure_cooldown > 0:
            self.failure_cooldown -= 1
            if self.current_astar_path: return self.get_carrot_waypoint(self.current_astar_path)
            return None

        # Logic Replan (Giữ nguyên)
        if self.drone.state in ['RETURNING', 'DROPPING', 'END_GAME']:
            # ... (Dijkstra Logic cũ giữ nguyên) ...
            self.dijkstra_update_timer += 1
            should_update_map = False
            if self.dijkstra_target_cached is None or np.linalg.norm(final_target - self.dijkstra_target_cached) > 20.0: should_update_map = True
            elif self.dijkstra_update_timer > 40: should_update_map = True

            if should_update_map:
                self.obstacle_map.update_dijkstra_map(final_target)
                self.dijkstra_target_cached = final_target.copy()
                self.dijkstra_update_timer = 0
            
            raw_path = self.obstacle_map.get_dijkstra_path(self.drone.estimated_pos, max_steps=40)
            if len(raw_path) > 0: 
                self.current_astar_path = raw_path
                self.last_path_index = 0 # Reset index khi có path mới
            else: self.current_astar_path = []

        else: # A* Logic
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
                self.last_path_index = 0 # [IMPORTANT] Reset index khi có đường mới
                if not self.current_astar_path: self.failure_cooldown = 30; return None

        if not self.current_astar_path: return final_target

        # [UPDATE] Dùng Dynamic Lookahead thay vì số cứng
        if self.drone.state == 'RESCUING': 
            LOOKAHEAD = 30.0 # Khi cứu người thì luôn cần chính xác
        else:
            LOOKAHEAD = self.get_adaptive_lookahead()
        
        # Gọi hàm Carrot mới (có tính progress)
        carrot_wp = self.get_carrot_waypoint(self.current_astar_path, lookahead_dist=LOOKAHEAD)
        
        # Check LOS
        if not self.obstacle_map.check_line_of_sight(self.drone.estimated_pos, carrot_wp, safety_radius=2, check_cost=False):
            # Fallback: Nếu không nhìn thấy Carrot xa, lấy điểm gần hơn trên path
            fallback_idx = min(self.last_path_index + 2, len(self.current_astar_path)-1)
            return self.current_astar_path[fallback_idx]
            
        return carrot_wp