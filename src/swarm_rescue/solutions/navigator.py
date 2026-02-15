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

        # [FIX] 1. Lấy dữ liệu Semantic TRƯỚC để lọc vị trí nạn nhân
        semantic_data = self.drone.semantic_values()
        nearby_victims = []
        if semantic_data:
            from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    # Tính tọa độ Global của nạn nhân
                    angle_global = self.drone.estimated_angle + data.angle
                    vx = self.drone.estimated_pos[0] + data.distance * math.cos(angle_global)
                    vy = self.drone.estimated_pos[1] + data.distance * math.sin(angle_global)
                    nearby_victims.append(np.array([vx, vy]))

        # [FIX] 2. Update Obstacle Map (Truyền thêm victims để lọc khỏi tường)
        lidar_data = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        if lidar_data is not None:
            self.obstacle_map.update_from_lidar(
                self.drone.estimated_pos, 
                self.drone.estimated_angle,
                lidar_data, 
                lidar_angles,
                nearby_drones_pos=nearby_drones,
                nearby_victims_pos=nearby_victims # <--- THAM SỐ MỚI
            )
            
        # 3. Update Victim Map (Giữ nguyên)
        self.victim_map.update_from_semantic(
            self.drone.estimated_pos,
            self.drone.estimated_angle,
            semantic_data
        )

    def find_nearest_walkable(self, target_pos: np.ndarray, search_radius_grid: int = 10) -> Optional[np.ndarray]:
        """
        Tìm một điểm grid gần target_pos nhất mà có Cost thấp (đi được).
        Dùng BFS để loang ra từ target.
        """
        gx_t, gy_t = self.obstacle_map.world_to_grid(target_pos[0], target_pos[1])
        
        # Hàng đợi BFS: (gx, gy)
        queue = [(gx_t, gy_t)]
        visited = set([(gx_t, gy_t)])
        
        # Cost an toàn (tương đương cách tường ~30cm)
        SAFE_COST = 100.0 
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Check xem ô này có an toàn không
            if 0 <= cx < self.obstacle_map.grid_w and 0 <= cy < self.obstacle_map.grid_h:
                # Nếu cost thấp -> Đây là điểm gần nhất đi được!
                if hasattr(self.obstacle_map, 'cost_map') and self.obstacle_map.cost_map[cy, cx] < SAFE_COST:
                    wx, wy = self.obstacle_map.grid_to_world(cx, cy)
                    return np.array([wx, wy])
            
            # Nếu chưa tìm thấy, loang ra các ô xung quanh
            # Chỉ loang trong bán kính giới hạn để đỡ tốn CPU
            if abs(cx - gx_t) > search_radius_grid or abs(cy - gy_t) > search_radius_grid:
                continue

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
        return None

    def get_next_waypoint(self, final_target: np.ndarray, force_replan: bool = False) -> Optional[np.ndarray]:
        if final_target is None: return None
        
        if self.failure_cooldown > 0:
            self.failure_cooldown -= 1
            if self.current_astar_path: return self.current_astar_path[0]
            return None

        # =========================================================
        # 1. CHẾ ĐỘ VỀ NHÀ (Dijkstra Flow Field)
        # =========================================================
        if self.drone.state in ['RETURNING', 'DROPPING', 'END_GAME']:
            self.dijkstra_update_timer += 1
            
            should_update_map = False
            # Update NGAY nếu chưa có map hoặc đích thay đổi xa
            if self.dijkstra_target_cached is None or np.linalg.norm(final_target - self.dijkstra_target_cached) > 20.0:
                should_update_map = True
            # Update ĐỊNH KỲ
            elif self.dijkstra_update_timer > 40:
                if (self.dijkstra_update_timer + self.drone.identifier * 5) % 10 == 0:
                     should_update_map = True

            if should_update_map:
                self.obstacle_map.update_dijkstra_map(final_target)
                self.dijkstra_target_cached = final_target.copy()
                self.dijkstra_update_timer = 0
            
            raw_path = self.obstacle_map.get_dijkstra_path(self.drone.estimated_pos, max_steps=40)
            
            if len(raw_path) > 0:
                self.current_astar_path = raw_path
            else:
                self.current_astar_path = []
                # Nếu không tìm thấy đường về -> Đừng return None vội, có thể đang ở rất gần đích
                # Hãy để logic smoothing bên dưới xử lý hoặc trôi theo quán tính
                if np.linalg.norm(self.drone.estimated_pos - final_target) < 50:
                    return final_target
                return None

        # =========================================================
        # 2. CHẾ ĐỘ KHÁM PHÁ (Cost-based A*)
        # =========================================================
        else:
            self.replan_timer += 1
            need_replan = False
            if force_replan: need_replan = True
            elif self.last_astar_target is None: need_replan = True
            elif np.linalg.norm(final_target - self.last_astar_target) > 30.0: need_replan = True
            elif len(self.current_astar_path) == 0: need_replan = True
            elif self.replan_timer > 100: need_replan = True

            if need_replan:
                self.replan_timer = 0
                self.current_astar_path = self.obstacle_map.find_path_astar(
                    self.drone.estimated_pos, 
                    final_target
                )

                # [NEW] NẾU KHÔNG TÌM ĐƯỢC ĐƯỜNG (Do đích kẹt trong tường)
                if not self.current_astar_path:
                    # Hãy thử tìm một điểm "thoáng" gần đích nhất
                    # print(f"[{self.drone.identifier}] ⚠️ Target unreachable. Searching nearest walkable...")
                    safe_target = self.find_nearest_walkable(final_target, search_radius_grid=15)
                    
                    if safe_target is not None:
                        # Thử tìm đường đến điểm an toàn đó
                        self.current_astar_path = self.obstacle_map.find_path_astar(
                            self.drone.estimated_pos, 
                            safe_target
                        )

                self.last_astar_target = final_target.copy()
                
                if not self.current_astar_path:
                    self.failure_cooldown = 30
                    return None

        # =========================================================
        # 3. LOC BỎ CÁC ĐIỂM QUÁ GẦN (QUAN TRỌNG ĐỂ TRÁNH DIST=0.0)
        # =========================================================
        # Loại bỏ các điểm đầu tiên nếu nó cách drone < 20cm
        while len(self.current_astar_path) > 0:
            if np.linalg.norm(self.drone.estimated_pos - self.current_astar_path[0]) < 20.0:
                self.current_astar_path.pop(0)
            else:
                break
        
        # Nếu sau khi lọc mà hết đường -> Nghĩa là đã đến rất gần đích
        if not self.current_astar_path: 
            return final_target

        # =========================================================
        # 4. SMOOTHING (Lookahead)
        # =========================================================
        LOOKAHEAD_DIST = 100.0 
        if self.drone.state == 'RESCUING': LOOKAHEAD_DIST = 40.0
        
        check_radius = 2
        # Khi Return: Map an toàn hơn, nhưng vẫn check kỹ để không đâm tường
        if self.drone.state in ['RETURNING', 'DROPPING']: 
            check_radius = 1 
        
        best_wp = self.current_astar_path[0]
        
        # [FIX] Nếu điểm đầu tiên vẫn còn quá gần (<30cm) sau khi lọc -> Bắt buộc lấy điểm xa hơn
        # bằng cách nới lỏng check_cost cho đoạn ngắn
        
        for i, wp in enumerate(self.current_astar_path):
            dist = np.linalg.norm(self.drone.estimated_pos - wp)
            if dist < LOOKAHEAD_DIST:
                # Logic: Nếu điểm rất gần (<50cm), ta cho phép đi qua vùng Cost cao nhẹ (check_cost=False)
                # để drone thoát ra khỏi chỗ chật hẹp. Nếu xa hơn thì phải an toàn (check_cost=True)
                do_strict_check = True
                if dist < 50.0 and i < 3: 
                    do_strict_check = False # Nới lỏng kiểm tra cho vài điểm đầu tiên

                if self.obstacle_map.check_line_of_sight(self.drone.estimated_pos, wp, safety_radius=check_radius, check_cost=do_strict_check):
                    best_wp = wp
                else:
                    break
            else:
                break
        
        return best_wp