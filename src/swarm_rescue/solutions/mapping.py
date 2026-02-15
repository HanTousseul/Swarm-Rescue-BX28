import math
import heapq
import numpy as np
import cv2
from typing import List, Tuple, Optional

# --- CONFIGURATION ---
RESOLUTION = 8     
MAX_LIDAR_RANGE = 300 
VAL_EMPTY = -0.5    
VAL_OBSTACLE = 2.0  
VAL_FREE = -2.0     
THRESHOLD_MIN = -50.0
THRESHOLD_MAX = 50.0

class GridMap:
    def __init__(self, map_size: Tuple[int, int], resolution: int = RESOLUTION):
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.resolution = resolution
        self.grid_w = int(self.map_width / self.resolution) + 1
        self.grid_h = int(self.map_height / self.resolution) + 1
        self.offset_x = self.map_width / 2.0
        self.offset_y = self.map_height / 2.0
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Cache cho bản đồ chi phí và dòng chảy
        self.cost_map = None
        self.dijkstra_grid = None

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x + self.offset_x) / self.resolution)
        gy = int((y + self.offset_y) / self.resolution)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = (gx * self.resolution) - self.offset_x + (self.resolution / 2)
        wy = (gy * self.resolution) - self.offset_y + (self.resolution / 2)
        return wx, wy

    # [FIX] Thêm tham số nearby_victims_pos
    def update_from_lidar(self, drone_pos: np.ndarray, drone_angle: float, lidar_data: List[float], lidar_angles: List[float], nearby_drones_pos: List[np.ndarray] = [], nearby_victims_pos: List[np.ndarray] = []):
        if lidar_data is None or lidar_angles is None: return
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        update_layer = np.zeros_like(self.grid)
        step = 3 
        
        DRONE_RADIUS_IGNORE = 40.0 
        VICTIM_RADIUS_IGNORE = 30.0 # [NEW] Bán kính của nạn nhân (~30cm)

        for i in range(0, len(lidar_data), step):
            dist = lidar_data[i]
            angle = lidar_angles[i] + drone_angle 
            LIDAR_DIST_CLIP = 40.0
            dist_empty = max(0.0, dist - LIDAR_DIST_CLIP)
            empty_x = drone_pos[0] + dist_empty * math.cos(angle)
            empty_y = drone_pos[1] + dist_empty * math.sin(angle)
            ex, ey = self.world_to_grid(empty_x, empty_y)
            cv2.line(update_layer, (cx, cy), (ex, ey), VAL_EMPTY, thickness=1)
            
            if dist < (MAX_LIDAR_RANGE - 5.0):
                obs_x = drone_pos[0] + dist * math.cos(angle)
                obs_y = drone_pos[1] + dist * math.sin(angle)
                
                is_ignored = False

                # 1. Check Drone (Giữ nguyên)
                for d_pos in nearby_drones_pos:
                    if math.hypot(obs_x - d_pos[0], obs_y - d_pos[1]) < DRONE_RADIUS_IGNORE:
                        is_ignored = True; break
                
                # 2. [NEW] Check Victim (Nếu trùng vị trí nạn nhân -> Bỏ qua)
                if not is_ignored:
                    for v_pos in nearby_victims_pos:
                        if math.hypot(obs_x - v_pos[0], obs_y - v_pos[1]) < VICTIM_RADIUS_IGNORE:
                            is_ignored = True; break
                
                ox, oy = self.world_to_grid(obs_x, obs_y)
                if 0 <= ox < self.grid_w and 0 <= oy < self.grid_h:
                    if not is_ignored:
                        update_layer[oy, ox] = VAL_OBSTACLE # Vẽ là tường
                    else:
                        update_layer[oy, ox] = VAL_FREE # Vẽ là KHÔNG KHÍ (để Cost thấp)

        cv2.circle(update_layer, (cx, cy), 2, VAL_FREE, -1)
        self.grid += update_layer
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

    def update_from_remote_points(self, points: List[Tuple[float, float]]):
        for (wx, wy) in points:
            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                if self.grid[gy, gx] < 20.0:
                    self.grid[gy, gx] += 2.0 

    # =========================================================
    # CORE LOGIC: COST MAP & DIJKSTRA MAP
    # =========================================================

    # [FIX] Thêm phạt nặng cho vùng Unknown để Flow Field ưu tiên đường đã khám phá
    def update_cost_map(self):
        """
        Tạo bản đồ chi phí: 
        1. Càng gần tường -> Giá càng đắt (Safety).
        2. Vùng chưa khám phá -> Giá rất đắt (Unknown Penalty).
        """
        # 1. Tính Safety Cost (Dựa trên khoảng cách tường)
        # Tạo ảnh nhị phân: Tường (>20) là 0, Đường là 1
        binary_grid = (self.grid <= 20.0).astype(np.uint8)
        
        # Tính khoảng cách tới tường
        self.dist_map = cv2.distanceTransform(binary_grid, cv2.DIST_L2, 5)
        
        # Cost cơ bản dựa trên độ an toàn
        SAFETY_WEIGHT = 400.0 
        self.cost_map = 1.0 + (SAFETY_WEIGHT / (self.dist_map + 0.1))
        
        # 2. [NEW] Thêm phạt cho vùng UNKNOWN
        # Vùng Unknown thường có giá trị 0.0 (hoặc xấp xỉ 0)
        # Vùng Free (đã quét) thường có giá trị âm (ví dụ -2.0, -0.5)
        # Ta phạt nặng những ô có giá trị > -1.0 (tức là Unknown hoặc Obstacle nhẹ)
        
        # Mask cho vùng Unknown (Lớn hơn -1 nhưng không phải tường cứng > 20)
        unknown_mask = (self.grid > -1.0) & (self.grid <= 20.0)
        
        UNKNOWN_PENALTY = 100.0 # Giá phạt đắt đỏ (tương đương đi thêm 100 bước)
        self.cost_map[unknown_mask] += UNKNOWN_PENALTY
        
        # 3. Chặn tường tuyệt đối
        self.cost_map[self.grid > 20.0] = 9999.0

    def update_dijkstra_map(self, target_pos: np.ndarray):
        """
        Tính Flow Field từ đích lan ra toàn map (Dùng cho Returning).
        """
        self.update_cost_map()
        self.dijkstra_grid = np.full_like(self.grid, np.inf)
        
        gx_t, gy_t = self.world_to_grid(target_pos[0], target_pos[1])
        
        # Nếu đích bị kẹt trong tường, tìm điểm thoáng gần nhất
        if self.cost_map[gy_t, gx_t] > 1000:
             found = False
             for r in range(1, 10):
                 for dy in range(-r, r+1):
                     for dx in range(-r, r+1):
                         ny, nx = gy_t + dy, gx_t + dx
                         if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                             if self.cost_map[ny, nx] < 100: 
                                 gy_t, gx_t = ny, nx
                                 found = True; break
                     if found: break
                 if found: break

        # Dijkstra Initialization
        self.dijkstra_grid[gy_t, gx_t] = 0.0
        pq = [(0.0, gx_t, gy_t)] 
        neighbors = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        while pq:
            curr_val, cx, cy = heapq.heappop(pq)
            if curr_val > self.dijkstra_grid[cy, cx]: continue
            
            for dx, dy, dist_w in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    step_risk = self.cost_map[ny, nx] 
                    if step_risk >= 9999.0: continue
                    
                    new_val = curr_val + (dist_w * step_risk)
                    if new_val < self.dijkstra_grid[ny, nx]:
                        self.dijkstra_grid[ny, nx] = new_val
                        heapq.heappush(pq, (new_val, nx, ny))

    def get_dijkstra_path(self, current_pos: np.ndarray, max_steps: int = 40) -> List[np.ndarray]:
        if not hasattr(self, 'dijkstra_grid') or self.dijkstra_grid is None: return []
        
        cx, cy = self.world_to_grid(current_pos[0], current_pos[1])
        
        # [FIX] KHÔNG thêm current_pos vào path nữa. 
        # path = [current_pos] <-- XÓA DÒNG NÀY HOẶC COMMENT LẠI
        path = [] 
        
        curr_x, curr_y = cx, cy
        
        for _ in range(max_steps):
            min_val = self.dijkstra_grid[curr_y, curr_x]
            best_n = None
            
            # Quét 8 ô xung quanh
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = curr_x + dx, curr_y + dy
                    
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        val = self.dijkstra_grid[ny, nx]
                        # Phải nhỏ hơn ô hiện tại đáng kể mới bước tới (tránh dao động)
                        if val < min_val:
                            min_val = val
                            best_n = (nx, ny)
            
            if best_n:
                curr_x, curr_y = best_n
                wx, wy = self.grid_to_world(curr_x, curr_y)
                path.append(np.array([wx, wy]))
                
                # Nếu đã về đến đích (giá trị ~ 0) thì dừng
                if min_val <= 0.0:
                    break
            else:
                # Không tìm thấy đường xuống dốc -> Dừng
                break
                
        return path

    # [UPDATE] A* sử dụng Cost Map (Robust hơn margin cứng)
    def find_path_astar(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        ex, ey = self.world_to_grid(end_pos[0], end_pos[1])
        
        # Luôn đảm bảo cost map mới nhất
        self.update_cost_map()

        if self.cost_map[ey, ex] > 1000: return [] # Đích là tường

        open_list = []
        heapq.heappush(open_list, (0, sx, sy))
        came_from = {}
        g_score = { (sx, sy): 0 }
        
        neighbors = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)]
        
        while open_list:
            _, cx, cy = heapq.heappop(open_list)
            
            if (cx, cy) == (ex, ey):
                path = []
                curr = (ex, ey)
                while curr in came_from:
                    wx, wy = self.grid_to_world(curr[0], curr[1])
                    path.append(np.array([wx, wy]))
                    curr = came_from[curr]
                path.reverse()
                return path
            
            for dx, dy, move_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    # Lấy Cost từ bản đồ rủi ro
                    cell_cost = self.cost_map[ny, nx]
                    if cell_cost >= 9999.0: continue 

                    new_g = g_score[(cx, cy)] + (move_cost * cell_cost)
                    
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = math.hypot(ex - nx, ey - ny) 
                        heapq.heappush(open_list, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
        return []
    
    # [UPDATE] Thêm tham số nearby_drones để tính "Vùng ảnh hưởng" (Influence Map)
    def get_frontier_target(self, drone_pos: np.ndarray, drone_angle: float, busy_targets: List[dict] = [], nearby_drones: List[np.ndarray] = []) -> Optional[np.ndarray]:
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        candidates = []
        step = 3 
        
        # 1. Lọc Frontier (Giữ nguyên logic cũ)
        for y in range(0, self.grid_h, step):
            for x in range(0, self.grid_w, step):
                if -5.0 < self.grid[y, x] < 5.0: # Vùng Unknown
                    free_neighbors = 0
                    unknown_neighbors = 0
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                            val = self.grid[ny, nx]
                            if val < -1.0: free_neighbors += 1
                            elif -5.0 < val < 5.0: unknown_neighbors += 1
                    
                    if free_neighbors >= 1 and unknown_neighbors >= 1:
                        candidates.append((x, y))

        if not candidates: return None

        best_target = None
        min_cost = float('inf')
        
        MIN_DIST_PREFERRED = 80.0
        ANGLE_WEIGHT = 40.0       
        DENSITY_REWARD = 50.0 
        WALL_PENALTY_WEIGHT = 2000.0
        WALL_CHECK_RADIUS = 4
        
        # [NEW] CROWD PENALTY: Bán kính ảnh hưởng của drone khác (150px ~ 1.5m)
        CROWD_RADIUS = 150.0 
        CROWD_PENALTY = 2000.0 # Phạt cực nặng nếu chọn điểm gần drone bạn

        from swarm_rescue.simulation.utils.utils import normalize_angle
        
        for (gx, gy) in candidates:
            wx, wy = self.grid_to_world(gx, gy)
            
            # Check Busy Targets (Đích đến của người khác)
            is_busy = False
            for item in busy_targets:
                if math.hypot(wx - item['pos'][0], wy - item['pos'][1]) < 60.0:
                    is_busy = True; break
            if is_busy: continue

            # [NEW] STRATEGY 1: INFLUENCE MAP (Tránh vùng đồng đội đang đứng)
            crowd_cost = 0
            for d_pos in nearby_drones:
                d_dist = math.hypot(wx - d_pos[0], wy - d_pos[1])
                if d_dist < CROWD_RADIUS:
                    # Càng gần càng phạt nặng
                    crowd_cost += CROWD_PENALTY * (1.0 - d_dist/CROWD_RADIUS)

            # Tính các chi phí cơ bản
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            dist_penalty = 0
            if dist < 20.0: dist_penalty = 10000.0 
            elif dist < MIN_DIST_PREFERRED: dist_penalty = 500.0   
            
            angle_to_target = math.atan2(wy - drone_pos[1], wx - drone_pos[0])
            angle_diff = abs(normalize_angle(angle_to_target - drone_angle))
            
            wall_penalty = 0
            y_w_min = max(0, gy - WALL_CHECK_RADIUS)
            y_w_max = min(self.grid_h, gy + WALL_CHECK_RADIUS + 1)
            x_w_min = max(0, gx - WALL_CHECK_RADIUS)
            x_w_max = min(self.grid_w, gx + WALL_CHECK_RADIUS + 1)
            if np.max(self.grid[y_w_min:y_w_max, x_w_min:x_w_max]) > 20.0:
                wall_penalty = WALL_PENALTY_WEIGHT

            range_check = 2 
            y_min = max(0, gy - range_check)
            y_max = min(self.grid_h, gy + range_check + 1)
            x_min = max(0, gx - range_check)
            x_max = min(self.grid_w, gx + range_check + 1)
            unknown_count = np.sum(np.abs(self.grid[y_min:y_max, x_min:x_max]) < 5.0)
            density_bonus = unknown_count * DENSITY_REWARD

            # Tổng hợp chi phí
            cost = dist + (angle_diff * ANGLE_WEIGHT) + dist_penalty - density_bonus + wall_penalty + crowd_cost
            
            if cost < min_cost:
                min_cost = cost
                best_target = np.array([wx, wy])
                
        return best_target

    # [NEW] STRATEGY 2: FALLBACK TARGET (Tìm vùng tối bất kỳ)
    def get_unknown_target(self, drone_pos: np.ndarray, nearby_drones: List[np.ndarray] = []) -> Optional[np.ndarray]:
        """
        Nếu không tìm được Frontier đẹp, hãy tìm bất kỳ ô Unknown nào còn sót lại.
        Ưu tiên ô xa drone bạn để phân tán đội hình.
        """
        # Tìm tất cả các ô có giá trị gần 0 (-5 < val < 5)
        # np.argwhere trả về danh sách [y, x]
        unknown_indices = np.argwhere(np.abs(self.grid) < 5.0)
        
        if len(unknown_indices) == 0:
            return None # Map đã sáng hết 100%

        # Chọn ngẫu nhiên 50 điểm để check cho nhanh (thay vì check cả nghìn điểm)
        num_samples = min(len(unknown_indices), 50)
        indices = np.random.choice(len(unknown_indices), num_samples, replace=False)
        
        best_target = None
        max_score = -float('inf')
        
        for idx in indices:
            gy, gx = unknown_indices[idx]
            wx, wy = self.grid_to_world(gx, gy)
            
            # 1. Khoảng cách tới bản thân (Ưu tiên gần để tiện đi, hoặc xa để mở rộng - tùy chiến thuật)
            # Ở đây ta chọn ngẫu nhiên nên không quan trọng lắm distance
            
            # 2. Quan trọng: Phải XA các drone khác (Dispersion)
            min_dist_to_friend = float('inf')
            for d_pos in nearby_drones:
                d = math.hypot(wx - d_pos[0], wy - d_pos[1])
                if d < min_dist_to_friend: min_dist_to_friend = d
            
            # Score cao nếu xa bạn bè -> Tự động tản ra chỗ vắng
            score = min_dist_to_friend 
            
            if score > max_score:
                max_score = score
                best_target = np.array([wx, wy])
                
        return best_target

    def get_random_free_target(self, drone_pos: np.ndarray, min_dist: float = 200.0) -> Optional[np.ndarray]:
        free_indices = np.argwhere(self.grid < -40.0)
        if len(free_indices) == 0: return None
        np.random.shuffle(free_indices)
        for (gy, gx) in free_indices:
            wx, wy = self.grid_to_world(gx, gy)
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            if dist > min_dist: return np.array([wx, wy])
        return None

    def check_line_of_sight(self, start_pos: np.ndarray, end_pos: np.ndarray, safety_radius: int = 1, check_cost: bool = True) -> bool:
        x0, y0 = start_pos
        x1, y1 = end_pos
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1.0: return True 
        
        steps = int(dist / (self.resolution / 2)) 
        if steps == 0: steps = 1
        
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        
        # Ngưỡng an toàn (Cost < 50 nghĩa là cách tường khoảng 5 ô ~ 40cm)
        SAFE_COST_THRESHOLD = 50.0 
        
        for x, y in zip(xs, ys):
            gx, gy = self.world_to_grid(x, y)
            
            # Check bán kính (Collision)
            for dy in range(-safety_radius, safety_radius + 1):
                for dx in range(-safety_radius, safety_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        # 1. Check Tường Cứng (Luôn check)
                        if self.grid[ny, nx] > 10.0: return False
                        
                        # 2. Check Vùng Nguy Hiểm (Chỉ check nếu được yêu cầu)
                        if check_cost and hasattr(self, 'cost_map') and self.cost_map is not None:
                            if self.cost_map[ny, nx] > SAFE_COST_THRESHOLD:
                                return False # Đường cắt qua vùng quá gần tường -> Hủy
        return True

    def get_cost_at(self, world_pos: np.ndarray) -> float:
        if not hasattr(self, 'cost_map') or self.cost_map is None: return 1.0 
        gx, gy = self.world_to_grid(world_pos[0], world_pos[1])
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h: return self.cost_map[gy, gx]
        return 9999.0

    def display(self, drone_pos: np.ndarray, current_target: Optional[np.ndarray] = None, current_path: List[np.ndarray] = [], window_name="Obstacle Map"):
        normalized_grid = (self.grid - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)
        normalized_grid = np.clip(normalized_grid, 0.0, 1.0) * 255.0
        norm_grid = normalized_grid.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(norm_grid, cv2.COLORMAP_JET)
        if current_path:
            pts = []
            for wp in current_path:
                px, py = self.world_to_grid(wp[0], wp[1])
                pts.append([px, py])
            if len(pts) > 1:
                pts_np = np.array([pts], dtype=np.int32)
                cv2.polylines(heatmap_img, pts_np, isClosed=False, color=(255, 0, 0), thickness=1)
        if current_target is not None:
            tx, ty = self.world_to_grid(current_target[0], current_target[1])
            cv2.circle(heatmap_img, (tx, ty), 3, (0, 255, 0), -1) 
        gx, gy = self.world_to_grid(drone_pos[0], drone_pos[1])
        cv2.circle(heatmap_img, (gx, gy), 2, (0, 0, 0), -1)
        target_width = 800; scale = target_width / heatmap_img.shape[1]; target_height = int(heatmap_img.shape[0] * scale)
        display_img = cv2.resize(heatmap_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        display_img = cv2.flip(display_img, 0) 
        cv2.imshow(window_name, display_img); cv2.waitKey(1)

class VictimHeatmap: # GIỮ NGUYÊN CLASS NÀY CỦA BẠN
    def __init__(self, map_size: Tuple[int, int], resolution: int = 8):
        self.map_width = map_size[0]; self.map_height = map_size[1]; self.resolution = resolution
        self.offset_x = self.map_width / 2.0; self.offset_y = self.map_height / 2.0
        self.grid_w = int(self.map_width / self.resolution) + 1; self.grid_h = int(self.map_height / self.resolution) + 1
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.decay_rate = 1.0; self.score_add = 10.0; self.max_score = 100.0
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x + self.offset_x) / self.resolution); gy = int((y + self.offset_y) / self.resolution)
        gx = max(0, min(gx, self.grid_w - 1)); gy = max(0, min(gy, self.grid_h - 1)); return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = (gx * self.resolution) - self.offset_x + (self.resolution / 2); wy = (gy * self.resolution) - self.offset_y + (self.resolution / 2); return wx, wy
    
    def update_from_semantic(self, drone_pos: np.ndarray, drone_angle: float, semantic_data):
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1]); radius = 8
        y_indices, x_indices = np.ogrid[:self.grid_h, :self.grid_w]
        dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        mask = dist_from_center <= radius; self.grid[mask] -= self.decay_rate
        if semantic_data:
            from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    angle_global = drone_angle + data.angle
                    vx = drone_pos[0] + data.distance * math.cos(angle_global); vy = drone_pos[1] + data.distance * math.sin(angle_global)
                    gx, gy = self.world_to_grid(vx, vy)
                    y_min = max(0, gy-1); y_max = min(self.grid_h, gy+2); x_min = max(0, gx-1); x_max = min(self.grid_w, gx+2)
                    self.grid[y_min:y_max, x_min:x_max] += self.score_add
        self.grid = np.clip(self.grid, 0, self.max_score)

    def get_highest_score_target(self, obstacle_map: Optional['GridMap'] = None) -> Optional[np.ndarray]:
        max_val = np.max(self.grid)
        if max_val < 30.0: return None
        
        # Lấy toạ độ đỉnh nhiệt (vị trí nạn nhân)
        gy, gx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        
        # [NEW LOGIC] SNAP TO NEAREST SAFE CELL
        # Nếu có bản đồ vật cản, hãy kiểm tra xem vị trí này có đứng được không?
        if obstacle_map is not None:
            # Lấy chi phí tại điểm đó
            # Lưu ý: grid của 2 map phải cùng kích thước/resolution (đều là 8)
            if hasattr(obstacle_map, 'cost_map') and obstacle_map.cost_map is not None:
                # Hàm check an toàn
                def is_safe(x, y):
                    if 0 <= x < obstacle_map.grid_w and 0 <= y < obstacle_map.grid_h:
                        # Cost < 200 nghĩa là cách tường khoảng 10-15cm (An toàn để bay đến)
                        return obstacle_map.cost_map[y, x] < 200.0
                    return False

                # Nếu vị trí đỉnh không an toàn (quá sát tường)
                if not is_safe(gx, gy):
                    # Quét xoắn ốc bán kính 6 ô (~50cm) để tìm điểm đứng gần nhất
                    found_safe = False
                    for r in range(1, 8):
                        # Tạo các điểm xung quanh hình vuông bán kính r
                        candidates = []
                        for d in range(-r, r + 1):
                            candidates.append((gx + d, gy - r)) # Top
                            candidates.append((gx + d, gy + r)) # Bottom
                            candidates.append((gx - r, gy + d)) # Left
                            candidates.append((gx + r, gy + d)) # Right
                        
                        # Check xem điểm nào an toàn
                        for cx, cy in candidates:
                            if is_safe(cx, cy):
                                gx, gy = cx, cy
                                found_safe = True
                                break
                        if found_safe: break
                    
                    # Nếu tìm mãi không thấy chỗ đứng (victim kẹt trong tường sâu), 
                    # vẫn trả về vị trí gốc để drone thử vận may hoặc bỏ qua
                    
        wx, wy = self.grid_to_world(gx, gy)
        return np.array([wx, wy])
    
    def clear_area(self, center_pos: np.ndarray, radius_grid: int = 5):
        gx, gy = self.world_to_grid(center_pos[0], center_pos[1])
        y_min = max(0, gy - radius_grid); y_max = min(self.grid_h, gy + radius_grid + 1)
        x_min = max(0, gx - radius_grid); x_max = min(self.grid_w, gx + radius_grid + 1)
        self.grid[y_min:y_max, x_min:x_max] = 0.0