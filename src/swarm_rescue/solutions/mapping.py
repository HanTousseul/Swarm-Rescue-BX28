import math
import heapq
import numpy as np
import cv2
from typing import List, Tuple, Optional

# --- CONFIGURATION ---
RESOLUTION = 10     
MAX_LIDAR_RANGE = 300 

# Xác suất (Probabilities)
VAL_EMPTY = -0.5    
VAL_OBSTACLE = 2.0  
VAL_FREE = -2.0     

# Kẹp giá trị (Saturation)
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

    def update_from_lidar(self, drone_pos: np.ndarray, drone_angle: float, lidar_data: List[float], lidar_angles: List[float]):
        if lidar_data is None or lidar_angles is None: return

        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        update_layer = np.zeros_like(self.grid)

        step = 3 
        
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
                ox, oy = self.world_to_grid(obs_x, obs_y)
                
                if 0 <= ox < self.grid_w and 0 <= oy < self.grid_h:
                    update_layer[oy, ox] = VAL_OBSTACLE

        cv2.circle(update_layer, (cx, cy), 2, VAL_FREE, -1)

        self.grid += update_layer
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

    def find_path_astar(self, start_pos: np.ndarray, end_pos: np.ndarray, safety_margin: int = 2) -> List[np.ndarray]:
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        ex, ey = self.world_to_grid(end_pos[0], end_pos[1])

        WALL_THRESHOLD = 20.0
        if self.grid[ey, ex] > WALL_THRESHOLD: return []
        
        open_list = []
        heapq.heappush(open_list, (0, sx, sy))
        came_from = {}
        g_score = { (sx, sy): 0 }
        
        neighbors = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)]
        
        MARGIN_RANGE = safety_margin 

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
            
            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    if self.grid[ny, nx] > WALL_THRESHOLD: continue

                    is_too_close_to_wall = False
                    if MARGIN_RANGE > 0:
                        for offset in range(1, MARGIN_RANGE + 1):
                            check_points = [(nx + offset, ny), (nx - offset, ny), (nx, ny + offset), (nx, ny - offset)]
                            for cpx, cpy in check_points:
                                if 0 <= cpx < self.grid_w and 0 <= cpy < self.grid_h:
                                    if self.grid[cpy, cpx] > WALL_THRESHOLD:
                                        is_too_close_to_wall = True
                                        break
                            if is_too_close_to_wall: break
                    
                    if is_too_close_to_wall: continue

                    risk_cost = max(0, self.grid[ny, nx]) * 0.5 
                    new_g = g_score[(cx, cy)] + cost + risk_cost
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = math.hypot(ex - nx, ey - ny)
                        heapq.heappush(open_list, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
        return []
    
    def get_frontier_target(self, drone_pos: np.ndarray, drone_angle: float) -> Optional[np.ndarray]:
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        candidates = []
        step = 4 
        
        # 1. Thu thập ứng viên (Lọc sơ bộ)
        for y in range(0, self.grid_h, step):
            for x in range(0, self.grid_w, step):
                # Vùng chưa khám phá (-5 < val < 5)
                if -5.0 < self.grid[y, x] < 5.0:
                    free_neighbors = 0
                    unknown_neighbors = 0
                    
                    # Kiểm tra 4 hướng
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                            val = self.grid[ny, nx]
                            if val < -5.0: free_neighbors += 1
                            elif -5.0 < val < 5.0: unknown_neighbors += 1
                    
                    # [QUAN TRỌNG] Chỉ chọn làm ứng viên nếu:
                    # 1. Tiếp giáp vùng Free (để đi tới được)
                    # 2. Tiếp giáp vùng Unknown khác (để tránh nhiễu hạt lấm tấm)
                    if free_neighbors >= 1 and unknown_neighbors >= 1:
                        candidates.append((x, y))

        if not candidates: return None

        # 2. Chấm điểm (Scoring)
        best_target = None
        min_cost = float('inf')
        
        MIN_DIST_PREFERRED = 80.0 # Tăng nhẹ để drone thích đi xa hơn chút
        
        # Hệ số chấm điểm
        ANGLE_WEIGHT = 40.0       # Phạt góc quay (vừa phải)
        DENSITY_REWARD = 50.0     # [NEW] Điểm thưởng cho mật độ Unknown (Càng lớn càng thích)

        from swarm_rescue.simulation.utils.utils import normalize_angle
        
        for (gx, gy) in candidates:
            wx, wy = self.grid_to_world(gx, gy)
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            
            # --- PHẠT KHOẢNG CÁCH ---
            dist_penalty = 0
            if dist < 20.0: 
                dist_penalty = 10000.0 # Cấm điểm dưới chân
            elif dist < MIN_DIST_PREFERRED:
                dist_penalty = 500.0   # Hạn chế đi điểm quá gần (lắt nhắt)
            
            # [FIX] BỎ PHẠT KHOẢNG CÁCH XA! 
            # Để drone dám bay sang đầu kia bản đồ nếu cần.

            # --- TÍNH GÓC ---
            angle_to_target = math.atan2(wy - drone_pos[1], wx - drone_pos[0])
            angle_diff = abs(normalize_angle(angle_to_target - drone_angle))
            
            # --- [NEW] TÍNH MẬT ĐỘ UNKNOWN (ĐỘ LỚN VÙNG KHÁM PHÁ) ---
            # Quét vùng 5x5 xung quanh điểm candidate xem có bao nhiêu ô Unknown
            # Vùng nào càng nhiều Unknown -> Càng rộng -> Càng nên đi
            density_score = 0
            range_check = 2 # Bán kính 2 ô (tổng 5x5)
            y_min = max(0, gy - range_check)
            y_max = min(self.grid_h, gy + range_check + 1)
            x_min = max(0, gx - range_check)
            x_max = min(self.grid_w, gx + range_check + 1)
            
            # Cắt vùng grid con ra để đếm cho nhanh
            sub_grid = self.grid[y_min:y_max, x_min:x_max]
            # Đếm số ô có giá trị tuyệt đối < 5 (Unknown)
            unknown_count = np.sum(np.abs(sub_grid) < 5.0)
            
            # Điểm thưởng = Số lượng ô Unknown * Hệ số
            density_bonus = unknown_count * DENSITY_REWARD

            # --- TỔNG HỢP COST ---
            # Cost = (Khoảng cách) + (Góc lệch) + (Phạt gần) - (THƯỞNG MỞ RỘNG)
            # Chúng ta muốn Min Cost -> Nên trừ đi điểm thưởng
            cost = dist + (angle_diff * ANGLE_WEIGHT) + dist_penalty - density_bonus
            
            if cost < min_cost:
                min_cost = cost
                best_target = np.array([wx, wy])
                
        return best_target
    
    # [NEW] Hàm tìm điểm Free ngẫu nhiên để tái định vị
    def get_random_free_target(self, drone_pos: np.ndarray, min_dist: float = 200.0) -> Optional[np.ndarray]:
        """
        Tìm một điểm đã biết là TRỐNG (Free) nhưng ở xa vị trí hiện tại.
        Dùng để thoát kẹt khi không tìm thấy Frontier.
        """
        # Lấy tất cả các ô có giá trị < -40 (Rất an toàn)
        free_indices = np.argwhere(self.grid < -40.0)
        
        if len(free_indices) == 0:
            return None
            
        # Xáo trộn ngẫu nhiên để chọn đại một điểm
        np.random.shuffle(free_indices)
        
        # Duyệt qua các điểm ngẫu nhiên, lấy điểm đầu tiên đủ xa
        for (gy, gx) in free_indices:
            wx, wy = self.grid_to_world(gx, gy)
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            
            # Phải xa hơn min_dist mới chọn (để buộc drone di chuyển)
            if dist > min_dist:
                return np.array([wx, wy])
                
        # Nếu map nhỏ quá không có điểm nào xa > 200px, lấy đại điểm xa nhất tìm được
        # Hoặc trả về None để drone xoay vòng
        return None

    def check_line_of_sight(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        x0, y0 = start_pos
        x1, y1 = end_pos
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1.0: return True 
        steps = int(dist / (self.resolution / 2)) 
        if steps == 0: steps = 1
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        for x, y in zip(xs, ys):
            gx, gy = self.world_to_grid(x, y)
            if self.grid[gy, gx] > 10.0: return False
        return True

    def display(self, drone_pos: np.ndarray, current_target: Optional[np.ndarray] = None, current_path: List[np.ndarray] = [], window_name="Obstacle Map"):
        norm_grid = (self.grid - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)
        norm_grid = np.clip(norm_grid, 0.0, 1.0) * 255.0
        norm_grid = norm_grid.astype(np.uint8)
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
            cv2.circle(heatmap_img, (tx, ty), 5, (0, 255, 0), 1) 

        gx, gy = self.world_to_grid(drone_pos[0], drone_pos[1])
        cv2.circle(heatmap_img, (gx, gy), 2, (0, 0, 0), -1)

        target_width = 800
        scale = target_width / heatmap_img.shape[1]
        target_height = int(heatmap_img.shape[0] * scale)
        display_img = cv2.resize(heatmap_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        display_img = cv2.flip(display_img, 0) 

        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)

# Class VictimHeatmap giữ nguyên như cũ
class VictimHeatmap:
    def __init__(self, map_size: Tuple[int, int], resolution: int = 10):
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.resolution = resolution
        
        self.offset_x = self.map_width / 2.0
        self.offset_y = self.map_height / 2.0
        
        self.grid_w = int(self.map_width / self.resolution) + 1
        self.grid_h = int(self.map_height / self.resolution) + 1
        
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.decay_rate = 1.0
        self.score_add = 10.0
        self.max_score = 100.0
        
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

    def update_from_semantic(self, drone_pos: np.ndarray, drone_angle: float, semantic_data):
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        radius = 8
        y_indices, x_indices = np.ogrid[:self.grid_h, :self.grid_w]
        dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        mask = dist_from_center <= radius
        self.grid[mask] -= self.decay_rate
        
        if semantic_data:
            from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    angle_global = drone_angle + data.angle
                    vx = drone_pos[0] + data.distance * math.cos(angle_global)
                    vy = drone_pos[1] + data.distance * math.sin(angle_global)
                    gx, gy = self.world_to_grid(vx, vy)
                    y_min = max(0, gy-1)
                    y_max = min(self.grid_h, gy+2)
                    x_min = max(0, gx-1)
                    x_max = min(self.grid_w, gx+2)
                    self.grid[y_min:y_max, x_min:x_max] += self.score_add
        self.grid = np.clip(self.grid, 0, self.max_score)

    def is_hot_spot(self, world_pos: np.ndarray) -> bool:
        if world_pos is None: return False
        gx, gy = self.world_to_grid(world_pos[0], world_pos[1])
        return self.grid[gy, gx] > 20.0

    def get_highest_score_target(self) -> Optional[np.ndarray]:
        max_val = np.max(self.grid)
        if max_val < 30.0: return None
        gy, gx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        wx, wy = self.grid_to_world(gx, gy)
        return np.array([wx, wy])

    def display(self, drone_pos: np.ndarray, window_name="Victim Heatmap"):
        normalized_grid = (self.grid / self.max_score) * 255.0
        normalized_grid = normalized_grid.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(normalized_grid, cv2.COLORMAP_JET)
        
        gx, gy = self.world_to_grid(drone_pos[0], drone_pos[1])
        cv2.circle(heatmap_img, (gx, gy), 2, (255, 255, 255), -1)
        
        target_width = 800
        scale = target_width / heatmap_img.shape[1]
        target_height = int(heatmap_img.shape[0] * scale)
        display_img = cv2.resize(heatmap_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        display_img = cv2.flip(display_img, 0)
        
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)