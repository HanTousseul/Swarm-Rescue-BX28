import math
import heapq
import numpy as np
import cv2
from typing import List, Tuple, Optional

# --- CONFIGURATION ---
RESOLUTION = 8      # Size of one grid cell in cm
MAX_LIDAR_RANGE = 300 
VAL_EMPTY = -0.5    # Lidar ray passed through (Free space)
VAL_OBSTACLE = 2.0  # Lidar hit an obstacle
VAL_FREE = -2.0     # Confirmed free space (Trajectory)
THRESHOLD_MIN = -50.0
THRESHOLD_MAX = 50.0

class GridMap:
    """
    Occupancy Grid Map representation of the world.
    Handles lidar updates, cost map generation, pathfinding algorithms (A*, Dijkstra),
    and frontier exploration logic.
    """
    def __init__(self, map_size: Tuple[int, int], resolution: int = RESOLUTION):
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.resolution = resolution
        # Initialize grid dimensions
        self.grid_w = int(self.map_width / self.resolution) + 1
        self.grid_h = int(self.map_height / self.resolution) + 1
        self.offset_x = self.map_width / 2.0
        self.offset_y = self.map_height / 2.0
        # The main grid: Positive values = Obstacles, Negative values = Free space
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
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
    
    def update_from_lidar(self, drone_pos: np.ndarray, drone_angle: float, lidar_data: List[float], lidar_angles: List[float], nearby_drones_pos: List[np.ndarray] = [], nearby_victims_pos: List[np.ndarray] = []):
        if lidar_data is None or lidar_angles is None: return
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        update_layer = np.zeros_like(self.grid)
        step = 3 # Optimization: Process every 3rd ray
        
        DRONE_RADIUS_IGNORE = 40.0 
        VICTIM_RADIUS_IGNORE = 30.0 

        for i in range(0, len(lidar_data), step):
            dist = lidar_data[i]
            angle = lidar_angles[i] + drone_angle 
            LIDAR_DIST_CLIP = 40.0
            
            # Mark free space along the ray
            dist_empty = max(0.0, dist - LIDAR_DIST_CLIP)
            empty_x = drone_pos[0] + dist_empty * math.cos(angle)
            empty_y = drone_pos[1] + dist_empty * math.sin(angle)
            ex, ey = self.world_to_grid(empty_x, empty_y)
            cv2.line(update_layer, (cx, cy), (ex, ey), VAL_EMPTY, thickness=1)
            
            # Mark obstacle at the end of the ray
            if dist < (MAX_LIDAR_RANGE - 5.0):
                obs_x = drone_pos[0] + dist * math.cos(angle)
                obs_y = drone_pos[1] + dist * math.sin(angle)
                
                # Check if this obstacle is actually a dynamic entity (Drone/Victim)
                is_ignored = False
                for d_pos in nearby_drones_pos:
                    if math.hypot(obs_x - d_pos[0], obs_y - d_pos[1]) < DRONE_RADIUS_IGNORE:
                        is_ignored = True; break
                if not is_ignored:
                    for v_pos in nearby_victims_pos:
                        if math.hypot(obs_x - v_pos[0], obs_y - v_pos[1]) < VICTIM_RADIUS_IGNORE:
                            is_ignored = True; break
                
                ox, oy = self.world_to_grid(obs_x, obs_y)
                if 0 <= ox < self.grid_w and 0 <= oy < self.grid_h:
                    if not is_ignored: update_layer[oy, ox] = VAL_OBSTACLE 
                    else: update_layer[oy, ox] = VAL_FREE # Treat dynamic entities as free space

        cv2.circle(update_layer, (cx, cy), 2, VAL_FREE, -1) # Clear space under the drone
        self.grid += update_layer
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.update_cost_map()

<<<<<<< HEAD
    def update_from_trajectory(self, trajectory_points: List[Tuple[float, float]]):
        if not trajectory_points or len(trajectory_points) < 2: return
        for i in range(len(trajectory_points) - 1):
            p1 = trajectory_points[i]
            p2 = trajectory_points[i+1]
            gx1, gy1 = self.world_to_grid(p1[0], p1[1])
            gx2, gy2 = self.world_to_grid(p2[0], p2[1])
            cv2.line(self.grid, (gx1, gy1), (gx2, gy2), VAL_FREE, thickness=2)
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

    def update_cost_map(self):
        binary_grid = (self.grid <= 20.0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8) 
        eroded_grid = cv2.erode(binary_grid, kernel, iterations=1)
        self.dist_map = cv2.distanceTransform(eroded_grid, cv2.DIST_L2, 5)
        
=======
    def update_cost_map(self):
        """Generates a Cost Map using Distance Transform. High cost near walls."""
        binary_grid = (self.grid <= 20.0).astype(np.uint8)
        self.dist_map = cv2.distanceTransform(binary_grid, cv2.DIST_L2, 5)
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        SAFETY_WEIGHT = 300.0 
        # Calculate cost: Inversely proportional to distance from obstacles
        self.cost_map = 1.0 + (SAFETY_WEIGHT / (self.dist_map + 0.1))
        
<<<<<<< HEAD
        ROBOT_RADIUS_GRID = 3.0 
        self.cost_map[self.dist_map < ROBOT_RADIUS_GRID] = 9999.0

        unknown_mask = (self.grid > -1.0) & (self.grid <= 20.0)
        UNKNOWN_PENALTY = 100.0 
        self.cost_map[unknown_mask] += UNKNOWN_PENALTY
        self.cost_map[self.grid > 20.0] = 9999.0

=======
        # Penalize Unknown areas slightly to encourage exploring free space
        unknown_mask = (self.grid > -1.0) & (self.grid <= 20.0)
        UNKNOWN_PENALTY = 100.0 
        self.cost_map[unknown_mask] += UNKNOWN_PENALTY
        self.cost_map[self.grid > 20.0] = 9999.0 # Absolute walls

    # ... (Dijkstra and A* implementations omitted for brevity, keeping logic as provided) ...
    def update_dijkstra_map(self, target_pos: np.ndarray):
        self.update_cost_map()
        self.dijkstra_grid = np.full_like(self.grid, np.inf)
        gx_t, gy_t = self.world_to_grid(target_pos[0], target_pos[1])
        if self.cost_map[gy_t, gx_t] > 1000:
             found = False
             for r in range(1, 10):
                 for dy in range(-r, r+1):
                     for dx in range(-r, r+1):
                         ny, nx = gy_t + dy, gx_t + dx
                         if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                             if self.cost_map[ny, nx] < 600: 
                                 gy_t, gx_t = ny, nx
                                 found = True; break
                     if found: break
                 if found: break
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

>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
    def get_dijkstra_path(self, current_pos: np.ndarray, max_steps: int = 40) -> List[np.ndarray]:
        if not hasattr(self, 'dijkstra_grid') or self.dijkstra_grid is None: return []
        cx, cy = self.world_to_grid(current_pos[0], current_pos[1])
        path = [] 
        curr_x, curr_y = cx, cy
        for _ in range(max_steps):
            min_val = self.dijkstra_grid[curr_y, curr_x]
            best_n = None
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        val = self.dijkstra_grid[ny, nx]
                        if val < min_val:
                            min_val = val
                            best_n = (nx, ny)
            if best_n:
                curr_x, curr_y = best_n
                wx, wy = self.grid_to_world(curr_x, curr_y)
                path.append(np.array([wx, wy]))
                if min_val <= 0.0: break
            else: break
        return path

    def find_path_astar(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        ex, ey = self.world_to_grid(end_pos[0], end_pos[1])
        self.update_cost_map()
        if self.cost_map[ey, ex] > 2000: return [] 
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
                    cell_cost = self.cost_map[ny, nx]
                    if cell_cost >= 9999.0: continue 
                    new_g = g_score[(cx, cy)] + (move_cost * cell_cost)
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = math.hypot(ex - nx, ey - ny) 
                        heapq.heappush(open_list, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
        return []
    
<<<<<<< HEAD
    def update_dijkstra_map(self, target_pos: np.ndarray):
        self.update_cost_map()
        self.dijkstra_grid = np.full_like(self.grid, np.inf)
        gx_t, gy_t = self.world_to_grid(target_pos[0], target_pos[1])
        if self.cost_map[gy_t, gx_t] > 1000:
             found = False
             for r in range(1, 10):
                 for dy in range(-r, r+1):
                     for dx in range(-r, r+1):
                         ny, nx = gy_t + dy, gx_t + dx
                         if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                             if self.cost_map[ny, nx] < 600: 
                                 gy_t, gx_t = ny, nx
                                 found = True; break
                     if found: break
                 if found: break
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

    # [HEAVILY UPDATED] Smart Frontier Selection
    def get_reachable_frontier_and_path(self, drone_pos: np.ndarray, drone_angle: float) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Dijkstra Flood Fill with Advanced Scoring:
        1. Distance Filter: Prefer targets > 150cm away (avoid short-sightedness).
        2. Density Score: Prefer areas with MORE unknown cells (Information Gain).
        3. Fallback: If no far targets, accept close ones.
        """
        start_gx, start_gy = self.world_to_grid(drone_pos[0], drone_pos[1])
        
        dist_matrix = np.full((self.grid_h, self.grid_w), float('inf'))
        parent_matrix = {} 
        dist_matrix[start_gy, start_gx] = 0.0
=======
    def get_frontier_target(self, drone_pos: np.ndarray, drone_angle: float, drone_id: int, current_step: int, 
                            busy_targets: List[dict] = [], nearby_drones: List[np.ndarray] = [],
                            rescue_center_pos: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Scans the map to find the best exploration target (Frontier).
        A frontier is a boundary between free space and unknown space.
        Includes logic for: Directional Bias, Lava (Center avoidance), and Hard Filtering (Rescue Center).
        """
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        candidates = []
        step = 3 
        
        # 1. Scan grid for candidate frontiers
        for y in range(0, self.grid_h, step):
            for x in range(0, self.grid_w, step):
                # Check if cell is roughly Unknown/Free boundary
                if -5.0 < self.grid[y, x] < 5.0: 
                    free_neighbors = 0; unknown_neighbors = 0
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                            val = self.grid[ny, nx]
                            if val < -1.0: free_neighbors += 1
                            elif -5.0 < val < 5.0: unknown_neighbors += 1
                    if free_neighbors >= 1 and unknown_neighbors >= 1: candidates.append((x, y))

        if not candidates: return None
        best_target = None; min_cost = float('inf')
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        
        pq = [(0.0, start_gx, start_gy)]
        frontier_candidates = [] # List of (travel_cost, gx, gy)
        
        moves = [(0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), 
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

<<<<<<< HEAD
        # 1. FLOOD FILL
        while pq:
            curr_dist, cx, cy = heapq.heappop(pq)
            if curr_dist > dist_matrix[cy, cx]: continue
            
            # CHECK FRONTIER
            is_frontier = False
            for dx, dy, _ in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    # Neighbor is Unknown?
                    if -5.0 < self.grid[ny, nx] < 5.0: 
                        is_frontier = True
                        break
            
            if is_frontier:
                frontier_candidates.append((curr_dist, cx, cy))
=======
        # Strategy Constants
        MIN_DIST_PREFERRED = 80.0
        ANGLE_WEIGHT = 20.0        
        DENSITY_REWARD = 50.0
        WALL_PENALTY_WEIGHT = 2000.0; WALL_CHECK_RADIUS = 4
        CROWD_RADIUS = 300.0; CROWD_PENALTY = 3000.0 
        
        # Center Lava Logic (Force expansion to edges in early game)
        CENTER_X = self.map_width / 2.0; CENTER_Y = self.map_height / 2.0
        EARLY_GAME_STEPS = 800; LAVA_RADIUS = 350.0; LAVA_PENALTY = 10000.0
        
        # Directional Bias Logic (Split swarm left/right)
        BIAS_WEIGHT = 3.0         
        RC_BAN_RADIUS = 250.0 
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af

            # EXPAND
            for dx, dy, cost_mult in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    if self.grid[ny, nx] > -1.0: continue # Only expand on Free Space

                    cell_risk = 1.0
                    if self.cost_map is not None: cell_risk = self.cost_map[ny, nx]
                    if cell_risk >= 9999.0: continue 
                    
                    new_cost = curr_dist + (cost_mult * cell_risk)
                    if new_cost < dist_matrix[ny, nx]:
                        dist_matrix[ny, nx] = new_cost
                        parent_matrix[(nx, ny)] = (cx, cy)
                        heapq.heappush(pq, (new_cost, nx, ny))

        if not frontier_candidates: return None, []

        # 2. ADVANCED FILTERING & SCORING
        best_candidate = None
        min_score = float('inf')
        
        # Heuristic Params
        MIN_DIST_CM = 150.0 # Don't look at feet
        DENSITY_RADIUS = 3  # Check 7x7 area
        DENSITY_REWARD = 40.0 # Reward for each unknown cell
        
        filtered_candidates = []

        # First Pass: Filter for "Far Enough" candidates
        for cost, gx, gy in frontier_candidates:
            wx, wy = self.grid_to_world(gx, gy)
            dist_air = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            
<<<<<<< HEAD
            if dist_air > MIN_DIST_CM:
                filtered_candidates.append((cost, gx, gy))
        
        # Fallback: If no far candidates, use all valid candidates
        if not filtered_candidates:
            filtered_candidates = frontier_candidates

        # Second Pass: Score Calculation (Travel Cost vs Information Gain)
        for cost, gx, gy in filtered_candidates:
            # Calculate Unknown Density
            unknown_count = 0
            for dy in range(-DENSITY_RADIUS, DENSITY_RADIUS+1):
                for dx in range(-DENSITY_RADIUS, DENSITY_RADIUS+1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        if -5.0 < self.grid[ny, nx] < 5.0:
                            unknown_count += 1
=======
            # [FILTER] HARD FILTER: Do not select targets inside the Rescue Center
            if rescue_center_pos is not None:
                dist_to_rc = math.hypot(wx - rescue_center_pos[0], wy - rescue_center_pos[1])
                if dist_to_rc < RC_BAN_RADIUS: continue

            # Check busy targets (from other drones)
            is_busy = False
            for item in busy_targets:
                if math.hypot(wx - item['pos'][0], wy - item['pos'][1]) < 60.0: is_busy = True; break
            if is_busy: continue

            # Crowd Cost (Avoid clustering)
            crowd_cost = 0
            for d_pos in nearby_drones:
                d_dist = math.hypot(wx - d_pos[0], wy - d_pos[1])
                if d_dist < CROWD_RADIUS:
                    crowd_cost += CROWD_PENALTY * (1.0 - d_dist/CROWD_RADIUS)

            # Directional Bias (Even ID -> Left, Odd ID -> Right)
            bias_cost = 0.0
            if drone_id % 2 == 0: bias_cost = wx * BIAS_WEIGHT
            else: bias_cost = (self.map_width - wx) * BIAS_WEIGHT

            # Lava Cost (Penalize center in early game)
            lava_cost = 0.0
            dist_to_center = math.hypot(wx - CENTER_X, wy - CENTER_Y)
            if current_step < EARLY_GAME_STEPS:
                if dist_to_center < LAVA_RADIUS:
                    lava_cost = LAVA_PENALTY * (1.0 - dist_to_center/LAVA_RADIUS)

            # Standard Costs
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            dist_penalty = 0
            if dist < 20.0: dist_penalty = 10000.0 # Too close
            elif dist < MIN_DIST_PREFERRED: dist_penalty = 500.0   
            
            angle_to_target = math.atan2(wy - drone_pos[1], wx - drone_pos[0])
            angle_diff = abs(normalize_angle(angle_to_target - drone_angle))
            
            # Wall proximity penalty
            wall_penalty = 0
            y_w_min = max(0, gy - WALL_CHECK_RADIUS); y_w_max = min(self.grid_h, gy + WALL_CHECK_RADIUS + 1)
            x_w_min = max(0, gx - WALL_CHECK_RADIUS); x_w_max = min(self.grid_w, gx + WALL_CHECK_RADIUS + 1)
            if np.max(self.grid[y_w_min:y_w_max, x_w_min:x_w_max]) > 20.0: wall_penalty = WALL_PENALTY_WEIGHT
            
            # Information Gain Reward (Density of unknown cells)
            range_check = 2 
            y_min = max(0, gy - range_check); y_max = min(self.grid_h, gy + range_check + 1)
            x_min = max(0, gx - range_check); x_max = min(self.grid_w, gx + range_check + 1)
            unknown_count = np.sum(np.abs(self.grid[y_min:y_max, x_min:x_max]) < 5.0)
            density_bonus = unknown_count * DENSITY_REWARD

            # Total Cost Calculation
            cost = dist + (angle_diff * ANGLE_WEIGHT) + dist_penalty - density_bonus + wall_penalty + crowd_cost + bias_cost + lava_cost
            if cost < min_cost: min_cost = cost; best_target = np.array([wx, wy])
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
            
            # Score Formula: Lower is better
            # Score = Travel Cost - (Unknown Density * Reward)
            # If density is high, score drops (maybe even negative), making it very attractive.
            score = cost - (unknown_count * DENSITY_REWARD)
            
            if score < min_score:
                min_score = score
                best_candidate = (gx, gy)

        if best_candidate is None: return None, [] # Should not happen

        best_gx, best_gy = best_candidate
        
        # 3. BACKTRACK PATH
        path = []
        curr = (best_gx, best_gy)
        while curr != (start_gx, start_gy):
            wx, wy = self.grid_to_world(curr[0], curr[1])
            path.append(np.array([wx, wy]))
            curr = parent_matrix.get(curr)
            if curr is None: break 
        path.reverse()
        best_target_world = np.array(self.grid_to_world(best_gx, best_gy))
        
        return best_target_world, path

    # ... (Keep other fallback functions: get_unknown_target, get_random_free_target, check_line_of_sight, get_cost_at, display, mask_rescue_center)
    # Ensure they are present as before.
    def get_unknown_target(self, drone_pos: np.ndarray, nearby_drones: List[np.ndarray] = []) -> Optional[np.ndarray]:
<<<<<<< HEAD
=======
        """Fallback: Sample random unknown points if no clear frontier is found."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        unknown_indices = np.argwhere(np.abs(self.grid) < 5.0)
        if len(unknown_indices) == 0: return None
        num_samples = min(len(unknown_indices), 50)
        indices = np.random.choice(len(unknown_indices), num_samples, replace=False)
        best_target = None; max_score = -float('inf')
        for idx in indices:
            gy, gx = unknown_indices[idx]
            wx, wy = self.grid_to_world(gx, gy)
            min_dist_to_friend = float('inf')
            for d_pos in nearby_drones:
                d = math.hypot(wx - d_pos[0], wy - d_pos[1])
                if d < min_dist_to_friend: min_dist_to_friend = d
            score = min_dist_to_friend 
            if score > max_score: max_score = score; best_target = np.array([wx, wy])
        return best_target

    def get_random_free_target(self, drone_pos: np.ndarray, min_dist: float = 200.0) -> Optional[np.ndarray]:
<<<<<<< HEAD
=======
        """Ultimate Fallback: Go to a random known free space."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        free_indices = np.argwhere(self.grid < -40.0)
        if len(free_indices) == 0: return None
        np.random.shuffle(free_indices)
        for (gy, gx) in free_indices:
            wx, wy = self.grid_to_world(gx, gy)
            dist = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            if dist > min_dist: return np.array([wx, wy])
        return None

    def check_line_of_sight(self, start_pos: np.ndarray, end_pos: np.ndarray, safety_radius: int = 1, check_cost: bool = True) -> bool:
<<<<<<< HEAD
=======
        """Checks if a direct line between two points is clear of obstacles."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        x0, y0 = start_pos; x1, y1 = end_pos
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1.0: return True 
        steps = int(dist / (self.resolution / 2)); 
        if steps == 0: steps = 1
        xs = np.linspace(x0, x1, steps); ys = np.linspace(y0, y1, steps)
        SAFE_COST_THRESHOLD = 300.0 
        for x, y in zip(xs, ys):
            gx, gy = self.world_to_grid(x, y)
            for dy in range(-safety_radius, safety_radius + 1):
                for dx in range(-safety_radius, safety_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        if self.grid[ny, nx] > 10.0: return False
                        if check_cost and hasattr(self, 'cost_map') and self.cost_map is not None:
                            if self.cost_map[ny, nx] > SAFE_COST_THRESHOLD: return False
        return True

    def get_cost_at(self, world_pos: np.ndarray) -> float:
<<<<<<< HEAD
=======
        """Returns the cost map value at a given world position."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        if not hasattr(self, 'cost_map') or self.cost_map is None: return 1.0 
        gx, gy = self.world_to_grid(world_pos[0], world_pos[1])
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h: return self.cost_map[gy, gx]
        return 9999.0

    def display(self, drone_pos: np.ndarray, current_target: Optional[np.ndarray] = None, current_path: List[np.ndarray] = [], window_name="Obstacle Map"):
<<<<<<< HEAD
=======
        """Debug function to visualize the map using OpenCV."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
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
    
    def mask_rescue_center(self, center_pos: np.ndarray):
<<<<<<< HEAD
=======
        """Hard-masks the Rescue Center area as FREE to prevent generating frontiers inside walls."""
>>>>>>> 15e7390dc466a994bd710da6d4f0b5aeda95b4af
        if center_pos is None: return
        RC_WIDTH = 150.0; RC_HEIGHT = 260.0
        cx, cy = self.world_to_grid(center_pos[0], center_pos[1])
        w_grid = int((RC_WIDTH / 2) / self.resolution) + 2
        h_grid = int((RC_HEIGHT / 2) / self.resolution) + 2
        y_min = max(0, cy - h_grid); y_max = min(self.grid_h, cy + h_grid)
        x_min = max(0, cx - w_grid); x_max = min(self.grid_w, cx + w_grid)
        self.grid[y_min:y_max, x_min:x_max] = VAL_FREE