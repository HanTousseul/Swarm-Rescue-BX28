import math
import heapq
import numpy as np
import cv2
from typing import List, Tuple, Optional

# --- CONFIGURATION ---
RESOLUTION = 8      # Size of one grid cell in pixels
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
        self.panic_mode = False 
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts world coordinates (cm) to grid indices (x, y)."""
        gx = int((x + self.offset_x) / self.resolution)
        gy = int((y + self.offset_y) / self.resolution)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Converts grid indices (x, y) to world coordinates (px)."""
        wx = (gx * self.resolution) - self.offset_x + (self.resolution / 2)
        wy = (gy * self.resolution) - self.offset_y + (self.resolution / 2)
        return wx, wy
    
    def update_from_lidar(self, drone_pos: np.ndarray, drone_angle: float, lidar_data: List[float], lidar_angles: List[float], nearby_drones_pos: List[np.ndarray] = [], nearby_victims_pos: List[np.ndarray] = []):
        """
        Updates the grid map using Lidar raycasting.
        Filters out obstacles caused by other drones or victims to keep the map static.
        """
        if lidar_data is None or lidar_angles is None: return
        cx, cy = self.world_to_grid(drone_pos[0], drone_pos[1])
        update_layer = np.zeros_like(self.grid)
        step = 3 # Optimization: Process every 3rd ray
        
        DRONE_RADIUS_IGNORE = 15
        VICTIM_RADIUS_IGNORE = 15

        for i in range(0, len(lidar_data), step):
            dist = lidar_data[i]
            angle = lidar_angles[i] + drone_angle 
            LIDAR_DIST_CLIP = 20.0
            MAX_CLEAR_DIST = 200.0 
            clear_dist = min(dist, MAX_CLEAR_DIST) # Limit clear dist of lidar to similar as semantic
            
            # Mark free space along the ray
            dist_empty = max(0.0, clear_dist - LIDAR_DIST_CLIP)
            empty_x = drone_pos[0] + dist_empty * math.cos(angle)
            empty_y = drone_pos[1] + dist_empty * math.sin(angle)
            ex, ey = self.world_to_grid(empty_x, empty_y)
            cv2.line(update_layer, (cx, cy), (ex, ey), VAL_EMPTY, thickness=1)
            
            # Mark obstacle at the end of the ray
            if dist < (MAX_LIDAR_RANGE - 5.0): # - 5.0 because of noise
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

        # cv2.circle(update_layer, (cx, cy), 2, VAL_FREE, -1) # Clear space under the current drone so that it won't think itself is the obstacle
        self.grid += update_layer
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.update_cost_map()

    def update_cost_map(self):
        """
        Generates a Cost Map using Distance Transform. 
        Higher values represent higher risk near obstacles.
        """
        
        # --- 1. CREATE VIRTUAL CONCRETE MASK ---
        # Create a binary mask where 1 represents a solid wall (grid value > 10.0)
        # We don't modify the main grid directly to keep historical lidar data intact.
        obstacle_mask = (self.grid > 1.0).astype(np.uint8)
        
        # Morphological Closing: Seals small gaps (e.g., a 3x3 kernel ~ 24cm) between dots or pillars.
        # This prevents the pathfinder from attempting to squeeze through impossible tiny gaps.
        closing_kernel = np.ones((2, 2), np.uint8)
        closed_obstacles = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # --- 2. SPATIAL RISK CALCULATION ---
        # Invert the mask: 1 represents free space, 0 represents obstacles/walls.
        binary_grid = (1 - closed_obstacles).astype(np.uint8)
        
        # This allows the distance transform to start exactly at the physical wall edge.
        # Calculate the Euclidean distance from each free cell to its nearest wall.
        self.dist_map = cv2.distanceTransform(binary_grid, cv2.DIST_L2, 5)
        
        # Calculate the base cost: Values are inversely proportional to the distance from walls.
        # Formula: Cost = 1.0 + (Weight / (Distance + Offset))
        SAFETY_WEIGHT = 10.0 if self.panic_mode else 60.0
        self.cost_map = 1.0 + (SAFETY_WEIGHT / (self.dist_map + 0.1))
        
        # --- 3. APPLY LETHAL RADIUS AND PENALTIES ---
        # LETHAL ZONE: If the distance is smaller than the drone's physical radius, 
        # set cost to 9999.0 to strictly forbid the pathfinder from entering.
        # Normal: 2.0 grid (~16px), Panic: 1.5 grid (~12px).
        ROBOT_RADIUS_GRID = 1.0 if self.panic_mode else 1.5 
        self.cost_map[self.dist_map < ROBOT_RADIUS_GRID] = 9999.0

        # UNKNOWN SPACE PENALTY: Incentivize the drone to stay on confirmed 'Free' paths.
        # We apply a heavy penalty to any area that hasn't been cleared by Lidar or Trajectory.
        unknown_mask = (self.grid > -1.0) & (self.grid <= 20.0)
        UNKNOWN_PENALTY = 10.0 if self.panic_mode else 100.0
        self.cost_map[unknown_mask] += UNKNOWN_PENALTY
        
        # ABSOLUTE WALL LOCK: Ensure actual walls and gaps sealed by the virtual concrete 
        # are locked to the maximum cost.
        self.cost_map[closed_obstacles == 1] = 9999.0
        self.cost_map[self.grid > 20.0] = 9999.0

    def get_dijkstra_path(self, current_pos: np.ndarray, max_steps: int = 40) -> List[np.ndarray]:
        """
        Extracts a path by performing gradient descent (steepest descent) on the pre-calculated Dijkstra grid.
        
        Args:
            current_pos (np.ndarray): The current world coordinates [x, y] of the drone.
            max_steps (int): The maximum number of lookahead waypoints to generate. Default is 40.
            
        Returns:
            List[np.ndarray]: A list of world coordinate waypoints leading towards the target.
        """
        # Ensure the dijkstra map has been initialized
        if not hasattr(self, 'dijkstra_grid') or self.dijkstra_grid is None: 
            return []
            
        cx, cy = self.world_to_grid(current_pos[0], current_pos[1])
        path = [] 
        curr_x, curr_y = cx, cy
        
        # Iteratively find the neighbor with the lowest cost value (roll down the hill)
        for _ in range(max_steps):
            min_val = self.dijkstra_grid[curr_y, curr_x]
            best_n = None
            
            # Scan all 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: 
                        continue
                        
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        val = self.dijkstra_grid[ny, nx]
                        # Track the neighbor with the strictly smallest value
                        if val < min_val:
                            min_val = val
                            best_n = (nx, ny)
                            
            # If a descending path is found, append it to the path list
            if best_n:
                curr_x, curr_y = best_n
                wx, wy = self.grid_to_world(curr_x, curr_y)
                path.append(np.array([wx, wy]))
                
                # Stop if we have reached the target (value <= 0.0)
                if min_val <= 0.0: 
                    break
            else: 
                # Stuck in a local minimum, cannot descend further
                break
                
        return path
    
    def update_dijkstra_map(self, target_pos: np.ndarray):
        """
        Generates a global navigation flow field radiating outwards from the target position.
        It uses Dijkstra's algorithm to calculate the cumulative travel cost from the target to every cell.
        
        Args:
            target_pos (np.ndarray): The world coordinates [x, y] of the destination (e.g., Rescue Center).
        """
        self.update_cost_map()
        
        # Initialize the grid with infinity
        self.dijkstra_grid = np.full_like(self.grid, np.inf)
        gx_t, gy_t = self.world_to_grid(target_pos[0], target_pos[1])
        
        # --- TARGET RELOCATION LOGIC ---
        # If the requested target is inside an obstacle (Cost > 1000), search outward in concentric 
        # squares (up to radius 9) to find the nearest safe cell (Cost < 600) to act as the new target.
        if self.cost_map[gy_t, gx_t] > 1000:
             found = False
             for r in range(1, 10):
                 for dy in range(-r, r+1):
                     for dx in range(-r, r+1):
                         ny, nx = gy_t + dy, gx_t + dx
                         if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                             if self.cost_map[ny, nx] < 600: 
                                 gy_t, gx_t = ny, nx
                                 found = True
                                 break
                     if found: break
                 if found: break
                 
        # Set target value to 0 and initialize Priority Queue (Cost, X, Y)
        self.dijkstra_grid[gy_t, gx_t] = 0.0
        pq = [(0.0, gx_t, gy_t)] 
        
        # Define movement directions and their base Euclidean distances
        neighbors = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0), 
                     (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
                     
        # --- DIJKSTRA EXPANSION ---
        while pq:
            curr_val, cx, cy = heapq.heappop(pq)
            
            # Skip if we already found a shorter path to this cell
            if curr_val > self.dijkstra_grid[cy, cx]: 
                continue
                
            for dx, dy, dist_w in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    
                    # Prevent squeezing diagonally through physical wall corners
                    if dx != 0 and dy != 0: 
                        if self.grid[cy, nx] > 1.0 or self.grid[ny, cx] > 1.0:
                            continue
                            
                    step_risk = self.cost_map[ny, nx] 
                        
                    # Stop expanding into lethal obstacles
                    if step_risk >= 9999.0: 
                        continue

                    # Increase dangerousness of wall when returning
                    step_risk = step_risk ** 2.0

                    # Massive penalty for UNKNOWN space to force Dijkstra to stay in cleared VAL_FREE areas
                    if -1.0 < self.grid[ny, nx] <= 20.0:
                        penalty = 50.0 if self.panic_mode else 20000.0
                        step_risk += penalty

                    # Calculate new cumulative cost
                    new_val = curr_val + (dist_w * step_risk)
                    
                    # Relaxation step
                    if new_val < self.dijkstra_grid[ny, nx]:
                        self.dijkstra_grid[ny, nx] = new_val
                        heapq.heappush(pq, (new_val, nx, ny))

    def find_path_astar(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        """
        Calculates the optimal, collision-free path from a start position to an end position 
        using the A* (A-Star) search algorithm.
        
        This implementation incorporates the generated 'cost_map' to keep the drone safely away 
        from obstacles, and heavily penalizes unknown areas to encourage traversing explored space.

        Args:
            start_pos (np.ndarray): The starting world coordinates [x, y] of the drone.
            end_pos (np.ndarray): The target world coordinates [x, y] to reach.

        Returns:
            List[np.ndarray]: A list of world coordinates representing the path. Returns an empty list 
                              if no valid path is found or if the target is inside an obstacle.
        """
        sx, sy = self.world_to_grid(start_pos[0], start_pos[1])
        ex, ey = self.world_to_grid(end_pos[0], end_pos[1])
        
        self.update_cost_map()
        
        # Abort immediately if the exact destination is inside a lethal obstacle
        if self.cost_map[ey, ex] > 9000: 
            return [] 
            
        # Priority queue storing tuples of (f_score, grid_x, grid_y)
        open_list = []
        heapq.heappush(open_list, (0, sx, sy))
        
        # Dictionary to reconstruct the path once the goal is reached
        came_from = {}
        
        # Dictionary storing the exact cost from start to a specific node (g_score)
        g_score = { (sx, sy): 0 }
        
        # 8-directional movement: (dx, dy, movement_cost_multiplier)
        neighbors = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), 
                     (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)]
                     
        while open_list:
            # Pop the node with the lowest f_score
            _, cx, cy = heapq.heappop(open_list)
            
            # GOAL REACHED: Reconstruct path from end to start
            if (cx, cy) == (ex, ey):
                path = []
                curr = (ex, ey)
                while curr in came_from:
                    wx, wy = self.grid_to_world(curr[0], curr[1])
                    path.append(np.array([wx, wy]))
                    curr = came_from[curr]
                path.reverse() # Reverse to get the path from start to goal
                return path
                
            # Explore all valid neighbors
            for dx, dy, move_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                
                # Ensure neighbor is within grid bounds
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    
                    # --- [NEW] ANTI-DIAGONAL CUTTING ---
                    # Prevent A* from finding a theoretical path that squeezes diagonally 
                    # through the exact intersection of two adjacent solid walls.
                    if dx != 0 and dy != 0: 
                        if self.grid[cy, nx] > 1.0 or self.grid[ny, cx] > 1.0:
                            continue 
                    # -----------------------------------
                    
                    # Base risk cost from the distance transform cost map
                    cell_cost = self.cost_map[ny, nx]
                    
                    # --- UNKNOWN SPACE PENALTY ---
                    # Heavily penalize cells that haven't been confirmed as 'Free Space' (-2.0) 
                    # or 'Empty' (-0.5). This forces the drone to stick to reliable, known paths.
                    if -1.0 < self.grid[ny, nx] <= 20.0:
                        cell_cost += 5000.0
                        
                    # Skip lethal obstacles entirely
                    if cell_cost >= 9000.0: 
                        continue 
                        
                    # Calculate new path cost to reach this neighbor
                    new_g = g_score[(cx, cy)] + (move_cost * cell_cost)
                    
                    # If this is the best path to this neighbor so far, record it
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        
                        # Heuristic (h): Euclidean distance to the goal
                        h = math.hypot(ex - nx, ey - ny) 
                        
                        # Push to priority queue with f_score = g_score + h
                        heapq.heappush(open_list, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
                        
        # Open list is empty and goal was not reached (No path exists)
        return []

    def get_reachable_frontier_and_path(self, drone_pos: np.ndarray, drone_angle: float, preferred_angle: float = 0.0, initial_pos: Optional[np.ndarray] = None, rescue_center_pos: Optional[np.ndarray] = None, blacklisted_targets: List[np.ndarray] = []) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Explores the map to find the best 'Frontier' (boundary between known and unknown space) 
        using a Dijkstra-based Flood Fill algorithm, followed by a heuristic scoring system.

        Scoring Heuristics (Lower is better):
        1. Base Travel Cost: Distance (in grid cells) to reach the frontier.
        2. Distance Penalty: Heavily penalizes distant frontiers to prioritize local exploration.
        3. Information Gain (Density Reward): Rewards frontiers surrounded by unexplored fog.
        4. Directional Bias: Penalizes deviations from the preferred dispersion angle. 
           This penalty decays over distance to allow free exploration later in the mission.

        Args:
            drone_pos: Current world coordinates [x, y] of the drone.
            drone_angle: Current heading of the drone.
            preferred_angle: The global angle assigned during the dispersion phase.
            initial_pos: The spawn point of the drone (used for decay calculations).
            rescue_center_pos: Position of the rescue center to avoid exploring its immediate vicinity.

        Returns:
            Tuple containing the chosen frontier's world coordinates and the path (List of waypoints) to it.
        """
        start_gx, start_gy = self.world_to_grid(drone_pos[0], drone_pos[1])
        
        dist_matrix = np.full((self.grid_h, self.grid_w), float('inf'))
        parent_matrix = {} 
        dist_matrix[start_gy, start_gx] = 0.0
        
        pq = [(0.0, start_gx, start_gy)]
        frontier_candidates = [] # List of tuples: (travel_cost, gx, gy)
        
        # 8-directional movement costs
        moves = [(0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), 
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        # =========================================================
        # 1. FLOOD FILL (Find all reachable boundaries)
        # =========================================================
        while pq:
            curr_dist, cx, cy = heapq.heappop(pq)
            if curr_dist > dist_matrix[cy, cx]: 
                continue
            
            # --- CHECK FRONTIER ---
            is_frontier = False
            for dx, dy, _ in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    # Strict frontier definition: only true unknown (>=0.0) or soft walls (<5.0)
                    # Excludes cleared trajectory (-2.0) and lidar-cleared space (-0.5)
                    if -0.1 < self.grid[ny, nx] < 5.0: 
                        is_frontier = True
                        break
            
            if is_frontier:
                frontier_candidates.append((curr_dist, cx, cy))

            # --- EXPAND ---
            for dx, dy, cost_mult in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    
                    # Prevent diagonal squeezing through solid wall corners
                    if dx != 0 and dy != 0: 
                        if self.grid[cy, nx] > 1.0 or self.grid[ny, cx] > 1.0:
                            continue
                    
                    # Allow traversal only through Free Space OR if trapped near the start
                    is_free_space = (self.grid[ny, nx] < -0.1)
                    dist_sq_from_start = (nx - start_gx)**2 + (ny - start_gy)**2
                    is_near_start = (dist_sq_from_start < 225) # Radius ~15 grid cells
                    
                    if not is_free_space:
                        if not is_near_start:
                            continue
                        # Never traverse hard walls, even if near start
                        if self.grid[ny, nx] > 10.0:
                            continue

                    # Soften the wall risk penalty for floodfill to allow squeezing through gaps
                    cell_risk = 1.0
                    if self.cost_map is not None:
                        cell_risk = self.cost_map[ny, nx]
                        if cell_risk < 9999.0:
                            cell_risk = 1.0 + (cell_risk / 10.0)
                    
                    # Block traversal into lethal cores
                    if cell_risk >= 9999.0 and not is_near_start: 
                        continue 
                    
                    new_cost = curr_dist + (cost_mult * cell_risk)
                    
                    # Relaxation
                    if new_cost < dist_matrix[ny, nx]:
                        dist_matrix[ny, nx] = new_cost
                        parent_matrix[(nx, ny)] = (cx, cy)
                        heapq.heappush(pq, (new_cost, nx, ny))

        if not frontier_candidates: 
            return None, []

        # =========================================================
        # 2. SINGLE-PASS ADVANCED SCORING (Evaluate Frontiers)
        # =========================================================
        best_candidate = None
        min_score = float('inf')
        
        # Heuristic Parameters
        DENSITY_RADIUS = 3      # Radius (cells) to check for information gain
        DENSITY_REWARD = 40.0   # Score deduction per unknown cell found
        DISTANCE_WEIGHT = 3.0   # Weight multiplier for physical distance penalty
        
        for cost, gx, gy in frontier_candidates:
            wx, wy = self.grid_to_world(gx, gy)

            is_blacklisted = False
            for bad_target in blacklisted_targets:
                if math.hypot(wx - bad_target[0], wy - bad_target[1]) < 20.0:
                    is_blacklisted = True
                    break
            if is_blacklisted:
                continue
            
            # --- A. ABSOLUTE BLACKLIST ---
            # Ignore shadows cast directly around the Rescue Center
            if rescue_center_pos is not None:
                dist_to_base = math.hypot(wx - rescue_center_pos[0], wy - rescue_center_pos[1])
                if dist_to_base < 180.0:
                    continue 
            
            # --- B. INFORMATION GAIN (Density Reward) ---
            unknown_count = 0
            for dy in range(-DENSITY_RADIUS, DENSITY_RADIUS + 1):
                for dx in range(-DENSITY_RADIUS, DENSITY_RADIUS + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        if -0.1 < self.grid[ny, nx] < 5.0:
                            unknown_count += 1
                            
            # --- C. DISTANCE PENALTY (Prioritize Closeness) ---
            # Convert physical distance into a strong penalty. A target 400px away 
            # will get +1200 penalty, forcing the drone to prioritize closer areas 
            # unless the far area has MASSIVE information gain.
            dist_air = math.hypot(wx - drone_pos[0], wy - drone_pos[1])
            distance_penalty = dist_air * DISTANCE_WEIGHT

            # --- D. DIRECTIONAL BIAS (Dispersion logic) ---
            angle_to_frontier = math.atan2(wy - drone_pos[1], wx - drone_pos[0])
            
            # Shortest angular difference [-pi, pi]
            angle_diff = angle_to_frontier - preferred_angle
            while angle_diff > math.pi: angle_diff -= 2 * math.pi
            while angle_diff < -math.pi: angle_diff += 2 * math.pi
            abs_angle_diff = abs(angle_diff)
            
            # Decay factor: Decreases penalty as the drone travels further from spawn
            dist_from_start = 0.0
            if initial_pos is not None:
                dist_from_start = math.hypot(drone_pos[0] - initial_pos[0], drone_pos[1] - initial_pos[1])
            
            decay_factor = max(0.0, 1.0 - (dist_from_start / 600.0))
            BASE_WEIGHT = 800.0 
            DIRECTION_PENALTY_WEIGHT = BASE_WEIGHT * decay_factor
            
            direction_penalty = (abs_angle_diff ** 2) * DIRECTION_PENALTY_WEIGHT
            
            # --- E. FINAL SCORE COMPUTATION ---
            # Score Formula: Lower is better
            # Base Cost + Distance Penalty - Density Reward + Angle Penalty
            score = cost + distance_penalty - (unknown_count * DENSITY_REWARD) + direction_penalty
            
            if score < min_score:
                min_score = score
                best_candidate = (gx, gy)

        if best_candidate is None: 
            return None, [] 

        # =========================================================
        # 3. BACKTRACK PATH
        # =========================================================
        best_gx, best_gy = best_candidate
        path = []
        curr = (best_gx, best_gy)
        
        while curr != (start_gx, start_gy):
            wx, wy = self.grid_to_world(curr[0], curr[1])
            path.append(np.array([wx, wy]))
            curr = parent_matrix.get(curr)
            if curr is None: 
                break 
                
        path.reverse()
        best_target_world = np.array(self.grid_to_world(best_gx, best_gy))
        
        return best_target_world, path

    def check_line_of_sight(self, start_pos: np.ndarray, end_pos: np.ndarray, safety_radius: int = 2, check_cost: bool = True) -> bool:
        """
        Performs a raycasting check to determine if a straight, direct path between two points 
        is clear of physical obstacles and high-risk zones.

        Instead of checking an infinitely thin line, it sweeps a bounding box (defined by safety_radius) 
        along the path to account for the physical size of the drone.

        Args:
            start_pos (np.ndarray): The starting world coordinates [x, y].
            end_pos (np.ndarray): The target world coordinates [x, y].
            safety_radius (int): The radius (in grid cells) around the ray to check for obstacles, 
                                 creating a "safe corridor" (Default is 1 cell).
            check_cost (bool): If True, it also validates the path against the generated cost map 
                               to avoid dangerously hugging walls (Default is True).

        Returns:
            bool: True if the entire line of sight is clear and safe, False if an obstacle 
                  or high-cost cell is detected along the corridor.
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        # Calculate Euclidean distance
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1.0: 
            return True 
            
        # Determine the number of sampling steps to prevent skipping over thin walls.
        # Sampling at half the grid resolution ensures Nyquist-Shannon safety.
        steps = int(dist / (self.resolution / 2))
        if steps == 0: 
            steps = 1
            
        # Generate linearly spaced points along the line
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        
        # The threshold above which a cell is considered too dangerous to fly over
        SAFE_COST_THRESHOLD = 300.0 
        
        # Sweep along the generated points
        for x, y in zip(xs, ys):
            gx, gy = self.world_to_grid(x, y)
            
            # Check the neighborhood (corridor) around the current point
            for dy in range(-safety_radius, safety_radius + 1):
                for dx in range(-safety_radius, safety_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    
                    # Ensure coordinates are within grid boundaries
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        
                        # 1. HARD COLLISION CHECK: Abort if it hits a confirmed solid obstacle
                        if self.grid[ny, nx] > 1.0: 
                            return False
                            
                        # 2. SOFT RISK CHECK: Abort if the path grazes too close to an obstacle's inflation zone
                        if check_cost and hasattr(self, 'cost_map') and self.cost_map is not None:
                            if self.cost_map[ny, nx] > SAFE_COST_THRESHOLD: 
                                return False
                                
        # If the loop finishes without returning False, the path is completely clear
        return True

    def get_cost_at(self, world_pos: np.ndarray) -> float:
        """Returns the cost map value at a given world position."""
        if not hasattr(self, 'cost_map') or self.cost_map is None: return 1.0 
        gx, gy = self.world_to_grid(world_pos[0], world_pos[1])
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h: return self.cost_map[gy, gx]
        return 9999.0

    def display(self, drone_pos: np.ndarray, current_target: Optional[np.ndarray] = None, current_path: List[np.ndarray] = [], window_name="Obstacle Map"):
        """Debug function to visualize the map using OpenCV."""
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