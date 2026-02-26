import math
import numpy as np
from typing import List, Tuple, Optional
from swarm_rescue.simulation.utils.utils import normalize_angle

try:
    from .mapping import GridMap
except ImportError:
    from mapping import GridMap

class Navigator:
    """
    Handles path planning and target processing.
    Key Responsibilities:
    1. Update internal map state via Sensor data.
    2. Plan paths using A* (Exploration) or Dijkstra (Return/Rescue).
    3. Calculate the "Carrot" waypoint for the Pilot to chase.
    """
    def __init__(self, drone):
        self.drone = drone
        self.gps_last_known = None
        
        map_size = getattr(self.drone, 'map_size', (100, 100))
        self.obstacle_map = GridMap(map_size=map_size)
        
        self.current_path = []
        self.last_astar_target = None 
        
        # Tracks progress along the path to prevent backtracking
        self.last_path_index = 0 
        
        self.replan_timer = 0
        self.failure_cooldown = 0 
        self.cached_nearby_drones = []
        self.dijkstra_update_timer = 0
        self.dijkstra_target_cached = None

    def update_navigator(self, nearby_drones: List[np.ndarray] = [], nearby_victims: List[np.ndarray] = []):
        """
        Updates the drone's internal state regarding its global position, orientation, 
        and the surrounding occupancy grid map.

        This function acts as the core localization and mapping (SLAM-lite) pipeline. 
        It prioritizes absolute sensor data (GPS/Compass) and falls back to kinematic 
        dead reckoning (Odometry) if absolute sensors fail.

        Args:
            nearby_drones (List[np.ndarray]): A list of world coordinates of friendly drones 
                                              within Lidar range. Used to filter out dynamic 
                                              obstacles from the static map.
            nearby_victims (List[np.ndarray]): A list of world coordinates of detected victims. 
                                               Also used for dynamic obstacle filtering.
        """
        # --- 1. STATE ESTIMATION (LOCALIZATION) ---
        # Attempt to get absolute positioning data
        gps_pos = self.drone.measured_gps_position()
        compass_angle = self.drone.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            # Absolute sensors available: Overwrite current estimates
            self.drone.estimated_pos = gps_pos
            self.drone.estimated_angle = compass_angle
        else:
            # Fallback to Dead Reckoning using Odometry (relative movement)
            odom = self.drone.odometer_values()
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                
                # Calculate the global angle of the movement vector
                move_angle = self.drone.estimated_angle + alpha
                
                # Integrate movement into the current position estimate using trigonometry
                self.drone.estimated_pos[0] += dist * math.cos(move_angle)
                self.drone.estimated_pos[1] += dist * math.sin(move_angle)
                
                # Integrate angular velocity into the heading estimate
                self.drone.estimated_angle = normalize_angle(self.drone.estimated_angle + theta)
        
        # Record the spawn point on the first successful initialization
        if self.drone.initial_position is None: 
            self.drone.initial_position = self.drone.estimated_pos.copy()
        
        # --- 2. ENVIRONMENT PERCEPTION (MAPPING) ---
        # Retrieve raw sensory input from the Lidar
        lidar_data = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        
        if lidar_data is not None:
            # Project lidar rays into the global frame and update the occupancy grid
            self.obstacle_map.update_from_lidar(
                drone_pos=self.drone.estimated_pos, 
                drone_angle=self.drone.estimated_angle, 
                lidar_data=lidar_data, 
                lidar_angles=lidar_angles,
                nearby_drones_pos=nearby_drones,
                nearby_victims_pos=nearby_victims
            )

    def find_nearest_walkable(self, target_pos: np.ndarray, search_radius_grid: int = 10) -> Optional[np.ndarray]:
        """
        Performs a Breadth-First Search (BFS) to find the nearest valid, walkable location 
        around a given target position. 
        
        This acts as a "rescue" mechanism when a target waypoint accidentally falls inside 
        an obstacle or a high-risk zone (e.g., due to mapping noise or imprecise coordinate generation).
        
        Args:
            target_pos (np.ndarray): The original target world coordinates [x, y].
            search_radius_grid (int): The maximum distance in grid cells to expand the search.
                                      Default is 10 cells.
                                      
        Returns:
            Optional[np.ndarray]: The world coordinates [x, y] of the nearest safe point, 
                                  or None if no safe point is found within the search radius.
        """
        gx_t, gy_t = self.obstacle_map.world_to_grid(target_pos[0], target_pos[1])
        
        # Initialize BFS queue and visited set
        queue = [(gx_t, gy_t)]
        visited = set([(gx_t, gy_t)])
        
        # Threshold for a "safe" cell (well below the lethal threshold of 9999.0)
        SAFE_COST = 200.0 
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Check if current cell is within grid boundaries
            if 0 <= cx < self.obstacle_map.grid_w and 0 <= cy < self.obstacle_map.grid_h:
                
                # Check if the cell is safe to navigate
                if hasattr(self.obstacle_map, 'cost_map') and self.obstacle_map.cost_map[cy, cx] < SAFE_COST:
                    wx, wy = self.obstacle_map.grid_to_world(cx, cy)
                    return np.array([wx, wy])
                    
            # Stop expanding if we exceed the maximum search radius
            if abs(cx - gx_t) > search_radius_grid or abs(cy - gy_t) > search_radius_grid: 
                continue
                
            # Expand to 4 neighboring cells (Up, Down, Left, Right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
        # No safe point found within the radius
        return None

    def sanitize_target(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Validates and, if necessary, relocates a target waypoint to ensure it is safe 
        for the drone to approach.
        
        It checks the current cost of the target position. If the target is inside a wall 
        or a highly inflated risk zone, it delegates to `find_nearest_walkable` to find 
        a nearby safe alternative.
        
        Args:
            target_pos (np.ndarray): The intended target world coordinates [x, y].
            
        Returns:
            Optional[np.ndarray]: The original target if it is safe, a newly found safe target 
                                  if the original was blocked, or None if relocation fails.
        """
        # Get the risk cost at the exact target coordinate
        current_cost = self.obstacle_map.get_cost_at(target_pos)
        
        # If the cost is acceptable (< 400.0), the target is safe as-is
        if current_cost < 400.0: 
            return target_pos
            
        # Target is too dangerous. Search for a safe fallback point within a 20-grid radius.
        safe_pos = self.find_nearest_walkable(target_pos, search_radius_grid=20)
        
        return safe_pos

    def is_stuck(self) -> bool:
        '''
        function that returns whether or not the drone is stuck, to help unstuck it by re choosing next target
        
        :param self: self
        :return: Whether or not the drone is stuck
        :rtype: bool        
        '''

        self.drone.pos_history_long.append(self.drone.estimated_pos.copy())
        waiting = self.drone.STUCK_TIME_EXPLORING if self.drone.state == 'EXPLORING' else self.drone.STUCK_TIME_OTHER
        if len(self.drone.pos_history_long) > waiting: self.drone.pos_history_long.pop(0) 
        if self.drone.state not in ["END_GAME", "DISPERSING"] and len(self.drone.pos_history_long) == waiting and self.drone.steps_remaining > self.drone.RETURN_TRIGGER_STEPS:
            start_pos = self.drone.pos_history_long[0]
            dist_moved = np.linalg.norm(self.drone.estimated_pos - start_pos)
            if dist_moved < 8.0:
                # print(f"[{self.drone.identifier}] ⚠️ STUCK. Initiating Smart Unstuck...")
                return True
        return False


    def get_adaptive_lookahead(self) -> float:
        """
        Dynamically adjusts the lookahead distance for the Pure Pursuit path tracking 
        based on the environmental risk (clutter) at the drone's current position.
        
        - Open Space (Low Cost): Yields a longer lookahead distance, promoting faster, 
          smoother flight trajectories.
        - Cluttered Space (High Cost): Yields a shorter lookahead distance, prioritizing 
          tight cornering and precise path adherence over speed.
          
        Returns:
            float: The calculated lookahead distance in centimeters.
        """
        current_cost = self.obstacle_map.get_cost_at(self.drone.estimated_pos)
        
        if current_cost < 10.0: 
            return 110.0
        elif current_cost > 100.0: 
            return 40.0
        else: 
            # Linear interpolation for smooth transitions between open and cluttered spaces
            return 110.0 - (current_cost / 100.0) * 70.0 

    def get_carrot_waypoint(self, path: List[np.ndarray], lookahead_dist: float = 80.0) -> np.ndarray:
        """
        Calculates the optimal target waypoint (the 'Carrot') for the local flight controller 
        to track, based on a modified Pure Pursuit algorithm.
        
        This implementation is robust against physical obstacles. It prevents the drone from 
        erroneously skipping path segments separated by thin walls (Wall-Hacking) and dynamically 
        pulls the target closer to prevent hitting inner corners during sharp turns (Corner-Cutting).

        Args:
            path (List[np.ndarray]): The sequence of [x, y] coordinates forming the global path.
            lookahead_dist (float): The base tracking radius. Points further than this distance 
                                    are preferred to maintain smooth forward momentum.

        Returns:
            np.ndarray: The [x, y] coordinates of the selected carrot waypoint.
        """
        if not path: 
            return self.drone.estimated_pos
        
        # Optimization constraint: Limit the search window to maintain O(1)-like performance 
        # and prevent the algorithm from scanning the entire path matrix on every tick.
        search_range = 20
        start_idx = self.last_path_index
        end_idx = min(len(path), start_idx + search_range)
        
        closest_dist = float('inf')
        current_closest_idx = start_idx
        has_valid_los = False # Check if we can see any point in the current waypath

        # --- 1. STRICT PROGRESS TRACKING (LOS VALIDATION) ---
        # Locate the closest geometric point on the path that is physically visible.
        # This prevents the path index from jumping through solid walls in U-shaped corridors.
        for i in range(start_idx, end_idx):
            wp = path[i]
            dist = np.linalg.norm(self.drone.estimated_pos - wp)
            if dist < closest_dist:
                # Validate Line of Sight (LOS) ignoring the soft cost inflation (check_cost=False)
                # to ensure we only test against hard physical boundaries.
                if self.obstacle_map.check_line_of_sight(self.drone.estimated_pos, wp, safety_radius=1, check_cost=False):
                    closest_dist = dist
                    current_closest_idx = i
                    has_valid_los = True # Found at least one waypoint can go to
        
        # If can't go to waypoint, plan again!
        if not has_valid_los and self.drone.state == 'EXPLORING':
            return None
        
        # Commit the validated progress index
        self.last_path_index = current_closest_idx

        # --- 2. IDEAL CARROT IDENTIFICATION ---
        # Find the first point along the path that falls strictly outside the lookahead radius.
        ideal_idx = len(path) - 1
        for i in range(current_closest_idx, len(path)):
            d_to_carrot = np.linalg.norm(self.drone.estimated_pos - path[i])
            if d_to_carrot > lookahead_dist:
                ideal_idx = i
                break

        # --- 3. SLIDING CARROT (CORNER-CUTTING PREVENTION) ---
        # Retract the carrot along the path toward the drone until a clear, wide line of sight is established.
        # This prevents the trajectory from clipping the inner apex of a sharp corner.
        carrot_idx = ideal_idx
        while carrot_idx > current_closest_idx:
            carrot_wp = path[carrot_idx]
            
            # Use a thicker safety radius (e.g., 2 cells) to account for the drone's physical width.
            if self.obstacle_map.check_line_of_sight(self.drone.estimated_pos, carrot_wp, safety_radius=1, check_cost=False):
                break  # A clear, safe trajectory to the carrot is confirmed.
                
            carrot_idx -= 1 # Line of sight is blocked; pull the target closer.

        # Fallback: If heavy drift causes even the closest valid point to be obscured, 
        # force the drone to target the immediate next waypoint to regain path alignment.
        if carrot_idx == current_closest_idx and carrot_idx + 1 < len(path):
            carrot_idx += 1

        return path[carrot_idx]

    def get_next_waypoint(self, final_target: np.ndarray, force_replan: bool = False) -> Optional[np.ndarray]:
        """
        The main control loop for high-level navigation. It validates current paths, 
        selects the appropriate pathfinding algorithm based on the drone's current mission state, 
        and extracts a short-term 'carrot' waypoint for the local flight controller (Pilot) to follow.

        Args:
            final_target (np.ndarray): The intended world coordinates [x, y] to reach.
            force_replan (bool): If True, forces the pathfinder to recalculate the route immediately, 
                                 bypassing normal timer constraints. Defaults to False.

        Returns:
            Optional[np.ndarray]: The world coordinates [x, y] of the immediate next waypoint (the carrot). 
                                  Returns None if the path is blocked, invalid, or during a failure cooldown.
        """
        if final_target is None: 
            return None
        
        # --- 1. REAL-TIME PATH VALIDATION ---
        # Look ahead on the existing path to ensure it hasn't been blocked by newly discovered obstacles.
        if self.current_path:
            start_check_idx = self.last_path_index
            check_range = 35 # Lookahead distance for dynamic obstacle detection
            end_check_idx = min(len(self.current_path), start_check_idx + check_range)
            
            path_blocked = False
            
            for i in range(start_check_idx, end_check_idx):
                wp = self.current_path[i]
                cost = self.obstacle_map.get_cost_at(wp)
                if i - self.last_path_index < 5: 
                    continue
                
                # Thresholds: > 9000 indicates a lethal collision course based on updated Lidar data.
                cost_threshold = 9000.0 if self.drone.state == 'EXPLORING' else 8000.0
                if cost > cost_threshold: 
                    # print(f"[{self.drone.identifier}] ⚠️ Path Blocked at step {i}. Cost: {cost:.1f}")
                    path_blocked = True
                    break
            
            # Abort the current trajectory if a collision is imminent
            if path_blocked:
                self.current_path = [] 
                self.last_astar_target = None
                return None 

        # --- 2. TARGET SANITIZATION ---
        # Ensure the final target isn't inside a newly discovered wall
        safe_target = self.sanitize_target(final_target)
        if safe_target is None: 
            return None
        final_target = safe_target
        
        # --- 3. COOLDOWN MANAGEMENT ---
        # Pause calculations briefly after a pathfinding failure to save CPU 
        # and allow the map to update.
        if self.failure_cooldown > 0:
            self.failure_cooldown -= 1
            if self.current_path: 
                return self.get_carrot_waypoint(self.current_path)
            return None

        # --- 4. STATE-MACHINE PATHFINDING ---
        if self.drone.state in ['RETURNING', 'DROPPING', 'END_GAME']:
            # Use Global Dijkstra Field for homing operations
            self.dijkstra_update_timer += 1
            should_update_map = False
            
            # Recompute the field if the target moves significantly or the timer expires
            if self.dijkstra_target_cached is None or np.linalg.norm(final_target - self.dijkstra_target_cached) > 20.0: 
                should_update_map = True
            elif self.dijkstra_update_timer > 40: 
                should_update_map = True
                
            if should_update_map:
                self.obstacle_map.update_dijkstra_map(final_target)
                self.dijkstra_target_cached = final_target.copy()
                self.dijkstra_update_timer = 0
                
            # Extract the downhill path from the gradient field
            raw_path = self.obstacle_map.get_dijkstra_path(self.drone.estimated_pos, max_steps=40)
            if len(raw_path) > 0: 
                self.current_path = raw_path
                self.last_path_index = 0
            else: 
                self.current_path = []

        elif self.drone.state == 'RESCUING':
            # Use A* Search for precision point-to-point target acquisition
            self.replan_timer += 1
            need_replan = force_replan or (self.last_astar_target is None) or \
                          (np.linalg.norm(final_target - self.last_astar_target) > 35.0) or \
                          (len(self.current_path) == 0) or (self.replan_timer > 40)
                          
            if need_replan:
                self.replan_timer = 0
                self.current_path = self.obstacle_map.find_path_astar(self.drone.estimated_pos, final_target)
                        
                self.last_astar_target = final_target.copy()
                self.last_path_index = 0
                
                if not self.current_path: 
                    return None

        elif self.drone.state == 'EXPLORING':
            # Rely entirely on the external Frontier-based Floodfill path. 
            # Do not attempt A* here.
            if len(self.current_path) == 0:
                self.failure_cooldown = 20
                return None

        if not self.current_path: 
            return final_target
            
        # --- 5. CARROT EXTRACTION (PURE PURSUIT) ---
        # Determine how far ahead on the path the drone should aim
        if self.drone.state == 'RESCUING': 
            LOOKAHEAD = 10.0 # High precision, slow approach
        else: 
            LOOKAHEAD = self.get_adaptive_lookahead() # Dynamic based on clutter
        
        # Extract and return the carrot waypoint
        carrot_wp = self.get_carrot_waypoint(self.current_path, lookahead_dist=LOOKAHEAD)

        # [NEW] CATCH THE DETACHED PATH
        if carrot_wp is None:
            # print(f"[{self.drone.identifier}] ✂️ PATH DETACHED (Behind Wall). Clearing path.")
            self.current_path = []
            self.last_astar_target = None
            return None # Activate Fast Unstuck in Driver!
            
        return carrot_wp