import math
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class Pilot:
    """
    Low-level controller. Converts high-level targets (waypoints) into 
    motor commands (forward, lateral, rotation, grasper).
    Handles Obstacle Avoidance (Repulsion) and kinematics.
    """
    def __init__(self, drone):
        self.drone = drone
        self.last_pos = None
        self.current_speed = 0.0

    def calculate_repulsive_force(self):
        """Calculates repulsive force to avoid colliding with other drones."""
        total_lat = 0.0
        semantic_data = self.drone.semantic_values()
        if not semantic_data: return 0.0, 0.0

        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                dist = data.distance
                if 0.1 < dist < 150.0:
                    K = 500.0 
                    force_magnitude = K / (dist ** 2)
                    force_magnitude = min(1.5, force_magnitude)
                    push_angle = data.angle + math.pi - 0.5 
                    total_lat += force_magnitude * math.sin(push_angle)
        return 0.0, total_lat

    def calculate_wall_repulsion(self, aggressive: bool = False, angle_error: float = 0.0):
        """
        Calculates wall repulsion forces using Lidar.
        
        Args:
            aggressive (bool): If True, allows getting closer to walls (smaller exclusion zone).
            angle_error (float): Current alignment error. Used to reduce repulsion if aimed correctly.
        """
        lidar = self.drone.lidar_values()
        angles = self.drone.lidar_rays_angles()
        if lidar is None or angles is None: return 0.0, 1.0 

        total_lat = 0.0
        min_dist_detected = 300.0

        # Tuning Parameters
        if aggressive:
            K_wall = 150.0; ignore_dist = 40.0; critical_dist = 10.0; slow_down_threshold = 40.0
        else:
            K_wall = 400.0; ignore_dist = 80.0; critical_dist = 20.0; slow_down_threshold = 60.0

        # [SMART REPULSION] If aimed correctly at a gap (small angle error), 
        # reduce wall repulsion to allow the drone to slip through.
        if abs(angle_error) < 0.2:
            K_wall *= 0.4

        step = 5
        for i in range(0, len(lidar), step):
            dist = lidar[i]
            if 10.0 < dist < ignore_dist:
                if dist < min_dist_detected: min_dist_detected = dist
                
                # Force is inversely proportional to distance^1.8
                force = K_wall / (dist ** 1.8)
                force = min(1.0, force) 

                angle_obs = angles[i]
                push_angle = angle_obs + math.pi 
                total_lat += force * math.sin(push_angle)

        # Calculate speed reduction factor based on proximity to walls
        speed_factor = np.clip((min_dist_detected - critical_dist) / slow_down_threshold, 0.3, 1.0)
        if aggressive: speed_factor = max(0.5, speed_factor)

        return total_lat, speed_factor

    def move_to_target_carrot(self) -> CommandsDict:
        """
        Main control loop using 'Carrot Chasing' logic.
        """
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # 1. Estimate Speed
        if self.last_pos is not None:
            move_dist = np.linalg.norm(self.drone.estimated_pos - self.last_pos)
            self.current_speed = move_dist 
        self.last_pos = self.drone.estimated_pos.copy()

        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)
        target_angle = math.atan2(delta_y, delta_x)

        is_reversing = (self.drone.grasped_wounded_persons() and self.drone.state == 'RETURNING')
        
        # "SUICIDE MODE" / FINAL APPROACH: 
        # If very close to target during rescue, disable safety to ensure contact.
        is_final_approach = (self.drone.state == 'RESCUING' and dist_to_target < 25.0)
        
        is_aggressive = (self.drone.state == 'RESCUING') or (dist_to_target < 80.0)

        # 2. Rotation Control
        if is_reversing:
            desired_angle = normalize_angle(target_angle + math.pi)
        else:
            desired_angle = target_angle

        angle_error = normalize_angle(desired_angle - self.drone.estimated_angle)
        
        # Tune KP based on phase
        if is_final_approach:
            KP_ROT = 4.0 # Stronger rotation to align for grasp
        else:
            KP_ROT = 2.5 # Smoother rotation for travel

        rotation_cmd = KP_ROT * angle_error
        rotation_cmd = np.clip(rotation_cmd, -1.0, 1.0)

        # 3. Velocity and Wall Avoidance
        wall_lat, wall_speed_factor = self.calculate_wall_repulsion(aggressive=is_aggressive, angle_error=angle_error)

        if is_final_approach:
            wall_lat = 0.0          # Disable wall repulsion
            wall_speed_factor = 1.0 # Disable speed reduction

        MAX_SPEED = 1.2 
        
        # ALIGN THEN MOVE: If angle error is large, reduce forward speed significantly.
        # This prevents "drifting" while turning.
        alignment_factor = max(0.0, math.cos(angle_error) ** 5)
        
        forward_cmd = MAX_SPEED * alignment_factor * wall_speed_factor

        # 4. Active Braking
        BRAKE_DIST = 60.0 
        if dist_to_target < BRAKE_DIST:
            # Maintain a minimum 'nudge' speed (0.15) to ensure we reach the target
            forward_cmd = max(0.15, dist_to_target * 0.03) 
            # Apply reverse thrust if moving too fast
            if self.current_speed > 5.0: forward_cmd = -0.5 

        if is_reversing: forward_cmd = -forward_cmd
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        # 5. Lateral Control (Drone + Wall Repulsion)
        cmd_lateral = 0.0
        _, drone_lat = self.calculate_repulsive_force()
        cmd_lateral = drone_lat + wall_lat

        if abs(angle_error) > 0.5 and not is_reversing:
            cmd_lateral += -0.5 * np.sign(angle_error) # Drift assist

        # 6. Grasper Control
        # Only activate if close enough (<20cm) AND aligned (angle error < 1.0 rad)
        # This prevents grasping with the back/side of the drone.
        can_grasp = (dist_to_target < 20.0) and (abs(angle_error) < 1.0)
        
        grasper_val = 1 if (self.drone.grasped_wounded_persons() or (self.drone.state == "RESCUING" and can_grasp)) else 0

        return {
            "forward": forward_cmd,
            "lateral": np.clip(cmd_lateral, -1.0, 1.0),
            "rotation": rotation_cmd,
            "grasper": grasper_val
        }