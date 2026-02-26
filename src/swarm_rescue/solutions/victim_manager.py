import numpy as np
import math
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

SAFE_DISTANCE_FROM_OTHER_DRONE = 30.0 

class VictimManager:
    """
    Manages the list of tracking victims.
    It handles identifying new victims, updating their positions, and filtering out 
    victims that are already being carried by other drones (Anti-Steal).
    """
    def __init__(self, drone):
        """Initialize victim tracking and association thresholds."""
        # List of victims. Each record: {'id': str, 'pos': np.array, 'ts': int}
        self.drone = drone
        self.registry = []
        self.next_id = 0
        self.merge_threshold = 80.0 # Distance threshold (cm) to consider two points as the same victim

    @staticmethod
    def _estimate_velocity_from_history(history):
        """
        Estimate velocity with linear interpolation/regression on recent positions.
        history: list of (ts, np.array([x, y]))
        """
        if history is None or len(history) < 2:
            return np.array([0.0, 0.0])

        t0 = history[0][0]
        times = np.array([float(ts - t0) for ts, _ in history], dtype=float)
        xs = np.array([float(pos[0]) for _, pos in history], dtype=float)
        ys = np.array([float(pos[1]) for _, pos in history], dtype=float)

        # Degenerate time axis fallback
        if np.allclose(times, times[0]):
            dt = max(1.0, float(history[-1][0] - history[-2][0]))
            return (history[-1][1] - history[-2][1]) / dt

        # Slope of best-fit line gives smoothed velocity.
        vx = np.polyfit(times, xs, 1)[0]
        vy = np.polyfit(times, ys, 1)[0]
        return np.array([vx, vy], dtype=float)

    def update_from_sensor(self, drone_pos, drone_angle, semantic_data, current_step):
        """
        Updates the victim registry based on current semantic sensor data.
        
        Args:
            drone_pos: Current position of the drone.
            drone_angle: Current heading of the drone.
            semantic_data: Data from the semantic sensor.
            current_step: Current simulation timestep.
        """
        if not semantic_data: return

        observed_victims = []
        observed_drones = []

        # 1. Parse Sensor Data
        for data in semantic_data:
            # Calculate absolute World Coordinates
            angle_global = drone_angle + data.angle
            wx = drone_pos[0] + data.distance * math.cos(angle_global)
            wy = drone_pos[1] + data.distance * math.sin(angle_global)
            pos = np.array([wx, wy])

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                # If simulator explicitly marks it as grasped, ignore immediately
                if getattr(data, 'grasped', False): continue
                observed_victims.append(pos)
            
            elif data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                observed_drones.append(pos)

        # 2. ANTI-STEAL LOGIC
        # Filter out victims that are physically too close to another drone.
        # Threshold: 30cm (Assumes if a victim is <30cm from a drone, it is being carried or rescued).

        for v_pos in observed_victims:
            is_being_carried = False
            for d_pos in observed_drones:
                if np.linalg.norm(v_pos - d_pos) < SAFE_DISTANCE_FROM_OTHER_DRONE:
                    is_being_carried = True
                    break
            
            if is_being_carried: continue # Skip this victim, do not register.

            # 3. Registration / Tracking Logic
            found = False
            for record in self.registry:
                dist = np.linalg.norm(record['pos'] - v_pos)
                if dist < self.merge_threshold:
                    # Update existing record with the new, potentially more accurate position
                    history = record.get('history')
                    if history is None:
                        history = [(record['ts'], record['pos'].copy())]
                    history.append((current_step, v_pos.copy()))
                    if len(history) > 6:
                        history = history[-6:]
                    v_new = self._estimate_velocity_from_history(history)
                    if 'vel' in record:
                        # Exponential smoothing for noisy semantic observations
                        record['vel'] = 0.60 * record['vel'] + 0.40 * v_new
                    else:
                        record['vel'] = v_new
                    record['history'] = history
                    record['pos'] = v_pos 
                    record['ts'] = current_step
                    found = True
                    break
            
            if not found:
                # Register new victim. Create a temporary ID based on coordinates.
                vid = f"{int(v_pos[0])}_{int(v_pos[1])}"
                self.registry.append({
                    'id': vid,
                    'pos': v_pos,
                    'ts': current_step,
                    'vel': np.array([0.0, 0.0]),
                    'history': [(current_step, v_pos.copy())]
                })

    def predict_intercept_point(self, drone_pos, observed_victim_pos, drone_speed_px_per_step: float = 5.8):
        """
        Predicts an intercept point ahead of a moving victim.
        Falls back to observed position when velocity is unknown/low.
        """
        if observed_victim_pos is None:
            return None

        best_record = None
        best_dist = float("inf")
        for record in self.registry:
            dist = np.linalg.norm(record['pos'] - observed_victim_pos)
            if dist < best_dist:
                best_dist = dist
                best_record = record

        if best_record is None:
            return observed_victim_pos.copy()

        victim_vel = best_record.get('vel', np.array([0.0, 0.0]))
        victim_speed = float(np.linalg.norm(victim_vel))
        if victim_speed < 0.05:
            return observed_victim_pos.copy()

        # Lead-time heuristic: farther targets and faster victims get more lead.
        dist_to_victim = np.linalg.norm(observed_victim_pos - drone_pos)
        rel_speed = max(1.0, drone_speed_px_per_step - 0.35 * victim_speed)
        lead_time = dist_to_victim / rel_speed
        lead_time = float(np.clip(lead_time, 2.0, 22.0))

        # Main intercept point.
        predicted = observed_victim_pos + victim_vel * lead_time

        # Push ahead on trajectory if we are trailing behind the moving person.
        tangent = victim_vel / max(1e-6, victim_speed)
        rel = drone_pos - observed_victim_pos
        trailing = float(np.dot(rel, tangent)) < -2.0
        if trailing:
            predicted = predicted + tangent * 40.0

        return predicted

    def get_velocity_for_position(self, observed_victim_pos):
        """Returns estimated victim velocity for the nearest tracked record."""
        if observed_victim_pos is None:
            return np.array([0.0, 0.0])

        best_record = None
        best_dist = float("inf")
        for record in self.registry:
            dist = np.linalg.norm(record['pos'] - observed_victim_pos)
            if dist < best_dist:
                best_dist = dist
                best_record = record

        if best_record is None:
            return np.array([0.0, 0.0])
        return best_record.get('vel', np.array([0.0, 0.0]))

    def is_victim_taken_care_of(self, position: tuple) -> bool: 
        '''
        Docstring for is_victim_taken_care_of
        
        :param self: Description
        :param position: Description
        :type position: tuple
        :return: Description
        :rtype: bool
        '''
        for elt in self.drone.comms.list_victims_taken_care_of:

            if np.hypot(position[0] - elt[0], position[1] - elt[1]) < SAFE_DISTANCE_FROM_OTHER_DRONE: return False
                
        return True

    def get_nearest_victim(self, drone_pos, blacklist):
        """Returns the position of the nearest registered victim."""
        if not self.registry: return None
        closest_dist = float('inf')
        best_pos = None
        for record in self.registry:
            check = False
            for bad in blacklist:
                dist = np.linalg.norm(record['pos'] - bad)
                if dist <= 20.0: check = True
            if check: continue
            dist = np.linalg.norm(record['pos'] - drone_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_pos = record['pos']
        return best_pos

    def delete_victim_at(self, position, radius=100.0):
        """
        Removes a victim from the registry at a specific location.
        Used when a victim is rescued or identified as a 'ghost' (sensor noise).
        """
        # Keep only victims that are OUTSIDE the deletion radius
        self.registry = [r for r in self.registry if np.linalg.norm(r['pos'] - position) > radius]
