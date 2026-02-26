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
        # List of victims. Each record: {'id': str, 'pos': np.array, 'ts': int}
        self.drone = drone
        self.registry = []
        self.next_id = 0
        self.rescued_victims = [] # List of victim rescued
        self.merge_threshold = 20.0 # Distance threshold (cm) to consider two points as the same victim

    def sync_rescued_victims(self, global_rescued):
        """Sync list of victim rescued from other drons"""
        for pos in global_rescued:
            is_new = True
            for my_pos in self.rescued_victims:
                if np.linalg.norm(pos - my_pos) < 20.0:
                    is_new = False
                    break
            if is_new:
                self.rescued_victims.append(pos)
        
        self.registry = [
            r for r in self.registry 
            if not any(np.linalg.norm(r['pos'] - b_pos) < 30.0 for b_pos in self.rescued_victims)
        ]

    def sync_registry(self, global_victims):
        """Syncronize victim encounter list"""
        for tv in global_victims:
            tv_pos = tv['pos']
            
            # Pass this victim if it's explored
            if any(np.linalg.norm(tv_pos - b_pos) < 30.0 for b_pos in self.rescued_victims):
                continue
                
            found = False
            for record in self.registry:
                # If victim in registry, update timestep
                if np.linalg.norm(record['pos'] - tv_pos) < self.merge_threshold:
                    if tv['ts'] > record['ts']:
                        record['pos'] = tv_pos
                        record['ts'] = tv['ts']
                    found = True
                    break
            
            # If new, write in registry
            if not found:
                self.registry.append(tv)

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
            is_already_rescued = False
            for rescued_pos in self.rescued_victims:
                if np.linalg.norm(v_pos - rescued_pos) < 30.0:
                    is_already_rescued = True
                    break
            if is_already_rescued: 
                continue

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
                    'ts': current_step
                })

    def is_victim_taken_care_of(self, position: tuple) -> bool: 
        '''
        Docstring for is_victim_taken_care_of
        Checks if a victim at the given position is already being rescued by a teammate.
        
        :param self: Description
        :param position: Description
        :type position: tuple
        :return: Description
        :rtype: bool
        '''
        # Check if it's rescued
        for r_pos in self.rescued_victims:
            if np.linalg.norm(position - r_pos) < 30.0: return True
        my_dist = float(np.linalg.norm(self.drone.estimated_pos - position))
        my_id = self.drone.identifier

        for other_pos, other_dist, other_id in self.drone.comms.list_victims_taken_care_of:
            if np.hypot(position[0] - other_pos[0], position[1] - other_pos[1]) < SAFE_DISTANCE_FROM_OTHER_DRONE:
                if other_dist < my_dist:
                    return True
                elif abs(other_dist - my_dist) < 1.0 and other_id > my_id:
                    return True
                
        return False

    def get_nearest_victim(self, drone_pos, blacklist):
        """Returns the position of the nearest registered victim."""
        if not self.registry: return None
        self.registry = [
            record for record in self.registry
            if not any(np.linalg.norm(record['pos'] - r_pos) < 30.0 for r_pos in self.rescued_victims)
        ]
        if not self.registry: return None
        closest_dist = float('inf')
        best_pos = None
        for record in self.registry:
            if self.is_victim_taken_care_of(record['pos']):
                continue
            
            check = False
            for bad in blacklist:
                dist = np.linalg.norm(record['pos'] - bad)
                if dist <= 30.0: check = True
            if check: continue
            
            dist = np.linalg.norm(record['pos'] - drone_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_pos = record['pos']
        return best_pos

    def delete_victim_at(self, position, radius=30.0):
        """
        Removes a victim from the registry at a specific location.
        Used when a victim is rescued or identified as a 'ghost' (sensor noise).
        """
        # Keep only victims that are OUTSIDE the deletion radius
        self.rescued_victims.append(position.copy())
        self.registry = [r for r in self.registry if np.linalg.norm(r['pos'] - position) > radius]