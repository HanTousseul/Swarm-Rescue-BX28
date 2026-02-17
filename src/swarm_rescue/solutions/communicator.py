import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.forbidden = dict()
        self.other_drones_pos = dict()
        self.FORBIDDEN_RADIUS = 50

    # The function should_wait_in_queue() was removed
    # to completely eliminate the old queue-based waiting mechanism

    def is_target_taken_or_better_candidate(self, target_person_pos):
        # If there is no target, nothing to coordinate
        if target_person_pos is None:
            return False

        # If communication is disabled, we cannot coordinate with teammates
        if self.drone.communicator_is_disabled():
            return False
        
        # Distance from this drone to the target person
        my_dist = np.linalg.norm(self.drone.estimated_pos - target_person_pos)

        # Threshold to consider two target coordinates as the same person
        COORDINATE_MATCH_THRESHOLD = 50.0 

        # Iterate through all received messages from other drones
        for msg_package in self.drone.communicator.received_messages:

            # Extract message content (handle both tuple and raw dict formats)
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict):
                continue
            
            # Information shared by the other drone
            other_id = content.get("id")
            other_state = content.get("state")
            other_person_pos = content.get("person_pos")
            other_current_pos = content.get("current_pos")
            
            # Skip invalid or incomplete messages
            if other_person_pos is None or other_current_pos is None:
                continue
            
            # Distance between both drones' targeted persons
            dist_between_targets = np.linalg.norm(target_person_pos - other_person_pos)
            
            # --------------------------------------------------
            # If both drones are aiming at the SAME victim
            # --------------------------------------------------
            if dist_between_targets < COORDINATE_MATCH_THRESHOLD:
                
                # Distance from the other drone to the victim
                other_dist_to_person = np.linalg.norm(other_current_pos - other_person_pos)

                # --------------------------------------------------
                # CASE 1: The other drone has already secured the victim
                # (RETURNING or DROPPING)
                # -> Give up immediately
                # --------------------------------------------------
                if other_state in ["RETURNING", "DROPPING"]:
                    return True 

                # --------------------------------------------------
                # CASE 2: Both drones are competing for the same victim
                # (RESCUING vs RESCUING, EXPLORING vs RESCUING, etc.)
                #
                # Strategy:
                #   1. Compare distances
                #   2. Use ID as tie-breaker
                # --------------------------------------------------
                
                # If the opponent is significantly closer (> 20 px) -> yield
                if other_dist_to_person < my_dist - 20.0:
                    return True
                
                # If distances are similar (within 20 px) -> compare IDs
                # Smaller ID gets priority (simple deterministic rule
                # to break deadlocks and avoid oscillation)
                if abs(other_dist_to_person - my_dist) <= 20.0:
                    if other_id < self.drone.identifier:
                        return True

        # No better candidate found -> we can continue targeting
        return False
    
    def other_pos(self):
        """Save other drones' position and detect lost drones safely. In addition,
        modify self.forbidden if a drone marked as dead has comms again.
        """
        if not hasattr(self, "other_drones_pos") or self.other_drones_pos is None:
            self.other_drones_pos = {}
    
        # Initialize forbidden dictionary if not exists
        if not hasattr(self, "forbidden") or self.forbidden is None:
            self.forbidden = {}  # key: drone_id, value: last-known position
    
        # Create a safe list of IDs we currently know
        id_list = list(self.other_drones_pos.keys())
    
        # Check if communicator is available and has messages
        if hasattr(self.drone, "communicator") and hasattr(self.drone.communicator, "received_messages"):
            msgs = self.drone.communicator.received_messages
            if msgs:
                for msg in msgs:
                    if msg and isinstance(msg, dict) and "id" in msg and "current_pos" in msg:
                        drone_id = msg["id"]
                        if drone_id == self.drone.identifier:
                            continue
                        pos = np.array(msg["current_pos"])
                        self.other_drones_pos[drone_id] = pos
                        if drone_id in id_list:
                            id_list.remove(drone_id)
                        # Remove from forbidden if drone is back online
                        if drone_id in self.forbidden:
                            del self.forbidden[drone_id]
    
        # Any drone ID left in id_list has lost comms → mark as forbidden
        for drone_id in id_list:
            pos = self.other_drones_pos.get(drone_id)
            if pos is not None:
                self.forbidden[drone_id] = pos  # store last-known position
                
    def is_forbidden(self, pos, R=None):
        """Check if a position is inside forbidden zones (self.forbidden)."""
        radius = R if R is not None else self.FORBIDDEN_RADIUS
        pos = np.array(pos)
        for f_pos in self.forbidden.values():
            if np.linalg.norm(pos - f_pos) <= radius:
                return True
        return False
    
    def avoid_forbidden_target(self, target, step=50.0):
        """
        Adjust the target to avoid forbidden zones (lost drones).
        Returns a new target that is not inside any forbidden zone.
        """
        if target is None:
            return None
        target = np.array(target)
    
        if not self.is_forbidden(target):
            return target
    
        # Simple avoidance: try to slide in 8 directions around original target
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for angle in angles:
            candidate = target + step * np.array([np.cos(angle), np.sin(angle)])
            if not self.is_forbidden(candidate):
                return candidate
    
        # If no safe candidate found, just return original target
        return target

    def mark_as_stuck(self, pos):
        """Adds a stationary drone's position to the forbidden zones."""
        if pos is None: return
        # Create a unique key to distinguish from lost-comms drones 
        # by dividing by 5, so we can overwrite the jitters
        stuck_key = f"stuck_{int(pos[0]/5)}_{int(pos[1]/5)}" 
        self.forbidden[stuck_key] = np.array(pos)