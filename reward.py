from typing import Dict, List, Any, Tuple
import numpy as np
import math

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:   # Penalties for each deactivated agent
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Penalties for not finding the goal
            rewards[i] = -0.1

    return rewards, evacuated_agents

def calculate_reward(prev_state: Dict[str, Any], 
                    next_state: Dict[str, Any], 
                    action: int, 
                    prev_dist_from_goal: List[float], 
                    next_dist_from_goal: List[float]) -> float:
    """
    Calculate the reward for the current step.
    
    Args:
        prev_state: The previous state.
        next_state: The current state.
        action: The action taken.
        prev_dist_from_goal: Previous distances from goal for all agents.
        next_dist_from_goal: Current distances from goal for all agents.
    
    Returns:
        A float representing the reward.
    """
    reward = 0.0
    
    # Evacuation reward (agent reached goal)
    if next_state["status"] == 1:  # 1 = evacuated (reached goal)
        reward += 1000.0
    
    # Deactivation penalty (agent hit obstacle)
    elif next_state["status"] == 2:  # 2 = deactivated (hit obstacle or wall)
        reward -= 100.0
    
    else:
        # Step penalty (encourages efficiency)
        reward -= 0.1
        
        # Progress reward - bonus for getting closer to goal
        agent_id = next_state["id"]
        distance_improvement = prev_dist_from_goal[agent_id] - next_dist_from_goal[agent_id]
        progress_reward = distance_improvement * 10.0
        reward += progress_reward
        
        # Safety reward - based on LIDAR readings
        # Main LIDAR (front)
        if next_state["lidar_main_type"] > 0:  # If there's an obstacle
            danger_factor = max(0, 1.0 - (next_state["lidar_main_dist"] / 10.0))
            # Higher penalty the closer we are and for more dangerous obstacles
            obstacle_type_factor = 1.0 if next_state["lidar_main_type"] == 1 else 2.0  # Higher for dynamic obstacles
            safety_penalty = -danger_factor * obstacle_type_factor * 2.0
            reward += safety_penalty
        
        # Side LIDARs (left and right)
        for lidar_dir, lidar_dist in [("lidar_left_dist", "lidar_left_type"), 
                                    ("lidar_right_dist", "lidar_right_type")]:
            if next_state[lidar_dist] > 0:  # If there's an obstacle
                danger_factor = max(0, 1.0 - (next_state[lidar_dir] / 5.0))
                obstacle_type_factor = 1.0 if next_state[lidar_dist] == 1 else 1.5  # Higher for dynamic obstacles
                safety_penalty = -danger_factor * obstacle_type_factor * 1.0
                reward += safety_penalty
        
        # Direction efficiency - reward for facing the goal
        goal_dx = next_state["goal_x"] - next_state["x"]
        goal_dy = next_state["goal_y"] - next_state["y"]
        goal_angle = math.atan2(goal_dy, goal_dx)
        
        # Agent orientation (0:right, 1:up, 2:left, 3:down)
        agent_angle = next_state["o"] * (math.pi / 2)
        
        # Calculate angle difference (0 means agent is facing goal)
        angle_diff = abs(((goal_angle - agent_angle + math.pi) % (2 * math.pi)) - math.pi)
        
        # Higher reward when facing goal (angle_diff close to 0)
        direction_reward = (math.pi - angle_diff) / math.pi
        reward += direction_reward * 0.5
        
        # Action-specific adjustment
        if action == 0:  # stay steady
            # Small penalty for staying still unless very close to an obstacle
            if (next_state["lidar_main_dist"] < 2 or 
                next_state["lidar_left_dist"] < 1 or 
                next_state["lidar_right_dist"] < 1):
                reward += 0.05  # Small reward for caution
            else:
                reward -= 0.05  # Small penalty for inaction
                
        elif action == 1 and direction_reward > 0.7:  # moving forward while facing goal
            reward += 0.2  # Bonus for efficient movement
    
    return reward