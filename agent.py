import numpy as np
import pickle
from collections import defaultdict

class MyAgent():
    def __init__(self, num_agents: int):
        # Parameters
        self.num_agents = num_agents
        self.n_actions = 7  # 0:stay, 1-4:movement, 5-6:rotation
        
        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.training_steps = 0
        
        # Initialize Q-tables as defaultdict to handle new states
        self.q_tables = [defaultdict(lambda: np.zeros(self.n_actions)) for _ in range(num_agents)]
        
        # Store previous states and actions for learning
        self.previous_states = [None] * num_agents
        self.previous_actions = [None] * num_agents
        
        # Random generator
        self.rng = np.random.default_rng()

    def _state_to_key(self, state):
        """Convert state array to a hashable key by focusing on important elements"""
        # Agent position and orientation
        pos_x, pos_y, orientation = state[0], state[1], state[2]
        
        # Goal position
        goal_x, goal_y = state[4], state[5]
        
        # Extract LIDAR information for main direction
        main_dist = state[6]
        main_type = state[7]
        
        # Secondary LIDAR information
        sec1_dist = state[8]
        sec1_type = state[9]
        sec2_dist = state[10]
        sec2_type = state[11]
        
        # Calculate relative position to goal
        rel_x = goal_x - pos_x
        rel_y = goal_y - pos_y
        
        # Return a tuple of relevant state information
        return (
            int(pos_x), int(pos_y), int(orientation),
            int(rel_x), int(rel_y),
            int(main_dist), int(main_type),
            int(sec1_dist), int(sec1_type),
            int(sec2_dist), int(sec2_type)
        )

    def get_action(self, states, evaluation=False):
        """Choose actions using epsilon-greedy policy"""
        actions = []
        
        for i in range(self.num_agents):
            state = states[i]
            
            # Skip deactivated or evacuated agents (-1 values)
            if state[0] == -1:
                actions.append(0)  # Stay in place
                continue
            
            state_key = self._state_to_key(state)
            q_values = self.q_tables[i][state_key]
            
            # Epsilon-greedy action selection
            if evaluation or self.rng.random() > self.epsilon:
                # Exploitation: choose best action
                action = np.argmax(q_values)
            else:
                # Exploration: choose random action
                action = self.rng.integers(0, self.n_actions)
            
            # Store current state and action for learning
            self.previous_states[i] = state_key
            self.previous_actions[i] = action
            
            actions.append(action)
        
        return actions

    def update_policy(self, actions, new_states, rewards):
        """Update Q-values using Q-learning update rule"""
        # Decay epsilon over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_steps += 1
        
        for i in range(self.num_agents):
            # Skip if no previous state (first step of an episode)
            if self.previous_states[i] is None:
                continue
            
            # Get previous state and action
            prev_state = self.previous_states[i]
            prev_action = self.previous_actions[i]
            reward = rewards[i]
            
            # Get current state
            current_state = new_states[i]
            
            # For deactivated or evacuated agents, we use the reward directly
            if current_state[0] == -1:
                # Q-learning update: Q(s,a) = Q(s,a) + alpha * (r - Q(s,a))
                # For terminal states, there's no future reward
                self.q_tables[i][prev_state][prev_action] += self.alpha * (
                    reward - self.q_tables[i][prev_state][prev_action]
                )
            else:
                # Convert current state to key
                current_key = self._state_to_key(current_state)
                
                # Q-learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s')) - Q(s,a))
                self.q_tables[i][prev_state][prev_action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_tables[i][current_key]) - 
                    self.q_tables[i][prev_state][prev_action]
                )
    
    def save_model(self, filename='q_agent.pkl'):
        """Save Q-tables to a file"""
        # Convert defaultdicts to regular dicts for serialization
        q_tables_dict = [{str(k): v for k, v in q_table.items()} for q_table in self.q_tables]
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_tables': q_tables_dict,
                'epsilon': self.epsilon
            }, f)
        
    def load_model(self, filename='q_agent.pkl'):
        """Load Q-tables from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        # Reconstruct defaultdicts
        self.q_tables = []
        for q_dict in data['q_tables']:
            q_table = defaultdict(lambda: np.zeros(self.n_actions))
            for k, v in q_dict.items():
                q_table[eval(k)] = v
            self.q_tables.append(q_table)
            
        self.epsilon = data['epsilon']