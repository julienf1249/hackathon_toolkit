import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MyAgent():
    def __init__(self, num_agents: int):
        # Parameters
        self.num_agents = num_agents
        self.n_actions = 7  # 0:stay, 1-4:movement, 5-6:rotation
        self.state_dim = 26  # State dimension after preprocessing
        
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory_size = 100000
        self.update_target_freq = 1000
        self.training_steps = 0
        
        # Create networks for each agent
        self.policy_nets = []
        self.target_nets = []
        self.optimizers = []
        self.replay_buffers = []
        
        for i in range(num_agents):
            # Policy network (for action selection)
            policy_net = DQN(self.state_dim, self.n_actions).to(self.device)
            
            # Target network (for stable Q-value targets)
            target_net = DQN(self.state_dim, self.n_actions).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()  # Target network is only used for inference
            
            # Optimizer
            optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
            
            # Experience replay buffer
            replay_buffer = deque(maxlen=self.memory_size)
            
            self.policy_nets.append(policy_net)
            self.target_nets.append(target_net)
            self.optimizers.append(optimizer)
            self.replay_buffers.append(replay_buffer)
        
        # Store previous states and actions for learning
        self.previous_states = [None] * num_agents
        self.previous_actions = [None] * num_agents
        self.loss_criterion = nn.MSELoss()
        
        # For parallel processing
        self.current_transitions = []

    def _state_to_tensor(self, state):
        """Convert state array to a tensor with feature engineering"""
        # Extract basic features
        pos_x, pos_y, orientation = state[0], state[1], state[2]
        goal_x, goal_y = state[4], state[5]
        
        # Calculate relative position to goal
        rel_x = goal_x - pos_x
        rel_y = goal_y - pos_y
        
        # Distance to goal
        goal_dist = np.sqrt(rel_x**2 + rel_y**2)
        
        # Relative angle to goal (in radians)
        goal_angle = np.arctan2(rel_y, rel_x) - orientation * np.pi/2
        goal_angle = (goal_angle + np.pi) % (2*np.pi) - np.pi  # Normalize to [-π, π]
        
        # One-hot encode orientation
        orientation_one_hot = [0, 0, 0, 0]
        orientation_one_hot[int(orientation)] = 1
        
        # Extract LIDAR data (main + 8 directions)
        lidar_data = []
        for i in range(6, 24, 2):
            distance = state[i]
            object_type = state[i+1]
            lidar_data.extend([distance, object_type])
        
        # Compile all features
        features = [
            pos_x, pos_y,
            rel_x, rel_y,
            goal_dist,
            np.sin(goal_angle), np.cos(goal_angle),
            *orientation_one_hot,
            *lidar_data
        ]
        
        return torch.FloatTensor(features).to(self.device)

    def get_action(self, states, evaluation=False):
        """Choose actions using epsilon-greedy policy"""
        actions = []
        
        for i in range(self.num_agents):
            state = states[i]
            
            # Skip deactivated or evacuated agents (-1 values)
            if state[0] == -1:
                actions.append(0)  # Stay in place
                continue
            
            state_tensor = self._state_to_tensor(state)
            
            # Epsilon-greedy action selection
            if evaluation or random.random() > self.epsilon:
                # Exploitation: choose best action
                with torch.no_grad():  # No need to compute gradients for action selection
                    q_values = self.policy_nets[i](state_tensor)
                    action = q_values.argmax().item()
            else:
                # Exploration: choose random action
                action = random.randint(0, self.n_actions - 1)
            
            # Store current state and action for learning
            self.previous_states[i] = state
            self.previous_actions[i] = action
            
            actions.append(action)
        
        return actions

    def update_policy(self, actions, new_states, rewards):
        """Update policy using experience replay and DQN"""
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store transitions
        for i in range(self.num_agents):
            if self.previous_states[i] is None:
                continue
                
            # Check if the agent is still active
            is_terminal = new_states[i][0] == -1
            
            # Store transition in replay buffer
            self.replay_buffers[i].append((
                self.previous_states[i],
                self.previous_actions[i],
                rewards[i],
                new_states[i] if not is_terminal else None,
                is_terminal
            ))
        
        # Train on mini-batches
        self.training_steps += 1
        
        # Process all agents in parallel for efficiency
        for i in range(self.num_agents):
            if len(self.replay_buffers[i]) < self.batch_size:
                continue
                
            # Sample a mini-batch
            mini_batch = random.sample(self.replay_buffers[i], self.batch_size)
            
            # Extract batch data with parallel processing
            states = torch.stack([self._state_to_tensor(s) for s, _, _, _, _ in mini_batch])
            actions = torch.LongTensor([a for _, a, _, _, _ in mini_batch]).to(self.device)
            rewards = torch.FloatTensor([r for _, _, r, _, _ in mini_batch]).to(self.device)
            
            # Process non-terminal states
            non_terminal_mask = torch.BoolTensor([not t for _, _, _, _, t in mini_batch]).to(self.device)
            next_states = torch.stack([
                self._state_to_tensor(ns) if ns is not None else self._state_to_tensor(np.zeros_like(mini_batch[0][0]))
                for _, _, _, ns, _ in mini_batch
            ])
            
            # Compute current Q values
            current_q_values = self.policy_nets[i](states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute next Q values
            next_q_values = torch.zeros(self.batch_size, device=self.device)
            if non_terminal_mask.any():
                with torch.no_grad():
                    next_q_values[non_terminal_mask] = self.target_nets[i](next_states[non_terminal_mask]).max(1)[0]
            
            # Compute target Q values
            target_q_values = rewards + self.gamma * next_q_values
            
            # Compute loss
            loss = self.loss_criterion(current_q_values, target_q_values)
            
            # Optimize the model
            self.optimizers[i].zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            for param in self.policy_nets[i].parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizers[i].step()
        
        # Update target networks periodically
        if self.training_steps % self.update_target_freq == 0:
            for i in range(self.num_agents):
                self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def save_model(self, filename='dqn_agent.pt'):
        """Save models to a file"""
        save_dict = {
            'epsilon': self.epsilon,
            'models': [net.state_dict() for net in self.policy_nets]
        }
        torch.save(save_dict, filename)
        
    def load_model(self, filename='dqn_agent.pt'):
        """Load models from a file"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.epsilon = checkpoint['epsilon']
        
        # Load model parameters
        for i, state_dict in enumerate(checkpoint['models']):
            self.policy_nets[i].load_state_dict(state_dict)
            self.target_nets[i].load_state_dict(state_dict)