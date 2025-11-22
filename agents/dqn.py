import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class DQNAgent:
    """
    Deep Q-Network Agent implementation following Mnih et al. (2015)
    and the dynamic pricing paper specifications.
   
    Key features:
    - Uses continuous state representation (actual prices, not indices)
    - Implements Double DQN with separate target network
    - Proper experience replay with adequate buffer size
    - Gradient clipping and Huber loss for stability
    - GPU acceleration for faster training in pricing competitions
    """
   
    def __init__(self, agent_id, state_dim=2, action_dim=15, seed=None):
        """
        Initialize DQN Agent.
       
        Args:
            agent_id: Agent identifier (0 or 1)
            state_dim: Dimension of state space (2 for price tuple)
            action_dim: Number of discrete actions (price grid size)
            seed: Random seed for reproducibility
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
       
        # Hyperparameters from papers
        self.gamma = 0.99 # Discount factor
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate per episode
        self.learning_rate = 0.0001 # Adam optimizer learning rate
        self.batch_size = 128 # Minibatch size for training
        self.memory_size = 50000 # Experience replay buffer size
        self.target_update_freq = 500 # Steps between target network updates
       
        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
       
        # For deterministic behavior on GPU
        if self.device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
       
        # Initialize neural networks
        self.network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
       
        # Copy weights to target network
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval() # Set target network to evaluation mode
       
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
       
        # Initialize experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
       
        # Step counter for target network updates
        self.steps = 0
       
    def _build_network(self):
        """
        Build the neural network architecture.
        Deeper network than original for better representation learning.
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
   
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
       
        Args:
            state: Current state (np.array of prices)
            action: Action taken (index)
            reward: Reward received
            next_state: Next state (np.array of prices)
            done: Episode termination flag
        """
        self.memory.append((state, action, reward, next_state, done))
   
    def select_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy.
       
        Args:
            state: Current state as np.array([own_price, opponent_price])
            explore: Whether to use exploration (training) or exploitation (evaluation)
       
        Returns:
            action: Selected action index
        """
        # Epsilon-greedy exploration
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
       
        # Exploitation: choose best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()
   
    def replay(self):
        """
        Train the network on a minibatch sampled from experience replay.
        Implements Double DQN with target network.
        """
        # Need minimum experiences before training
        if len(self.memory) < self.batch_size:
            return
       
        # Sample minibatch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
       
        # Separate batch components
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
       
        # Current Q values for taken actions
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
       
        # Double DQN implementation
        # Step 1: Use main network to select best actions for next states
        with torch.no_grad():
            next_actions = self.network(next_states).argmax(1)
           
        # Step 2: Use target network to evaluate those actions
        next_q_values = self.target_network(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
       
        # Compute targets (Bellman equation)
        targets = rewards + (1 - dones) * self.gamma * next_q_values
       
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), targets.detach())
       
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
       
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
       
        self.optimizer.step()
       
        # Increment step counter
        self.steps += 1
       
        # Periodically update target network
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
   
    def update_target_network(self):
        """
        Copy weights from main network to target network.
        This stabilizes training by keeping targets fixed for multiple updates.
        """
        self.target_network.load_state_dict(self.network.state_dict())
   
    def update_epsilon(self):
        """
        Decay exploration rate.
        Called at the end of each episode or at regular intervals.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
   
    def save(self, filepath):
        """
        Save model weights.
       
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
   
    def load(self, filepath):
        """
        Load model weights.
       
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
