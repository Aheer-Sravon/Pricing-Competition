import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
   
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
   
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
   
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for DDPG"""
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 300)
        self.fc3 = nn.Linear(300, action_dim)
       
        # Initialize weights
        self._init_weights()
   
    def _init_weights(self):
        nn.init.uniform_(self.fc1.weight, -1/np.sqrt(self.fc1.in_features), 1/np.sqrt(self.fc1.in_features))
        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(self.fc2.in_features), 1/np.sqrt(self.fc2.in_features))
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
   
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """Critic network for DDPG"""
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)
       
        # Initialize weights
        self._init_weights()
   
    def _init_weights(self):
        nn.init.uniform_(self.fc1.weight, -1/np.sqrt(self.fc1.in_features), 1/np.sqrt(self.fc1.in_features))
        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(self.fc2.in_features), 1/np.sqrt(self.fc2.in_features))
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
   
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
   
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
   
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent for pricing competition simulation"""
    def __init__(
        self,
        agent_id,
        state_dim,
        action_dim,
        hidden_dim=400,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_size=1000000,
        batch_size=64,
        seed=None
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
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
       
        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
       
        # Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-2)
       
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
       
        # Noise process
        self.noise = OUNoise(action_dim)
       
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
   
    def select_action(self, state, explore=True):
        """Select action using actor network with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
       
        if explore:
            noise = self.noise.sample() * self.epsilon
            action = action + noise
            action = np.clip(action, -1, 1)
       
        return action
   
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
   
    def replay(self):
        """Train on a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
       
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
       
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
       
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
       
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
       
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
       
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
       
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
       
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
   
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
   
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
   
    def reset_noise(self):
        """Reset the noise process"""
        self.noise.reset()
