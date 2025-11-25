import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.15, gamma=0.95, beta=1.5e-4, agent_id=0):
        self.Q = np.zeros((n_actions, n_actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Exploration decay rate
        self.t = 0
        self.id = agent_id
        self.n_actions = n_actions

    def choose_action(self, state):
        opp_idx = state[1 - self.id]
        epsilon = np.exp(-self.beta * self.t)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[opp_idx])  # Exploit

    def update(self, state, action, reward, next_state):
        opp_idx = state[1 - self.id]
        next_opp_idx = next_state[1 - self.id]
        td_target = reward + self.gamma * np.max(self.Q[next_opp_idx])
        self.Q[opp_idx, action] += self.alpha * (td_target - self.Q[opp_idx, action])
        self.t += 1
