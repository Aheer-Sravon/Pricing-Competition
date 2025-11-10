import numpy as np

class ModifiedQLearningAgent:
    def __init__(self, n_actions, alpha=0.15, gamma=0.95, beta=1.5e-4, agent_id=0):
        self.Q = np.zeros((n_actions, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.t = 0
        self.id = agent_id
        self.n_actions = n_actions

    def choose_action(self, state):
        opp_idx = state[1 - self.id]
        epsilon = np.exp(-self.beta * self.t)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[opp_idx])

    def update(self, state, action, reward, next_state):
        opp_idx = state[1 - self.id]
        # Modified: Assume opponent keeps current price for TD target
        td_target = reward + self.gamma * np.max(self.Q[opp_idx])
        self.Q[opp_idx, action] += self.alpha * (td_target - self.Q[opp_idx, action])
        self.t += 1
