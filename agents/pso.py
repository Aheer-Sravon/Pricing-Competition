import numpy as np

class PSOAgent:
    def __init__(self, env, agent_id, num_particles=20, w=0.729, c1=1.494, c2=1.494, price_min=0.0, price_max=2.0):
        self.env = env
        self.id = agent_id
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.price_min = price_min
        self.price_max = price_max
        self.particles = np.random.uniform(self.price_min, self.price_max, self.num_particles)
        self.velocities = np.zeros(self.num_particles)
        self.p_best_positions = self.particles.copy()
        self.p_best_scores = np.full(self.num_particles, -np.inf)
        self.g_best_position = np.mean(self.particles)
        self.g_best_score = -np.inf

    def update(self, opp_price):
        prices = np.full(self.env.n_firms, opp_price)
        scores = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            prices[self.id] = self.particles[i]
            _, profits = self.env.calculate_demand_and_profit(prices, np.zeros(self.env.n_firms))
            scores[i] = profits[self.id]
        
        better_p = scores > self.p_best_scores
        self.p_best_scores[better_p] = scores[better_p]
        self.p_best_positions[better_p] = self.particles[better_p]
        
        if np.max(scores) > self.g_best_score:
            self.g_best_score = np.max(scores)
            self.g_best_position = self.particles[np.argmax(scores)]
        
        r1 = np.random.uniform(0, 1, self.num_particles)
        r2 = np.random.uniform(0, 1, self.num_particles)
        self.velocities = (
            self.w * self.velocities +
            self.c1 * r1 * (self.p_best_positions - self.particles) +
            self.c2 * r2 * (self.g_best_position - self.particles)
        )
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.price_min, self.price_max)

    def choose_price(self):
        return self.g_best_position
