import numpy as np

class PSOAgent:
    """PSO with full restart when stuck (no improvement for N steps)"""
    def __init__(self, env, agent_id, num_particles=10, w0=0.025, c1=1.75, c2=1.75, 
                 price_min=0.0, price_max=2.0, restart_threshold=300):
        self.env = env
        self.id = agent_id
        self.num_particles = num_particles
        self.w0 = w0
        self.c1 = c1
        self.c2 = c2
        self.price_min = price_min
        self.price_max = price_max
        self.restart_threshold = restart_threshold
        
        # Initialize
        self.particles = np.random.uniform(self.price_min, self.price_max, self.num_particles)
        self.velocities = np.zeros(self.num_particles)
        self.p_best_positions = self.particles.copy()
        self.p_best_scores = np.full(self.num_particles, -np.inf)
        self.g_best_position = np.mean(self.particles)
        self.g_best_score = -np.inf
        
        # Restart tracking
        self.t = 0
        self.stuck_counter = 0
        self.last_g_best = -np.inf
        self.restart_count = 0

    def update(self, opp_price):
        # Inertia weight with floor
        w = max(0.4, 0.9 - 0.5 * self.t / 10000)
        
        # Evaluate particles
        prices = np.full(self.env.n_firms, opp_price)
        scores = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            prices[self.id] = self.particles[i]
            _, profits = self.env.calculate_demand_and_profit(prices, np.zeros(self.env.n_firms))
            scores[i] = profits[self.id]
        
        # Update bests
        better_p = scores > self.p_best_scores
        self.p_best_scores[better_p] = scores[better_p]
        self.p_best_positions[better_p] = self.particles[better_p]
        
        max_score = np.max(scores)
        if max_score > self.g_best_score:
            self.g_best_score = max_score
            self.g_best_position = self.particles[np.argmax(scores)]
        
        # Check if stuck (no significant improvement)
        if self.g_best_score <= self.last_g_best + 1e-6:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_g_best = self.g_best_score
        
        # RESTART if stuck too long
        if self.stuck_counter >= self.restart_threshold:
            self.particles = np.random.uniform(self.price_min, self.price_max, self.num_particles)
            self.velocities = np.zeros(self.num_particles)
            self.p_best_positions = self.particles.copy()
            self.p_best_scores = np.full(self.num_particles, -np.inf)
            # Keep g_best to remember best ever found
            self.stuck_counter = 0
            self.restart_count += 1
            # if self.restart_count <= 10:  # Only print first 10
                # print(f"      [PSO RESTART #{self.restart_count} at t={self.t}]")
        
        # Update velocities and positions
        r1 = np.random.uniform(0, 1, self.num_particles)
        r2 = np.random.uniform(0, 1, self.num_particles)
        
        self.velocities = (
            w * self.velocities +
            self.c1 * r1 * (self.p_best_positions - self.particles) +
            self.c2 * r2 * (self.g_best_position - self.particles)
        )
        self.velocities = np.clip(self.velocities, -0.3, 0.3)
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.price_min, self.price_max)
        
        self.t += 1

    def choose_price(self):
        return self.g_best_position
