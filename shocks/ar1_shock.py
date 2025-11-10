import numpy as np

class AR1_Shock:
    def __init__(self, rho, sigma_eta, seed=None):
        self.rho = rho
        self.sigma_eta = sigma_eta
        self.rng = np.random.RandomState(seed)
        self.current = 0.0

    def reset(self):
        self.current = 0.0

    def generate_next(self):
        eta = self.rng.normal(0, self.sigma_eta)
        self.current = self.rho * self.current + eta
        return self.current
