"""
Theoretical Benchmark Calculations for Market Models with Demand Shocks

This script calculates Nash and Monopoly benchmarks for all three models
(Logit, Hotelling, Linear) under different shock schemes.

For Logit: Shocks affect expected benchmarks due to convexity
For Hotelling/Linear: Expected benchmarks equal no-shock benchmarks
"""

import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd


class TheoreticalBenchmarks:
    """Calculate theoretical Nash and Monopoly prices/profits for market models"""
    
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def calculate_logit_benchmarks(self, shock_cfg=None):
        """
        Calculate Nash and Monopoly benchmarks for Logit model
        With shocks: Uses Monte Carlo integration
        Without shocks: Uses analytical formulas
        """
        # Model parameters
        a = 2.0
        c = 1.0
        mu = 0.25
        a0 = 0.0
        
        if shock_cfg is None or not shock_cfg.get('enabled', False):
            # No shocks - use standard benchmarks
            p_N = 1.473
            p_M = 1.925
            
            # Calculate profits at these prices
            exp_N = np.exp((a - p_N) / mu)
            exp_outside = np.exp(a0 / mu)
            q_N = exp_N / (2 * exp_N + exp_outside)
            E_pi_N = (p_N - c) * q_N
            
            exp_M = np.exp((a - p_M) / mu)
            q_M = exp_M / (2 * exp_M + exp_outside)
            E_pi_M = (p_M - c) * q_M
            
            return {
                'p_N': p_N,
                'E_pi_N': E_pi_N,
                'p_M': p_M,
                'E_pi_M': E_pi_M,
                'shock_enabled': False
            }
        
        # With shocks - use Monte Carlo
        scheme_params = {
            'A': {'rho': 0.3, 'sigma_eta': 0.5},
            'B': {'rho': 0.95, 'sigma_eta': 0.05},
            'C': {'rho': 0.9, 'sigma_eta': 0.3}
        }
        
        scheme = shock_cfg.get('scheme', 'A')
        params = scheme_params[scheme.upper()]
        rho = params['rho']
        sigma_eta = params['sigma_eta']
        
        # Unconditional variance of shocks
        sigma2 = sigma_eta**2 / (1 - rho**2)
        sigma = np.sqrt(sigma2)
        
        # Monte Carlo samples
        N = 10000
        shock_mode = shock_cfg.get('mode', 'independent')
        
        if shock_mode == 'independent':
            xi_samples = np.random.normal(0, sigma, (N, 2))
        else:  # correlated
            xi_samples = np.random.normal(0, sigma, N)
        
        # Expected profit functions
        def E_pi1(p1, p2):
            if shock_mode == 'independent':
                exps1 = np.exp((a - p1 + xi_samples[:, 0]) / mu)
                exps2 = np.exp((a - p2 + xi_samples[:, 1]) / mu)
            else:
                exps1 = np.exp((a - p1 + xi_samples) / mu)
                exps2 = np.exp((a - p2 + xi_samples) / mu)
            den = exps1 + exps2 + np.exp(a0 / mu)
            qs = exps1 / den
            return np.mean((p1 - c) * qs)
        
        def E_pi2(p1, p2):
            if shock_mode == 'independent':
                exps1 = np.exp((a - p1 + xi_samples[:, 0]) / mu)
                exps2 = np.exp((a - p2 + xi_samples[:, 1]) / mu)
            else:
                exps1 = np.exp((a - p1 + xi_samples) / mu)
                exps2 = np.exp((a - p2 + xi_samples) / mu)
            den = exps1 + exps2 + np.exp(a0 / mu)
            qs = exps2 / den
            return np.mean((p2 - c) * qs)
        
        # Nash equilibrium
        def best_response(p_j):
            def neg_E_pi(p_i):
                return -E_pi1(p_i, p_j)
            res = minimize_scalar(neg_E_pi, bounds=(c, c + 5), method='bounded')
            return res.x
        
        p_guess = 1.5
        for _ in range(50):
            p_guess = best_response(p_guess)
        p_N = p_guess
        E_pi_N = E_pi1(p_N, p_N)
        
        # Monopoly
        def neg_E_joint(p):
            return -(E_pi1(p, p) + E_pi2(p, p))
        
        res_M = minimize_scalar(neg_E_joint, bounds=(c, c + 5), method='bounded')
        p_M = res_M.x
        E_pi_M = E_pi1(p_M, p_M)
        
        return {
            'p_N': p_N,
            'E_pi_N': E_pi_N,
            'p_M': p_M,
            'E_pi_M': E_pi_M,
            'shock_enabled': True,
            'scheme': scheme,
            'mode': shock_mode,
            'sigma': sigma
        }
    
    def calculate_hotelling_benchmarks(self, shock_cfg=None):
        """
        Calculate Nash and Monopoly benchmarks for Hotelling model
        Shocks don't affect expected benchmarks (linearity in net shock)
        """
        # Model parameters
        c = 0.0
        v_bar = 1.75
        theta = 1.0
        
        # Standard benchmarks (unchanged with shocks)
        p_N = 1.00
        p_M = 1.25
        
        # Calculate profits
        # At Nash: both firms charge 1, split market equally
        q_N = 0.5
        E_pi_N = p_N * q_N
        
        # At Monopoly: both charge 1.25, split market equally
        q_M = 0.5
        E_pi_M = p_M * q_M
        
        return {
            'p_N': p_N,
            'E_pi_N': E_pi_N,
            'p_M': p_M,
            'E_pi_M': E_pi_M,
            'shock_enabled': shock_cfg is not None and shock_cfg.get('enabled', False),
            'note': 'Shocks do not affect expected benchmarks for Hotelling'
        }
    
    def calculate_linear_benchmarks(self, shock_cfg=None):
        """
        Calculate Nash and Monopoly benchmarks for Linear model
        Shocks don't affect expected benchmarks (linearity in shocks)
        """
        # Model parameters
        c = 0.0
        a_bar = 1.0
        d = 0.25
        
        # Standard benchmarks (unchanged with shocks)
        p_N = 0.4286
        p_M = 0.5
        
        # Calculate profits
        denominator = 1 - d**2
        
        # At Nash
        q_N = (a_bar - p_N - d * (a_bar - p_N)) / denominator
        E_pi_N = p_N * q_N
        
        # At Monopoly
        q_M = (a_bar - p_M - d * (a_bar - p_M)) / denominator
        E_pi_M = p_M * q_M
        
        return {
            'p_N': p_N,
            'E_pi_N': E_pi_N,
            'p_M': p_M,
            'E_pi_M': E_pi_M,
            'shock_enabled': shock_cfg is not None and shock_cfg.get('enabled', False),
            'note': 'Shocks do not affect expected benchmarks for Linear'
        }
    
    def calculate_all_benchmarks(self, shock_cfg=None):
        """Calculate benchmarks for all three models"""
        results = {
            'logit': self.calculate_logit_benchmarks(shock_cfg),
            'hotelling': self.calculate_hotelling_benchmarks(shock_cfg),
            'linear': self.calculate_linear_benchmarks(shock_cfg)
        }
        return results
    
    def generate_benchmark_table(self, shock_configs):
        """
        Generate a comprehensive table of benchmarks for different configurations
        
        Args:
            shock_configs: List of shock configurations to test
        
        Returns:
            pandas DataFrame with benchmarks
        """
        data = []
        
        for config in shock_configs:
            config_name = config.get('name', 'No Config')
            benchmarks = self.calculate_all_benchmarks(config)
            
            for model, bench in benchmarks.items():
                data.append({
                    'Configuration': config_name,
                    'Model': model.upper(),
                    'Nash Price': round(bench['p_N'], 4),
                    'Nash Profit': round(bench['E_pi_N'], 4),
                    'Monopoly Price': round(bench['p_M'], 4),
                    'Monopoly Profit': round(bench['E_pi_M'], 4),
                    'Shock Enabled': bench['shock_enabled']
                })
        
        return pd.DataFrame(data)


# Example usage and testing
if __name__ == "__main__":
    calculator = TheoreticalBenchmarks(seed=99)
    
    # Define shock configurations to test
    shock_configs = [
        {'name': 'No Shocks', 'enabled': False},
        {'name': 'Scheme A - Independent', 'enabled': True, 'scheme': 'A', 'mode': 'independent'},
        {'name': 'Scheme A - Correlated', 'enabled': True, 'scheme': 'A', 'mode': 'correlated'},
        {'name': 'Scheme B - Independent', 'enabled': True, 'scheme': 'B', 'mode': 'independent'},
        {'name': 'Scheme B - Correlated', 'enabled': True, 'scheme': 'B', 'mode': 'correlated'},
        {'name': 'Scheme C - Independent', 'enabled': True, 'scheme': 'C', 'mode': 'independent'},
        {'name': 'Scheme C - Correlated', 'enabled': True, 'scheme': 'C', 'mode': 'correlated'},
    ]
    
    # Generate comprehensive benchmark table
    print("=" * 100)
    print("THEORETICAL BENCHMARKS FOR ALL CONFIGURATIONS")
    print("=" * 100)
    
    benchmark_df = calculator.generate_benchmark_table(shock_configs)
    print(benchmark_df.to_string(index=False))
    
    # Save to CSV
    benchmark_df.to_csv("theoretical_benchmarks.csv", index=False)
    print("\n[Benchmarks saved to theoretical_benchmarks.csv]")
    
    # Print specific comparison for Logit model
    print("\n" + "=" * 100)
    print("LOGIT MODEL: IMPACT OF SHOCKS ON THEORETICAL BENCHMARKS")
    print("=" * 100)
    
    logit_only = benchmark_df[benchmark_df['Model'] == 'LOGIT']
    print(logit_only.to_string(index=False))
    
    # Calculate percentage changes for Scheme A Independent
    no_shock_logit = calculator.calculate_logit_benchmarks(None)
    with_shock_logit = calculator.calculate_logit_benchmarks({'enabled': True, 'scheme': 'A', 'mode': 'independent'})
    
    print("\n" + "=" * 100)
    print("SCHEME A (INDEPENDENT) IMPACT ON LOGIT MODEL")
    print("=" * 100)
    print(f"Nash Price:     {no_shock_logit['p_N']:.4f} → {with_shock_logit['p_N']:.4f} "
          f"({100*(with_shock_logit['p_N']/no_shock_logit['p_N']-1):+.2f}%)")
    print(f"Nash Profit:    {no_shock_logit['E_pi_N']:.4f} → {with_shock_logit['E_pi_N']:.4f} "
          f"({100*(with_shock_logit['E_pi_N']/no_shock_logit['E_pi_N']-1):+.2f}%)")
    print(f"Monopoly Price: {no_shock_logit['p_M']:.4f} → {with_shock_logit['p_M']:.4f} "
          f"({100*(with_shock_logit['p_M']/no_shock_logit['p_M']-1):+.2f}%)")
    print(f"Monopoly Profit:{no_shock_logit['E_pi_M']:.4f} → {with_shock_logit['E_pi_M']:.4f} "
          f"({100*(with_shock_logit['E_pi_M']/no_shock_logit['E_pi_M']-1):+.2f}%)")
    print("\nNote: Convexity of exp(ξ/μ) causes upward bias in expected utilities")
