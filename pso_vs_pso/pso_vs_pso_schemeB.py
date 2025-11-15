"""
PSO vs PSO with Scheme B Shocks
Uses theoretical_benchmarks.py for proper benchmark calculations
"""

import numpy as np
import pandas as pd
from environments import MarketEnvContinuous
from agents import PSOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run PSO vs PSO simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    agents = [PSOAgent(env, agent_id=0), PSOAgent(env, agent_id=1)]
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    current_prices = [agents[0].choose_price(), agents[1].choose_price()]
    
    for t in range(env.horizon):
        agents[0].update(current_prices[1])
        agents[1].update(current_prices[0])
        current_prices = [agents[0].choose_price(), agents[1].choose_price()]
        
        next_state, rewards, done, info = env.step(current_prices)
        state = next_state
        
        prices_history.append(current_prices)
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price1 = np.mean(last_prices[:, 0])
    avg_price2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit = np.mean(last_profits[:, 0])
    
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    
    delta = (avg_profit - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price1, avg_price2, delta, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'B',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("PSO vs PSO - SCHEME B (INDEPENDENT SHOCKS)")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 50
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices1 = []
        avg_prices2 = []
        deltas = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            ap1, ap2, d, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices1.append(ap1)
            avg_prices2.append(ap2)
            deltas.append(d)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Firm 1': np.mean(avg_prices1),
            'Theo Price': np.mean(theo_prices),
            'Avg Price Firm 2': np.mean(avg_prices2),
            'Delta': np.mean(deltas)
        }
        
        print(f"  Completed: Δ = {results[model]['Delta']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Firm 1 Avg. Prices': [round(results[m]['Avg Price Firm 1'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'Firm 2 Avg. Prices': [round(results[m]['Avg Price Firm 2'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Extra-profits Δ': [round(results[m]['Delta'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/pso_vs_pso_schemeB.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/pso_vs_pso_schemeB.csv]")


if __name__ == "__main__":
    main()
