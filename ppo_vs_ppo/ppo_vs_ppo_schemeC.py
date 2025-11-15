import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import PPOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run PPO vs PPO simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    agents = [
        PPOAgent(agent_id=0, price_min=0.0, price_max=2.0, seed=seed),
        PPOAgent(agent_id=1, price_min=0.0, price_max=2.0, seed=seed)
    ]
    
    state = env.reset()
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]
    state0 = (current_prices[0], current_prices[1])
    state1 = (current_prices[1], current_prices[0])
    
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        action0, log_prob0, value0 = agents[0].select_action(state0, explore=True)
        action1, log_prob1, value1 = agents[1].select_action(state1, explore=True)
        
        actions = [action0, action1]
        next_state, rewards, done, info = env.step(actions)
        
        agents[0].store_transition(state0, action0, rewards[0], log_prob0, value0, done)
        agents[1].store_transition(state1, action1, rewards[1], log_prob1, value1, done)
        
        state = next_state
        current_prices = info['prices']
        state0 = (current_prices[0], current_prices[1])
        state1 = (current_prices[1], current_prices[0])
        
        prices_history.append(info['prices'])
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
        'scheme': 'C',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("PPO vs PPO - SCHEME C")
    print("Scheme C: ρ=0.7, σ_η=0.5 (high persistence, high variance)")
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
    df.to_csv("./results/ppo_vs_ppo_schemeC.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/ppo_vs_ppo_schemeC.csv]")


if __name__ == "__main__":
    main()
