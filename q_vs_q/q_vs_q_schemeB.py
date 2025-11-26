
import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q vs Q simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    agents = [QLearningAgent(env.N, agent_id=0), QLearningAgent(env.N, agent_id=1)]
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        actions = [agents[0].choose_action(state), agents[1].choose_action(state)]
        next_state, rewards, done, info = env.step(actions)
        
        agents[0].update(state, actions[0], rewards[0], next_state)
        agents[1].update(state, actions[1], rewards[1], next_state)
        
        state = next_state
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
        'scheme': 'B',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs Q-LEARNING - SCHEME B")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices1 = []
        avg_prices2 = []
        deltas = []
        theo_prices = []
        
        for run in range(NUM_RUNS):
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
    df.to_csv("./results/q_vs_q_schemeB.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/q_vs_q_schemeB.csv]")


if __name__ == "__main__":
    main()
