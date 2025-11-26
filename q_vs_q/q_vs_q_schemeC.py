"""
Q-Learning vs Q-Learning with Scheme C Shocks
Scheme C: ρ=0.9, σ_η=0.3 (moderate persistence, moderate variance) - Independent shocks
"""

import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50

def run_simulation(model, seed, shock_cfg, benchmarks):
    np.random.seed(seed)
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    agents = [
        QLearningAgent(env.N, agent_id=0, price_grid=env.price_grid),
        QLearningAgent(env.N, agent_id=1, price_grid=env.price_grid)
    ]
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
    avg_profit1 = np.mean(last_profits[:, 0])
    avg_profit2 = np.mean(last_profits[:, 1])
    
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    delta1 = (avg_profit1 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta2 = (avg_profit2 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    rpdi1 = (avg_price1 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi2 = (avg_price2 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price1, avg_price2, delta1, delta2, rpdi1, rpdi2, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'C',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    print("=" * 80)
    print("Q-LEARNING vs Q-LEARNING - SCHEME C")
    print("=" * 80)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}
    
    for model in models:
        model_benchmarks = all_benchmarks[model]
        
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")
        
        avg_prices1, avg_prices2 = [], []
        deltas1, deltas2 = [], []
        rpdis1, rpdis2 = [], []
        
        for run in range(NUM_RUNS):
            seed = SEED + run
            ap1, ap2, d1, d2, r1, r2, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices1.append(ap1)
            avg_prices2.append(ap2)
            deltas1.append(d1)
            deltas2.append(d2)
            rpdis1.append(r1)
            rpdis2.append(r2)
            
            print(f"\nRun {run + 1}:")
            print(f"  Q-Agent 1 -> Delta: {d1:.4f}, RPDI: {r1:.4f}")
            print(f"  Q-Agent 2 -> Delta: {d2:.4f}, RPDI: {r2:.4f}")
            
            run_logs[model]['delta1'].append(d1)
            run_logs[model]['delta2'].append(d2)
            run_logs[model]['rpdi1'].append(r1)
            run_logs[model]['rpdi2'].append(r2)
        
        results[model] = {
            'Avg Price 1': np.mean(avg_prices1),
            'Avg Price 2': np.mean(avg_prices2),
            'Theo Price': model_benchmarks['p_N'],
            'Delta 1': np.mean(deltas1),
            'Delta 2': np.mean(deltas2),
            'RPDI 1': np.mean(rpdis1),
            'RPDI 2': np.mean(rpdis2)
        }
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q1 Avg. Prices': [round(results[m]['Avg Price 1'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'Q2 Avg. Prices': [round(results[m]['Avg Price 2'], 2) for m in models],
        'Q1 Δ': [round(results[m]['Delta 1'], 2) for m in models],
        'Q2 Δ': [round(results[m]['Delta 2'], 2) for m in models],
        'Q1 RPDI': [round(results[m]['RPDI 1'], 2) for m in models],
        'Q2 RPDI': [round(results[m]['RPDI 2'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/q_vs_q_schemeC.csv", index=False)
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print(f"{'='*80}\n")
    
    avg_delta1 = np.mean([results[m]['Delta 1'] for m in models])
    avg_delta2 = np.mean([results[m]['Delta 2'] for m in models])
    avg_rpdi1 = np.mean([results[m]['RPDI 1'] for m in models])
    avg_rpdi2 = np.mean([results[m]['RPDI 2'] for m in models])
    
    print(f"Q-Agent 1: Avg Δ = {avg_delta1:.4f}, Avg RPDI = {avg_rpdi1:.4f}")
    print(f"Q-Agent 2: Avg Δ = {avg_delta2:.4f}, Avg RPDI = {avg_rpdi2:.4f}")
    print("\n[Results saved to ./results/q_vs_q_schemeC.csv]")


if __name__ == "__main__":
    main()
