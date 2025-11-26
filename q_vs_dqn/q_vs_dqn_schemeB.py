"""
Q-Learning vs DQN with Scheme A Shocks
Scheme A: ρ=0.3, σ_η=0.5 (low persistence, high variance) - Independent shocks
"""

import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, DQNAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks, verbose=True):
    """Run a single simulation of Q-Learning vs DQN."""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    q_agent = QLearningAgent(
        env.N,
        agent_id=0,
        price_grid=env.price_grid
    )
    
    dqn_agent = DQNAgent(
        agent_id=1, 
        state_dim=2,
        action_dim=env.N,
        seed=seed
    )
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        dqn_action = dqn_agent.select_action(state, explore=True)
        
        actions = [q_action, dqn_action]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        dqn_agent.remember(state, dqn_action, rewards[1], next_state, done)
        dqn_agent.replay()
        
        if t % 100 == 0:
            dqn_agent.update_epsilon()
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_dqn = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_dqn = np.mean(last_profits[:, 1])
    
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    rpdi_q = (avg_price_q - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_dqn = (avg_price_dqn - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_q, avg_price_dqn, delta_q, delta_dqn, rpdi_q, rpdi_dqn, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'B',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)

    print("=" * 80)
    print("Q-LEARNING vs DQN - SCHEME B")
    print("=" * 80)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 5
    results = {}
    run_logs = {model: {'delta_q': [], 'delta_dqn': [], 'rpdi_q': [], 'rpdi_dqn': []} for model in models}
    
    for model in models:
        model_benchmarks = all_benchmarks[model]
        
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")
        print(f"Nash price: {model_benchmarks['p_N']:.3f}, Monopoly price: {model_benchmarks['p_M']:.3f}")
        
        avg_prices_q, avg_prices_dqn = [], []
        deltas_q, deltas_dqn = [], []
        rpdis_q, rpdis_dqn = [], []
        
        for run in range(num_runs):
            seed = SEED + run
            apq, apd, dq, dd, rq, rd, p_n = run_simulation(
                model, seed, shock_cfg, model_benchmarks, verbose=(run == 0)
            )
            
            avg_prices_q.append(apq)
            avg_prices_dqn.append(apd)
            deltas_q.append(dq)
            deltas_dqn.append(dd)
            rpdis_q.append(rq)
            rpdis_dqn.append(rd)
            
            print(f"\nRun {run + 1}:")
            print(f"  Q-Learning -> Delta: {dq:.4f}, RPDI: {rq:.4f}")
            print(f"  DQN        -> Delta: {dd:.4f}, RPDI: {rd:.4f}")
            
            run_logs[model]['delta_q'].append(dq)
            run_logs[model]['delta_dqn'].append(dd)
            run_logs[model]['rpdi_q'].append(rq)
            run_logs[model]['rpdi_dqn'].append(rd)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Avg Price DQN': np.mean(avg_prices_dqn),
            'Theo Price': model_benchmarks['p_N'],
            'Delta Q': np.mean(deltas_q),
            'Delta DQN': np.mean(deltas_dqn),
            'RPDI Q': np.mean(rpdis_q),
            'RPDI DQN': np.mean(rpdis_dqn)
        }
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
        'Q Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'DQN Δ': [round(results[m]['Delta DQN'], 2) for m in models],
        'Q RPDI': [round(results[m]['RPDI Q'], 2) for m in models],
        'DQN RPDI': [round(results[m]['RPDI DQN'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/q_vs_dqn_schemeA.csv", index=False)
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print(f"{'='*80}\n")
    
    avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
    avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
    avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
    avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])
    
    print(f"Q-Learning: Avg Δ = {avg_delta_q:.4f}, Avg RPDI = {avg_rpdi_q:.4f}")
    print(f"DQN:        Avg Δ = {avg_delta_dqn:.4f}, Avg RPDI = {avg_rpdi_dqn:.4f}")
    print("\n[Results saved to ./results/q_vs_dqn_schemeA.csv]")


if __name__ == "__main__":
    main()
