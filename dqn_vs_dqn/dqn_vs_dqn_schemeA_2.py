"""
dqn_vs_dqn_schemeA.py
"""

"""
DQN vs DQN with Scheme A Shocks
Scheme A: ρ=0.3, σ_η=0.5 (low persistence, high variance)
"""

import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)  # Insert at beginning to prioritize it

from environments import MarketEnvContinuous
from agents import DQNAgent  # Assuming DQNAgent is added to agents or adjust import
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run DQN vs DQN simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    dqn_agent1 = DQNAgent(
        agent_id=0,
        state_dim=2,
        action_dim=env.N,
        seed=seed
    )
    dqn_agent2 = DQNAgent(
        agent_id=1,
        state_dim=2,
        action_dim=env.N,
        seed=seed
    )
    
    shared_state = env.reset()
    profits_history = []
    prices_history = []
    
    # Define agent-specific states (own, opp)
    state1 = (shared_state[0], shared_state[1])
    state2 = (shared_state[1], shared_state[0])
    
    for t in range(env.horizon):
        dqn_action1 = dqn_agent1.select_action(state1, explore=True)
        dqn_action2 = dqn_agent2.select_action(state2, explore=True)
        
        actions = [dqn_action1, dqn_action2]
        shared_next_state, rewards, done, info = env.step(actions)
        
        # Define agent-specific next_states
        next_state1 = (shared_next_state[0], shared_next_state[1])
        next_state2 = (shared_next_state[1], shared_next_state[0])
        
        dqn_agent1.remember(state1, dqn_action1, rewards[0], next_state1, done)
        dqn_agent1.replay()
        dqn_agent1.update_epsilon()
        
        dqn_agent2.remember(state2, dqn_action2, rewards[1], next_state2, done)
        dqn_agent2.replay()
        dqn_agent2.update_epsilon()
        
        state1 = next_state1
        state2 = next_state2
        shared_state = shared_next_state
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_dqn1 = np.mean(last_prices[:, 0])
    avg_price_dqn2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_dqn1 = np.mean(last_profits[:, 0])
    avg_profit_dqn2 = np.mean(last_profits[:, 1])
    
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    # Calculate Delta (profit-based)
    delta_dqn1 = (avg_profit_dqn1 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_dqn2 = (avg_profit_dqn2 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_dqn1 = (avg_price_dqn1 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_dqn2 = (avg_price_dqn2 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_dqn1, avg_price_dqn2, delta_dqn1, delta_dqn2, rpdi_dqn1, rpdi_dqn2, p_n

def main():
    shock_cfg = {
        'enabled': True,
        'rho': 0.3,
        'sigma_eta': 0.5
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("DQN vs DQN - SCHEME A SHOCKS")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 5
    results = {}
    
    # Store individual run results for logging
    run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_dqn1 = []
        avg_prices_dqn2 = []
        deltas_dqn1 = []
        deltas_dqn2 = []
        rpdis_dqn1 = []
        rpdis_dqn2 = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apdqn1, apdqn2, ddqn1, ddqn2, rdqn1, rdqn2, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_dqn1.append(apdqn1)
            avg_prices_dqn2.append(apdqn2)
            deltas_dqn1.append(ddqn1)
            deltas_dqn2.append(ddqn2)
            rpdis_dqn1.append(rdqn1)
            rpdis_dqn2.append(rdqn2)
            theo_prices.append(p_n)
            
            # Log individual run results
            print(f"\nRun {run + 1}:")
            print(f"  DQN 1  -> Delta: {ddqn1:.4f}, RPDI: {rdqn1:.4f}")
            print(f"  DQN 2  -> Delta: {ddqn2:.4f}, RPDI: {rdqn2:.4f}")
            
            # Store for later access
            run_logs[model]['delta1'].append(ddqn1)
            run_logs[model]['delta2'].append(ddqn2)
            run_logs[model]['rpdi1'].append(rdqn1)
            run_logs[model]['rpdi2'].append(rdqn2)
        
        results[model] = {
            'Avg Price DQN1': np.mean(avg_prices_dqn1),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DQN2': np.mean(avg_prices_dqn2),
            'Delta DQN1': np.mean(deltas_dqn1),
            'Delta DQN2': np.mean(deltas_dqn2),
            'RPDI DQN1': np.mean(rpdis_dqn1),
            'RPDI DQN2': np.mean(rpdis_dqn2)
        }
        
        print(f"\n  Model Average: DQN1 Δ = {results[model]['Delta DQN1']:.3f}, DQN2 Δ = {results[model]['Delta DQN2']:.3f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    data = {
        'Model': [m.upper() for m in models],
        'DQN1 Avg. Prices': [round(results[m]['Avg Price DQN1'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN2 Avg. Prices': [round(results[m]['Avg Price DQN2'], 2) for m in models],
        'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN1 Extra-profits Δ': [round(results[m]['Delta DQN1'], 2) for m in models],
        'DQN2 Extra-profits Δ': [round(results[m]['Delta DQN2'], 2) for m in models],
        'DQN1 RPDI': [round(results[m]['RPDI DQN1'], 2) for m in models],
        'DQN2 RPDI': [round(results[m]['RPDI DQN2'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/dqn_vs_dqn_schemeA.csv", index=False)
    print(df.to_string(index=False))
    
    # Calculate and print overall averages across all models
    print(f"\n{'='*80}")
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print(f"{'='*80}\n")
    
    avg_delta1 = np.mean([results[m]['Delta DQN1'] for m in models])
    avg_delta2 = np.mean([results[m]['Delta DQN2'] for m in models])
    avg_rpdi1 = np.mean([results[m]['RPDI DQN1'] for m in models])
    avg_rpdi2 = np.mean([results[m]['RPDI DQN2'] for m in models])
    
    print(f"DQN Agent 1:")
    print(f"  Average Delta (Δ):  {avg_delta1:.4f}")
    print(f"  Average RPDI:       {avg_rpdi1:.4f}")
    print(f"\nDQN Agent 2:")
    print(f"  Average Delta (Δ):  {avg_delta2:.4f}")
    print(f"  Average RPDI:       {avg_rpdi2:.4f}")
    
    print(f"\n{'='*80}")
    print("[Results saved to ./results/dqn_vs_dqn_schemeA.csv]")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
