"""
ddpg_vs_ddpg_2.py

Simulation comparing DDPG vs DDPG agents.
"""

import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import DDPGAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run DDPG vs DDPG simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    ddpg_agent1 = DDPGAgent(
        agent_id=0,
        state_dim=2,
        action_dim=1,
        seed=seed
    )
    ddpg_agent2 = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed+1000
    )
    
    shared_state = env.reset()
    profits_history = []
    prices_history = []
    
    state1 = np.array([shared_state[0], shared_state[1]], dtype=np.float32)
    state2 = np.array([shared_state[1], shared_state[0]], dtype=np.float32)
    
    for t in range(env.horizon):
        ddpg_action1_raw = ddpg_agent1.select_action(state1, explore=True)
        ddpg_action2_raw = ddpg_agent2.select_action(state2, explore=True)
        
        # Map continuous actions to discrete indices
        ddpg_action1 = int((ddpg_action1_raw[0] + 1) / 2 * (env.N - 1))
        ddpg_action1 = np.clip(ddpg_action1, 0, env.N - 1)
        
        ddpg_action2 = int((ddpg_action2_raw[0] + 1) / 2 * (env.N - 1))
        ddpg_action2 = np.clip(ddpg_action2, 0, env.N - 1)
        
        actions = [ddpg_action1, ddpg_action2]
        shared_next_state, rewards, done, info = env.step(actions)
        
        next_state1 = np.array([shared_next_state[0], shared_next_state[1]], dtype=np.float32)
        next_state2 = np.array([shared_next_state[1], shared_next_state[0]], dtype=np.float32)
        
        ddpg_agent1.remember(state1, ddpg_action1_raw, rewards[0], next_state1, done)
        ddpg_agent1.replay()
        ddpg_agent1.update_epsilon()
        
        ddpg_agent2.remember(state2, ddpg_action2_raw, rewards[1], next_state2, done)
        ddpg_agent2.replay()
        ddpg_agent2.update_epsilon()
        
        state1 = next_state1
        state2 = next_state2
        shared_state = shared_next_state
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_ddpg1 = np.mean(last_prices[:, 0])
    avg_price_ddpg2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_ddpg1 = np.mean(last_profits[:, 0])
    avg_profit_ddpg2 = np.mean(last_profits[:, 1])
    
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    delta_ddpg1 = (avg_profit_ddpg1 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg2 = (avg_profit_ddpg2 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    rpdi_ddpg1 = (avg_price_ddpg1 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg2 = (avg_price_ddpg2 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_ddpg1, avg_price_ddpg2, delta_ddpg1, delta_ddpg2, rpdi_ddpg1, rpdi_ddpg2, p_n

def main():
    shock_cfg = {
        'enabled': False
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("DDPG vs DDPG - NO SHOCKS")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 5
    results = {}
    
    run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_ddpg1 = []
        avg_prices_ddpg2 = []
        deltas_ddpg1 = []
        deltas_ddpg2 = []
        rpdis_ddpg1 = []
        rpdis_ddpg2 = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apddpg1, apddpg2, dddpg1, dddpg2, rddpg1, rddpg2, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_ddpg1.append(apddpg1)
            avg_prices_ddpg2.append(apddpg2)
            deltas_ddpg1.append(dddpg1)
            deltas_ddpg2.append(dddpg2)
            rpdis_ddpg1.append(rddpg1)
            rpdis_ddpg2.append(rddpg2)
            theo_prices.append(p_n)
            
            print(f"\nRun {run + 1}:")
            print(f"  DDPG 1  -> Delta: {dddpg1:.4f}, RPDI: {rddpg1:.4f}")
            print(f"  DDPG 2  -> Delta: {dddpg2:.4f}, RPDI: {rddpg2:.4f}")
            
            run_logs[model]['delta1'].append(dddpg1)
            run_logs[model]['delta2'].append(dddpg2)
            run_logs[model]['rpdi1'].append(rddpg1)
            run_logs[model]['rpdi2'].append(rddpg2)
        
        results[model] = {
            'Avg Price DDPG1': np.mean(avg_prices_ddpg1),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DDPG2': np.mean(avg_prices_ddpg2),
            'Delta DDPG1': np.mean(deltas_ddpg1),
            'Delta DDPG2': np.mean(deltas_ddpg2),
            'RPDI DDPG1': np.mean(rpdis_ddpg1),
            'RPDI DDPG2': np.mean(rpdis_ddpg2)
        }
        
        print(f"\n  Model Average: DDPG1 Δ = {results[model]['Delta DDPG1']:.3f}, DDPG2 Δ = {results[model]['Delta DDPG2']:.3f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    data = {
        'Model': [m.upper() for m in models],
        'DDPG1 Avg. Prices': [round(results[m]['Avg Price DDPG1'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DDPG2 Avg. Prices': [round(results[m]['Avg Price DDPG2'], 2) for m in models],
        'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
        'DDPG1 Extra-profits Δ': [round(results[m]['Delta DDPG1'], 2) for m in models],
        'DDPG2 Extra-profits Δ': [round(results[m]['Delta DDPG2'], 2) for m in models],
        'DDPG1 RPDI': [round(results[m]['RPDI DDPG1'], 2) for m in models],
        'DDPG2 RPDI': [round(results[m]['RPDI DDPG2'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/ddpg_vs_ddpg_2.csv", index=False)
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print(f"{'='*80}\n")
    
    avg_delta1 = np.mean([results[m]['Delta DDPG1'] for m in models])
    avg_delta2 = np.mean([results[m]['Delta DDPG2'] for m in models])
    avg_rpdi1 = np.mean([results[m]['RPDI DDPG1'] for m in models])
    avg_rpdi2 = np.mean([results[m]['RPDI DDPG2'] for m in models])
    
    print(f"DDPG Agent 1:")
    print(f"  Average Delta (Δ):  {avg_delta1:.4f}")
    print(f"  Average RPDI:       {avg_rpdi1:.4f}")
    print(f"\nDDPG Agent 2:")
    print(f"  Average Delta (Δ):  {avg_delta2:.4f}")
    print(f"  Average RPDI:       {avg_rpdi2:.4f}")
    
    print(f"\n{'='*80}")
    print("[Results saved to ./results/ddpg_vs_ddpg_2.csv]")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
