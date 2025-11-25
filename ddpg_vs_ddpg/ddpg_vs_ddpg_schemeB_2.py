"""
ddpg_vs_ddpg_schemeB.py

DDPG vs DDPG with Scheme B Shocks
Scheme B: ρ=0.95, σ_η=0.05 (high persistence, low variance)
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
    
    # Get price range for DDPG scaling
    price_min = env.price_grid[0]
    price_max = env.price_grid[-1]
    
    ddpg_agent1 = DDPGAgent(
        agent_id=0,
        state_dim=2,
        action_dim=1,
        seed=seed,
        price_min=price_min,
        price_max=price_max
    )
    ddpg_agent2 = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed+1000,
        price_min=price_min,
        price_max=price_max
    )
    
    shared_state = env.reset()
    profits_history = []
    prices_history = []
    
    state1 = np.array([shared_state[0], shared_state[1]], dtype=np.float32)
    state2 = np.array([shared_state[1], shared_state[0]], dtype=np.float32)
    
    for t in range(env.horizon):
        # Get continuous prices from DDPG agents
        price1, normalized_action1 = ddpg_agent1.select_action(state1, explore=True)
        price2, normalized_action2 = ddpg_agent2.select_action(state2, explore=True)
        
        # Use continuous prices directly
        actions = [price1, price2]
        shared_next_state, rewards, done, info = env.step(actions)
        
        next_state1 = np.array([shared_next_state[0], shared_next_state[1]], dtype=np.float32)
        next_state2 = np.array([shared_next_state[1], shared_next_state[0]], dtype=np.float32)
        
        # Store normalized actions for replay
        ddpg_agent1.remember(state1, normalized_action1, rewards[0], next_state1, done)
        ddpg_agent1.replay()
        ddpg_agent1.update_epsilon()
        
        ddpg_agent2.remember(state2, normalized_action2, rewards[1], next_state2, done)
        ddpg_agent2.replay()
        ddpg_agent2.update_epsilon()
        
        # Reset noise periodically for better exploration
        if t % 100 == 0:
            ddpg_agent1.reset_noise()
            ddpg_agent2.reset_noise()
        
        state1 = next_state1
        state2 = next_state2
        shared_state = shared_next_state
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate metrics from last 1000 timesteps
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
    
    # Calculate Delta (profit-based collusion metric)
    profit_range = pi_m - pi_n
    if profit_range > 1e-6:
        delta_ddpg1 = (avg_profit_ddpg1 - pi_n) / profit_range
        delta_ddpg2 = (avg_profit_ddpg2 - pi_n) / profit_range
    else:
        delta_ddpg1 = 0
        delta_ddpg2 = 0
    
    # Calculate RPDI (price-based collusion metric)
    price_range = p_m - p_n
    if price_range > 1e-6:
        rpdi_ddpg1 = (avg_price_ddpg1 - p_n) / price_range
        rpdi_ddpg2 = (avg_price_ddpg2 - p_n) / price_range
    else:
        rpdi_ddpg1 = 0
        rpdi_ddpg2 = 0
    
    return {
        'avg_price1': avg_price_ddpg1,
        'avg_price2': avg_price_ddpg2,
        'avg_profit1': avg_profit_ddpg1,
        'avg_profit2': avg_profit_ddpg2,
        'delta1': delta_ddpg1,
        'delta2': delta_ddpg2,
        'rpdi1': rpdi_ddpg1,
        'rpdi2': rpdi_ddpg2,
        'p_n': p_n,
        'p_m': p_m,
        'pi_n': pi_n,
        'pi_m': pi_m
    }


def main():
    # Scheme B: High persistence, low variance
    shock_cfg = {
        'enabled': True,
        'scheme': 'B',
        'rho': 0.95,
        'sigma_eta': 0.05,
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("DDPG vs DDPG - SCHEME B SHOCKS (ρ=0.95, σ_η=0.05)")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 10
    results = {}
    
    # Store individual run metrics
    run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': [], 
                        'profit1': [], 'profit2': []} for model in models}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model.upper()}")
        print(f"{'='*60}")
        
        model_benchmarks = all_benchmarks[model]
        
        # Print benchmark info
        print(f"  Benchmarks: P_N={model_benchmarks['p_N']:.4f}, P_M={model_benchmarks['p_M']:.4f}")
        print(f"              π_N={model_benchmarks['E_pi_N']:.4f}, π_M={model_benchmarks['E_pi_M']:.4f}")
        print(f"              Profit Range: {model_benchmarks['E_pi_M'] - model_benchmarks['E_pi_N']:.4f}")
        
        for run in range(num_runs):
            seed = SEED + run
            result = run_simulation(model, seed, shock_cfg, model_benchmarks)
            
            # Log individual run results
            run_logs[model]['delta1'].append(result['delta1'])
            run_logs[model]['delta2'].append(result['delta2'])
            run_logs[model]['rpdi1'].append(result['rpdi1'])
            run_logs[model]['rpdi2'].append(result['rpdi2'])
            run_logs[model]['profit1'].append(result['avg_profit1'])
            run_logs[model]['profit2'].append(result['avg_profit2'])
            
            print(f"\n  Run {run + 1}:")
            print(f"    DDPG 1 -> Price: {result['avg_price1']:.4f}, Profit: {result['avg_profit1']:.4f}, "
                  f"Δ: {result['delta1']:.4f}, RPDI: {result['rpdi1']:.4f}")
            print(f"    DDPG 2 -> Price: {result['avg_price2']:.4f}, Profit: {result['avg_profit2']:.4f}, "
                  f"Δ: {result['delta2']:.4f}, RPDI: {result['rpdi2']:.4f}")
        
        # Calculate model averages
        results[model] = {
            'Avg Price DDPG1': np.mean([run_logs[model]['profit1'][i] for i in range(num_runs)]),
            'Avg Price DDPG2': np.mean([run_logs[model]['profit2'][i] for i in range(num_runs)]),
            'Theo Price': model_benchmarks['p_N'],
            'Delta DDPG1': np.mean(run_logs[model]['delta1']),
            'Delta DDPG2': np.mean(run_logs[model]['delta2']),
            'RPDI DDPG1': np.mean(run_logs[model]['rpdi1']),
            'RPDI DDPG2': np.mean(run_logs[model]['rpdi2']),
            'Std Delta1': np.std(run_logs[model]['delta1']),
            'Std Delta2': np.std(run_logs[model]['delta2'])
        }
        
        print(f"\n  Model Summary:")
        print(f"    DDPG1: Δ = {results[model]['Delta DDPG1']:.4f} ± {results[model]['Std Delta1']:.4f}")
        print(f"    DDPG2: Δ = {results[model]['Delta DDPG2']:.4f} ± {results[model]['Std Delta2']:.4f}")
    
    # Generate summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    data = {
        'Model': [m.upper() for m in models],
        'P_N': [round(all_benchmarks[m]['p_N'], 4) for m in models],
        'P_M': [round(all_benchmarks[m]['p_M'], 4) for m in models],
        'π_N': [round(all_benchmarks[m]['E_pi_N'], 4) for m in models],
        'π_M': [round(all_benchmarks[m]['E_pi_M'], 4) for m in models],
        'Profit Range': [round(all_benchmarks[m]['E_pi_M'] - all_benchmarks[m]['E_pi_N'], 4) for m in models],
        'DDPG1 Δ': [round(results[m]['Delta DDPG1'], 4) for m in models],
        'DDPG2 Δ': [round(results[m]['Delta DDPG2'], 4) for m in models],
        'DDPG1 RPDI': [round(results[m]['RPDI DDPG1'], 4) for m in models],
        'DDPG2 RPDI': [round(results[m]['RPDI DDPG2'], 4) for m in models]
    }
    
    df = pd.DataFrame(data)
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/ddpg_vs_ddpg_schemeB.csv", index=False)
    print(df.to_string(index=False))
    
    # Overall averages
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
    print("[Results saved to ./results/ddpg_vs_ddpg_schemeB.csv]")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
