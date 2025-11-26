import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, DDPGAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50


def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q-Learning vs DDPG simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    q_agent = QLearningAgent(env.N, agent_id=0, price_grid=env.price_grid)
    ddpg_agent = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed,
        price_min=price_min,
        price_max=price_max
    )
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        # Q-Learning selects discrete action index
        q_action_idx = q_agent.choose_action(state)
        q_price = env.price_grid[q_action_idx]
        
        # DDPG selects continuous price
        ddpg_state = state.astype(np.float32)
        ddpg_price, ddpg_norm = ddpg_agent.select_action(ddpg_state, explore=True)
        
        actions = [q_price, ddpg_price]
        next_state, rewards, done, info = env.step(actions)
        
        # Update Q-Learning
        q_agent.update(state, q_action_idx, rewards[0], next_state)
        
        # Update DDPG
        next_ddpg_state = next_state.astype(np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_norm, rewards[1], next_ddpg_state, done)
        ddpg_agent.replay()
        
        if t % 100 == 0:
            ddpg_agent.update_epsilon()
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Get benchmarks
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    
    # Calculate Delta (profit-based)
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_q = (avg_price_q - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_q, avg_price_ddpg, delta_q, delta_ddpg, rpdi_q, rpdi_ddpg, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'B',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs DDPG - SCHEME B")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_q = []
        avg_prices_ddpg = []
        deltas_q = []
        deltas_ddpg = []
        rpdis_q = []
        rpdis_ddpg = []
        theo_prices = []
        
        for run in range(NUM_RUNS):
            seed = SEED + run
            apq, apd, dq, dd, rq, rd, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_q.append(apq)
            avg_prices_ddpg.append(apd)
            deltas_q.append(dq)
            deltas_ddpg.append(dd)
            rpdis_q.append(rq)
            rpdis_ddpg.append(rd)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DDPG': np.mean(avg_prices_ddpg),
            'Delta Q': np.mean(deltas_q),
            'Delta DDPG': np.mean(deltas_ddpg),
            'RPDI Q': np.mean(rpdis_q),
            'RPDI DDPG': np.mean(rpdis_ddpg)
        }
        
        print(f"  Completed: Q Δ = {results[model]['Delta Q']:.3f}, DDPG Δ = {results[model]['Delta DDPG']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 2) for m in models],
        'Q RPDI': [round(results[m]['RPDI Q'], 2) for m in models],
        'DDPG RPDI': [round(results[m]['RPDI DDPG'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/q_vs_ddpg.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print("=" * 80)
    
    avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
    avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
    avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
    avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])
    
    print("\nQ-Learning:")
    print(f"  Average Delta (Δ): {avg_delta_q:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_q:.4f}")
    print("\nDDPG:")
    print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")
    
    print("\n[Results saved to ./results/q_vs_ddpg.csv]")


if __name__ == "__main__":
    main()
