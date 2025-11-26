import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import DQNAgent, DDPGAgent
from theoretical_benchmarks import TheoreticalBenchmarks
from shocks import AR1_Shock

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50


def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run DQN vs DDPG simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize DQN agent (agent 0)
    dqn_agent = DQNAgent(agent_id=0, state_dim=2, action_dim=env.N, seed=seed)
    
    # Initialize DDPG agent (agent 1)
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
        # DQN selects discrete action index
        dqn_action = dqn_agent.select_action(state, explore=True)
        
        # DDPG selects continuous price
        ddpg_state = state.astype(np.float32)
        ddpg_price, ddpg_norm = ddpg_agent.select_action(ddpg_state, explore=True)
        
        actions = [dqn_action, ddpg_price]
        next_state, rewards, done, info = env.step(actions)
        
        # Update DQN
        dqn_agent.remember(state, dqn_action, rewards[0], next_state, done)
        dqn_agent.replay()
        
        # Update DDPG
        next_ddpg_state = next_state.astype(np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_norm, rewards[1], next_ddpg_state, done)
        ddpg_agent.replay()
        
        # Decay exploration
        if t % 100 == 0:
            dqn_agent.update_epsilon()
            ddpg_agent.update_epsilon()
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_dqn = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_dqn = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Get benchmarks
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    
    # Calculate Delta (profit-based)
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_dqn = (avg_price_dqn - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_dqn, avg_price_ddpg, delta_dqn, delta_ddpg, rpdi_dqn, rpdi_ddpg, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'A',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("DQN vs DDPG - SCHEME A")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_dqn = []
        avg_prices_ddpg = []
        deltas_dqn = []
        deltas_ddpg = []
        rpdis_dqn = []
        rpdis_ddpg = []
        theo_prices = []
        
        for run in range(NUM_RUNS):
            seed = SEED + run
            ap_dqn, ap_ddpg, d_dqn, d_ddpg, r_dqn, r_ddpg, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_dqn.append(ap_dqn)
            avg_prices_ddpg.append(ap_ddpg)
            deltas_dqn.append(d_dqn)
            deltas_ddpg.append(d_ddpg)
            rpdis_dqn.append(r_dqn)
            rpdis_ddpg.append(r_ddpg)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price DQN': np.mean(avg_prices_dqn),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DDPG': np.mean(avg_prices_ddpg),
            'Delta DQN': np.mean(deltas_dqn),
            'Delta DDPG': np.mean(deltas_ddpg),
            'RPDI DQN': np.mean(rpdis_dqn),
            'RPDI DDPG': np.mean(rpdis_ddpg)
        }
        
        print(f"  Completed: DQN Δ = {results[model]['Delta DQN']:.3f}, DDPG Δ = {results[model]['Delta DDPG']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 2) for m in models],
        'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 2) for m in models],
        'DQN RPDI': [round(results[m]['RPDI DQN'], 2) for m in models],
        'DDPG RPDI': [round(results[m]['RPDI DDPG'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/dqn_vs_ddpg_schemeA.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print("=" * 80)
    
    avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
    avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
    avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])
    avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])
    
    print("\nDQN:")
    print(f"  Average Delta (Δ): {avg_delta_dqn:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_dqn:.4f}")
    print("\nDDPG:")
    print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")
    
    print("\n[Results saved to ./results/dqn_vs_ddpg_schemeA.csv]")


if __name__ == "__main__":
    main()
