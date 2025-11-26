import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import PSOAgent, DDPGAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50


def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run PSO vs DDPG simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize PSO agent (agent 0)
    pso_agent = PSOAgent(env, agent_id=0, price_min=price_min, price_max=price_max)
    
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
        # PSO updates with DDPG's last price (state[1])
        pso_agent.update(state[1])
        pso_price = pso_agent.choose_price()  # Continuous price
        
        # DDPG selects continuous action
        ddpg_state = state.astype(np.float32)
        ddpg_price, ddpg_norm = ddpg_agent.select_action(ddpg_state, explore=True)
        
        # Execute actions (both continuous prices)
        actions = [pso_price, ddpg_price]
        next_state, rewards, done, info = env.step(actions)
        
        # Update DDPG
        next_ddpg_state = next_state.astype(np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_norm, rewards[1], next_ddpg_state, done)
        ddpg_agent.replay()
        ddpg_agent.update_epsilon()
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_pso = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_pso = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Get benchmarks
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    
    # Calculate Delta (profit-based)
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_pso = (avg_price_pso - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_pso, avg_price_ddpg, delta_pso, delta_ddpg, rpdi_pso, rpdi_ddpg, p_n


def main():
    shock_cfg = {
        'enabled': False
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("PSO vs DDPG - SCHEME NONE")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_pso = []
        avg_prices_ddpg = []
        deltas_pso = []
        deltas_ddpg = []
        rpdis_pso = []
        rpdis_ddpg = []
        theo_prices = []
        
        for run in range(NUM_RUNS):
            seed = SEED + run
            ap_pso, ap_ddpg, d_pso, d_ddpg, r_pso, r_ddpg, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_pso.append(ap_pso)
            avg_prices_ddpg.append(ap_ddpg)
            deltas_pso.append(d_pso)
            deltas_ddpg.append(d_ddpg)
            rpdis_pso.append(r_pso)
            rpdis_ddpg.append(r_ddpg)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price PSO': np.mean(avg_prices_pso),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DDPG': np.mean(avg_prices_ddpg),
            'Delta PSO': np.mean(deltas_pso),
            'Delta DDPG': np.mean(deltas_ddpg),
            'RPDI PSO': np.mean(rpdis_pso),
            'RPDI DDPG': np.mean(rpdis_ddpg)
        }
        
        print(f"  Completed: PSO Δ = {results[model]['Delta PSO']:.3f}, DDPG Δ = {results[model]['Delta DDPG']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models],
        'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 2) for m in models],
        'PSO RPDI': [round(results[m]['RPDI PSO'], 2) for m in models],
        'DDPG RPDI': [round(results[m]['RPDI DDPG'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/pso_vs_ddpg.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print("=" * 80)
    
    avg_delta_pso = np.mean([results[m]['Delta PSO'] for m in models])
    avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
    avg_rpdi_pso = np.mean([results[m]['RPDI PSO'] for m in models])
    avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])
    
    print("\nPSO Agent:")
    print(f"  Average Delta (Δ): {avg_delta_pso:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_pso:.4f}")
    print("\nDDPG Agent:")
    print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")
    
    print("\n[Results saved to ./results/pso_vs_ddpg.csv]")


if __name__ == "__main__":
    main()
