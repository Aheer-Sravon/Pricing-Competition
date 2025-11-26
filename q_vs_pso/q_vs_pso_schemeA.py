import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, PSOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99
NUM_RUNS = 50


def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q-Learning vs PSO simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize Q-Learning agent (agent 0)
    q_agent = QLearningAgent(env.N, agent_id=0, price_grid=env.price_grid)
    
    # Initialize PSO agent (agent 1)
    pso_agent = PSOAgent(env, agent_id=1, price_min=price_min, price_max=price_max)
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        # Q-Learning selects discrete action index
        q_action = q_agent.choose_action(state)  # int index
        
        # PSO updates with Q's last price (state[0])
        pso_agent.update(state[0])
        pso_price = pso_agent.choose_price()  # float
        
        # Execute actions (discrete index + continuous price)
        actions = [q_action, pso_price]
        next_state, rewards, done, info = env.step(actions)
        
        # Update Q-Learning
        q_agent.update(state, q_action, rewards[0], next_state)
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_pso = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_pso = np.mean(last_profits[:, 1])
    
    # Get benchmarks
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    
    # Calculate Delta (profit-based)
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_q = (avg_price_q - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_pso = (avg_price_pso - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_q, avg_price_pso, delta_q, delta_pso, rpdi_q, rpdi_pso, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'A',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs PSO - SCHEME A")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_q = []
        avg_prices_pso = []
        deltas_q = []
        deltas_pso = []
        rpdis_q = []
        rpdis_pso = []
        theo_prices = []
        
        for run in range(NUM_RUNS):
            seed = SEED + run
            ap_q, ap_pso, d_q, d_pso, r_q, r_pso, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_q.append(ap_q)
            avg_prices_pso.append(ap_pso)
            deltas_q.append(d_q)
            deltas_pso.append(d_pso)
            rpdis_q.append(r_q)
            rpdis_pso.append(r_pso)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Theo Price': np.mean(theo_prices),
            'Avg Price PSO': np.mean(avg_prices_pso),
            'Delta Q': np.mean(deltas_q),
            'Delta PSO': np.mean(deltas_pso),
            'RPDI Q': np.mean(rpdis_q),
            'RPDI PSO': np.mean(rpdis_pso)
        }
        
        print(f"  Completed: Q Δ = {results[model]['Delta Q']:.3f}, PSO Δ = {results[model]['Delta PSO']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models],
        'Q RPDI': [round(results[m]['RPDI Q'], 2) for m in models],
        'PSO RPDI': [round(results[m]['RPDI PSO'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/q_vs_pso_schemeA.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Overall averages
    print("\n" + "=" * 80)
    print("OVERALL AVERAGES ACROSS ALL MODELS")
    print("=" * 80)
    
    avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
    avg_delta_pso = np.mean([results[m]['Delta PSO'] for m in models])
    avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
    avg_rpdi_pso = np.mean([results[m]['RPDI PSO'] for m in models])
    
    print("\nQ-Learning:")
    print(f"  Average Delta (Δ): {avg_delta_q:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_q:.4f}")
    print("\nPSO:")
    print(f"  Average Delta (Δ): {avg_delta_pso:.4f}")
    print(f"  Average RPDI:      {avg_rpdi_pso:.4f}")
    
    print("\n[Results saved to ./results/q_vs_pso_schemeA.csv]")


if __name__ == "__main__":
    main()
