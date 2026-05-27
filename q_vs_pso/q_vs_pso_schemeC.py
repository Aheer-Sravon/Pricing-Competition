import os
import numpy as np
import pandas as pd

from environments import MarketEnv
from agents import QLearningAgent, PSOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

import argparse
parser = argparse.ArgumentParser(prog="q_vs_q")
parser.add_argument("-s", "--seed", type=int, nargs=1, help="Specify the seed")
parser.add_argument("-r", "--num_runs", type=int, nargs=1, help="Number of batches per model")
args = parser.parse_args()

SEED = args.seed[0] if args.seed is not None else 99
NUM_RUNS = args.num_runs[0] if args.num_runs is not None else 50

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q-Learning vs PSO simulation"""
    np.random.seed(seed)
    
    env = MarketEnv(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize Q-Learning agent (agent 0)
    q_agent = QLearningAgent(env.N, agent_id=0, price_grid=env.price_grid)
    
    # Initialize PSO agent (agent 1)
    pso_agent = PSOAgent(env, agent_id=1, price_min=price_min, price_max=price_max)
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for _ in range(env.horizon):
        # Q-Learning selects discrete action index
        q_action = q_agent.choose_action(state)  # int index
        
        # PSO updates with Q's last price (state[0])
        pso_agent.update(state[0])
        pso_price = pso_agent.choose_price()  # float
        # choose action closest to the price chosen by PSO
        pso_action = int(np.abs(env.price_grid - pso_price).argmin())
        
        # Execute actions (discrete index + continuous price)
        actions = [q_action, pso_action]
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
        'scheme': 'C',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs PSO - SCHEME C")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    results = {}

    per_run_metrices = {
            "logit": {
                "run": [],
                "avg_price_firm_1": [],
                "avg_price_firm_2": [],
            },
            "hotelling": {
                "run": [],
                "avg_price_firm_1": [],
                "avg_price_firm_2": [],
            },
            "linear": {
                "run": [],
                "avg_price_firm_1": [],
                "avg_price_firm_2": [],
            }
    }
    
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

    os.makedirs("./results/shock_c", exist_ok=True)

    for model in models:
        metrices_df = pd.DataFrame(per_run_metrices[model])
        metrices_df["rolling_avg_firm_1"] = metrices_df["avg_price_firm_1"].rolling(window=3).mean().round(2)
        metrices_df["rolling_avg_firm_2"] = metrices_df["avg_price_firm_2"].rolling(window=3).mean().round(2)

        metrices_df.to_csv(f"./results/shock_c/per_round_metrices_{model}.csv", index=False)
    
    df = pd.DataFrame(data)
    df.to_csv("./results/shock_c/q_vs_pso_schemeC.csv", index=False)
    
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
    
    print("\n[Results saved to ./results/shock_c/q_vs_pso_schemeC.csv]")

main()
