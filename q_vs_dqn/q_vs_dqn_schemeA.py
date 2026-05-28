import os
import numpy as np
import pandas as pd

from environments import MarketEnv
from agents import QLearningAgent, DQNAgent
from theoretical_benchmarks import TheoreticalBenchmarks

import argparse
parser = argparse.ArgumentParser(prog="q_vs_q")
parser.add_argument("-r", "--num_runs", type=int, nargs=1, help="Number of batches per model")
args = parser.parse_args()

NUM_RUNS = args.num_runs[0] if args.num_runs is not None else 50

def run_simulation(model, shock_cfg, benchmarks):
    """Run a single simulation of Q-Learning vs DQN."""
    env = MarketEnv(market_model=model, shock_cfg=shock_cfg)
    
    q_agent = QLearningAgent(
        env.N,
        agent_id=0,
        price_grid=env.price_grid
    )
    
    dqn_agent = DQNAgent(
        agent_id=1, 
        state_dim=2,
        action_dim=env.N,
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
        'scheme': 'A',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks()
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    print("=" * 80)
    print("Q-LEARNING vs DQN - SCHEME A")
    print("=" * 80)
    
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
        
        for run in range(NUM_RUNS):
            apq, apd, dq, dd, rq, rd, p_n = run_simulation(model, shock_cfg, model_benchmarks)
            
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

            per_run_metrices[model]["run"].append(run+1)
            per_run_metrices[model]["avg_price_firm_1"].append(round(apq, 2))
            per_run_metrices[model]["avg_price_firm_2"].append(round(apd, 2))
        
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

    os.makedirs("./results/shock_a", exist_ok=True)

    for model in models:
        metrices_df = pd.DataFrame(per_run_metrices[model])
        metrices_df["rolling_avg_firm_1"] = metrices_df["avg_price_firm_1"].rolling(window=3).mean().round(2)
        metrices_df["rolling_avg_firm_2"] = metrices_df["avg_price_firm_2"].rolling(window=3).mean().round(2)

        metrices_df.to_csv(f"./results/shock_a/per_round_metrices_{model}.csv", index=False)
    
    df = pd.DataFrame(data)
    df.to_csv("./results/shock_a/q_vs_dqn_schemeA.csv", index=False)

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
    print("\n[Results saved to ./results/shock_a/q_vs_dqn_schemeA.csv]")

main()
