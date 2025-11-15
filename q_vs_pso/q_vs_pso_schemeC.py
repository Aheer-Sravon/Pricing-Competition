"""
Q-Learning vs PSO with Scheme C Shocks
Scheme C: ρ=0.9, σ_η=0.3 (high persistence, medium variance)
"""

import numpy as np
import pandas as pd
from environments import MarketEnvContinuous
from agents import QLearningAgent, PSOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q vs PSO simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    pso_agent = PSOAgent(env, agent_id=1)
    
    state = env.reset()
    profits_history = []
    prices_history = []
    
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        pso_agent.update(current_prices[0])
        pso_price = pso_agent.choose_price()
        
        actions = [q_action, pso_price]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        current_prices = info['prices']
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_pso = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_pso = np.mean(last_profits[:, 1])
    
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_pso, delta_q, delta_pso, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'C',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs PSO - SCHEME C")
    print("Scheme C: ρ=0.9, σ_η=0.3 (high persistence, medium variance)")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 50
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_q = []
        avg_prices_pso = []
        deltas_q = []
        deltas_pso = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apq, appso, dq, dpso, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_q.append(apq)
            avg_prices_pso.append(appso)
            deltas_q.append(dq)
            deltas_pso.append(dpso)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Theo Price': np.mean(theo_prices),
            'Avg Price PSO': np.mean(avg_prices_pso),
            'Delta Q': np.mean(deltas_q),
            'Delta PSO': np.mean(deltas_pso)
        }
        
        print(f"  Completed: Q Δ = {results[model]['Delta Q']:.3f}, PSO Δ = {results[model]['Delta PSO']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/q_vs_pso_schemeC.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/q_vs_pso_schemeC.csv]")


if __name__ == "__main__":
    main()
