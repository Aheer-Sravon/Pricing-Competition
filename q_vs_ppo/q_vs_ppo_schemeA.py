"""
q_vs_ppo_schemeA.py
"""
"""
Q-Learning vs PPO with Scheme A Shocks
Scheme A: ρ=0.3, σ_η=0.5 (low persistence, high variance)
"""

import numpy as np
import pandas as pd
from environments import MarketEnvContinuous
from agents import QLearningAgent, PPOAgent
from theoretical_benchmarks import TheoreticalBenchmarks

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q vs PPO simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    ppo_agent = PPOAgent(agent_id=1, price_min=0.0, price_max=2.0, seed=seed)
    
    state = env.reset()
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]
    state_ppo = (current_prices[1], current_prices[0])  # For agent_id=1: own=1, comp=0
    
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        ppo_price, log_prob, value = ppo_agent.select_action(state_ppo, explore=True)
        
        actions = [q_action, ppo_price]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        ppo_agent.store_transition(state_ppo, ppo_price, rewards[1], log_prob, value, done)
        
        state = next_state
        current_prices = info['prices']
        state_ppo = (current_prices[1], current_prices[0])
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_ppo = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_ppo = np.mean(last_profits[:, 1])
    
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ppo = (avg_profit_ppo - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_ppo, delta_q, delta_ppo, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'A',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs PPO - SCHEME A")
    print("Scheme A: ρ=0.3, σ_η=0.5 (low persistence, high variance)")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 50
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_q = []
        avg_prices_ppo = []
        deltas_q = []
        deltas_ppo = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apq, app, dq, dp, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_q.append(apq)
            avg_prices_ppo.append(app)
            deltas_q.append(dq)
            deltas_ppo.append(dp)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Theo Price': np.mean(theo_prices),
            'Avg Price PPO': np.mean(avg_prices_ppo),
            'Delta Q': np.mean(deltas_q),
            'Delta PPO': np.mean(deltas_ppo)
        }
        
        print(f"  Completed: Q Δ = {results[model]['Delta Q']:.3f}, PPO Δ = {results[model]['Delta PPO']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'PPO Avg. Prices': [round(results[m]['Avg Price PPO'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'PPO Extra-profits Δ': [round(results[m]['Delta PPO'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/q_vs_ppo_schemeA.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/q_vs_ppo_schemeA.csv]")


if __name__ == "__main__":
    main()
