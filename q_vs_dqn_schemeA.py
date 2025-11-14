"""
q_vs_dqn_schemeA.py
"""
"""
Q-Learning vs DQN with Scheme A Shocks
Scheme A: ρ=0.3, σ_η=0.5 (low persistence, high variance)
"""

import numpy as np
import pandas as pd
from environments import MarketEnvContinuous
from agents import QLearningAgent, DQNAgent  # Assuming DQNAgent is added to agents or adjust import
from theoretical_benchmarks import TheoreticalBenchmarks

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run Q vs DQN simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    dqn_agent = DQNAgent(
        agent_id=1,
        state_dim=2,
        action_dim=env.N,
        loss_type='huber',
        use_double=True,
        seed=seed
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
    
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    p_n = benchmarks['p_N']
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_dqn, delta_q, delta_dqn, p_n


def main():
    shock_cfg = {
        'enabled': True,
        'scheme': 'A',
        'mode': 'independent'
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("Q-LEARNING vs DQN - SCHEME A")
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
        avg_prices_dqn = []
        deltas_q = []
        deltas_dqn = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apq, apdqn, dq, ddqn, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_q.append(apq)
            avg_prices_dqn.append(apdqn)
            deltas_q.append(dq)
            deltas_dqn.append(ddqn)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price Q': np.mean(avg_prices_q),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DQN': np.mean(avg_prices_dqn),
            'Delta Q': np.mean(deltas_q),
            'Delta DQN': np.mean(deltas_dqn)
        }
        
        print(f"  Completed: Q Δ = {results[model]['Delta Q']:.3f}, DQN Δ = {results[model]['Delta DQN']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
        'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/q_vs_dqn_schemeA.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/q_vs_dqn_schemeA.csv]")


if __name__ == "__main__":
    main()
