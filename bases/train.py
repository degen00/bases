from .game import Bases
from .agent import QLearningAgent
from .agent import RandomAgent
import bases
import random
import csv
import numpy as np


def train_agents(grid_size: int, episodes: int) -> None:
    agent_a = QLearningAgent(grid_size)
    agent_b = QLearningAgent(grid_size)

    rewards = {'A': 1, 'B': -1, 'Tie': 0}

    game = Bases(grid_size)
    for episode in range(episodes):
        winner = game.play(agent_a, agent_b)

        reward_a = rewards[winner]
        reward_b = -reward_a

        state = game.get_detailed_state()
        for action in game.lines:
            completed_box, potential_box_opponent = \
                game.make_move(action['x1'], action['y1'],
                               action['x2'], action['y2'])[1:]
            specific_reward_a = \
                agent_a.enhance_reward(completed_box, potential_box_opponent)
            specific_reward_b = \
                agent_b.enhance_reward(completed_box, potential_box_opponent)
            next_state = game.get_detailed_state()
            available_moves = game.available_moves()
            agent_a.update(state, (action['x1'], action['y1'],
                                   action['x2'], action['y2']),
                           reward_a + specific_reward_a,
                           next_state, available_moves)
            agent_b.update(state, (action['x1'], action['y1'],
                                   action['x2'], action['y2']),
                           reward_b + specific_reward_b,
                           next_state, available_moves)
            state = next_state

    if game.wins['A'] >= game.wins['B']:
        agent_a.save_policy(bases.cfg['policy_path'])
        print("Saving AI Agent A's Policy")
    else:
        agent_b.save_policy(bases.cfg['policy_path'])
        print("Saving AI Agent B's Policy")


def evaluate_hyperparameters(learning_rate: float, discount_factor: float,
                             exploration_rate: float, grid_size: int,
                             train_episodes: int, test_episodes: int) -> float:
    agent = QLearningAgent(grid_size, learning_rate, discount_factor,
                           exploration_rate, training_mode=True)
    opponent = RandomAgent()
    game = Bases(grid_size)

    # Training phase
    for _ in range(train_episodes):
        game.play(agent, opponent)

    # Evaluation phase
    agent.training_mode = False
    wins = 0
    for _ in range(test_episodes):
        result = game.play(agent, opponent)
        if result == 'A':
            wins += 1

    win_rate = wins / test_episodes
    return win_rate


def hyperparameter_tuning(grid_size,
                          iterations,
                          train_episodes,
                          test_episodes) -> dict[str, float]:
    """Incomplete and poor implementation. """
    best_params = None
    best_score = -np.inf
    results = []

    for i in range(iterations):
        learning_rate = random.uniform(0.01, 1.0)
        discount_factor = random.uniform(0.5, 0.99)
        exploration_rate = random.uniform(0.01, 0.3)

        print(f"Testing combination {i+1}/{iterations}: \
              lr={learning_rate:.4f}, \
              df={discount_factor:.4f}, \
              er={exploration_rate:.4f}")

        score = evaluate_hyperparameters(
            learning_rate, discount_factor, exploration_rate,
            grid_size, train_episodes, test_episodes
        )

        print(f"Winning Rate: {score:.4f}")

        results.append({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'exploration_rate': exploration_rate,
            'win_rate': score
        })

        if score > best_score:
            best_score = score
            best_params = {
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'exploration_rate': exploration_rate
            }

    print("\n----- Best Hyperparameters Found -----")
    print(f"Learning Rate: {best_params['learning_rate']:.4f}")
    print(f"Discount Factor: {best_params['discount_factor']:.4f}")
    print(f"Exploration Rate: {best_params['exploration_rate']:.4f}")
    print(f"Winning Rate: {best_score:.4f}")

    with open(bases.cfg['hyperparameter_tuning_results'],
              'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['learning_rate',
                                               'discount_factor',
                                               'exploration_rate',
                                               'win_rate'])
        writer.writeheader()
        writer.writerows(results)

    return best_params
