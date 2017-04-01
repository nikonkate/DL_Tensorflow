#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_discrete
import numpy as np
import random
import operator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()


    # returns Q-value/avg rewards for each action given a state
    def qsv(state, av_table):
        zero = av_table[(state, 0)]
        one = av_table[(state, 1)]
        return np.array([zero, one])

    # Create the environment
    env = environment_discrete.EnvironmentDiscrete(args.env)

    # Create Q, C and other variables
    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions])

    epsilon = args.epsilon
    episode_rewards, episode_lengths = [], []

    DISCOUNT = 0.99

    for episode in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards, total_reward = [], [], [], 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            # compute action using epsilon-greedy policy
            actions_values = qsv(state, Q)
            if np.random.random() > epsilon and actions:
                # Exploit
                if(actions_values[0] > actions_values[1]):
                    action = 0
                else:
                    action = 1
            else:
                # Explore
                action = random.choice([1, 0])

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break

        # update C
        for s, a, r in zip(states, actions, rewards):
            if a == 0 :
                C[s][0] += 1
            else:
                C[s][1] += 1

        # Update Q by averaging action-state value with some discount
        for s, a in zip(states, actions):
            if a == 0:
                Q[s][0] = Q[s][0] + (1 / C[s][0]) * ((1.0 - 0.05 * epsilon) * total_reward - Q[s][0])
            else:
                Q[s][1] = Q[s][1] + (1 / C[s][1]) * ((1.0 - 0.05 * epsilon) * total_reward - Q[s][1])

        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon))

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

