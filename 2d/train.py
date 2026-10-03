from __future__ import division

from pathlib import Path

import numpy as np

from agent import QLearning2DAgent, QLearning2DConfig
from environment import GridWorld2D, GridWorld2DConfig
from visualize import plot_learning_curves, plot_q_table_heatmap, rollout_episode, save_video


class TrainConfig:
    def __init__(
        self,
        episodes=1000,
        snapshot_interval=100,
        rolling_window=50,
        video_fps=6,
        seed=7,
        output_dir="outputs",
        env=None,
        agent=None,
    ):
        self.episodes = int(episodes)
        self.snapshot_interval = int(snapshot_interval)
        self.rolling_window = int(rolling_window)
        self.video_fps = int(video_fps)
        self.seed = seed
        self.output_dir = output_dir
        self.env = env if env is not None else GridWorld2DConfig(seed=seed)
        self.agent = agent if agent is not None else QLearning2DConfig(seed=seed)


def run_episode(env, agent, epsilon, learn):
    obs = env.reset()
    total_reward = 0.0
    success = False
    steps = 0

    while True:
        requested_action = agent.select_action(obs, epsilon)
        next_obs, reward, done, info = env.step(requested_action)
        executed_action = int(info.get("action_executed", requested_action))
        if learn:
            agent.update(obs, executed_action, reward, next_obs, done)
        total_reward += float(reward)
        steps = int(info.get("steps", steps + 1))
        obs = next_obs
        if done:
            success = info.get("terminal") == "food"
            break

    return total_reward, success, steps


def train(config=None):
    config = config if config is not None else TrainConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = GridWorld2D(config.env)
    agent = QLearning2DAgent(config.agent)

    rewards = []
    successes = []
    epsilon = config.agent.epsilon_start
    training_reel_frames = []

    for episode in range(1, config.episodes + 1):
        total_reward, success, steps = run_episode(env, agent, epsilon, learn=True)
        rewards.append(total_reward)
        successes.append(1 if success else 0)
        epsilon = agent.decay_epsilon(epsilon)

        if episode % 50 == 0 or episode == 1:
            rolling = np.mean(successes[-config.rolling_window :]) if successes else 0.0
            print(
                "Episode {0:04d}/{1} | reward={2:+.3f} | steps={3:02d} | success={4} | epsilon={5:.3f} | rolling_success={6:.3f}".format(
                    episode,
                    config.episodes,
                    total_reward,
                    steps,
                    int(success),
                    epsilon,
                    rolling,
                )
            )

        if episode % config.snapshot_interval == 0:
            snap_env = GridWorld2D(config.env)
            snap = rollout_episode(agent, snap_env, episode_label="train_ep{0:04d}".format(episode))
            training_reel_frames.extend(snap["frames"])
            print("Captured training snapshot rollout for episode {0:04d}".format(episode))

    q_table_path = output_dir / "q_table.npy"
    agent.save(q_table_path)

    training_video_path = save_video(training_reel_frames, output_dir / "train_reel", fps=config.video_fps)
    learning_curve_path = plot_learning_curves(rewards, successes, output_dir / "learning_curve.png", config.rolling_window)
    heatmap_path = plot_q_table_heatmap(agent.q_table, output_dir / "q_table_heatmap.png")

    print("Saved Q-table to {0}".format(q_table_path))
    print("Saved learning curve to {0}".format(learning_curve_path))
    print("Saved Q-table heatmap to {0}".format(heatmap_path))
    print("Saved combined training video to {0}".format(training_video_path))

    return {
        "q_table_path": str(q_table_path),
        "learning_curve_path": str(learning_curve_path),
        "heatmap_path": str(heatmap_path),
        "training_video_path": str(training_video_path),
        "rewards": rewards,
        "successes": successes,
        "agent": agent,
    }


def main():
    train()


if __name__ == "__main__":
    main()
