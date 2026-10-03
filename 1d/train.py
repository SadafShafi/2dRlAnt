from pathlib import Path

import numpy as np

from agent import QLearningAgent, QLearningConfig
from environment import LineWorld, LineWorldConfig
from visualize import plot_learning_curves, plot_q_table_heatmap, rollout_episode, save_video


class TrainConfig:
    def __init__(
        self,
        episodes=1000,
        snapshot_interval=100,
        rolling_window=50,
        video_fps=3,
        seed=7,
        output_dir="outputs",
        env=None,
        agent=None,
    ):
        self.episodes = episodes
        self.snapshot_interval = snapshot_interval
        self.rolling_window = rolling_window
        self.video_fps = video_fps
        self.seed = seed
        self.output_dir = output_dir
        self.env = env if env is not None else LineWorldConfig()
        self.agent = agent if agent is not None else QLearningConfig(seed=seed)


def run_episode(
    env: LineWorld,
    agent: QLearningAgent,
    epsilon: float,
    learn: bool,
    episode_label: str,
    max_frames=None,
):
    observation = env.reset()
    trajectory = []
    total_reward = 0.0
    success = False

    while True:
        action = agent.select_action(observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        if learn:
            agent.update(observation, action, reward, next_observation, done)

        trajectory.append(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
                "info": info,
            }
        )
        total_reward += reward
        observation = next_observation

        if done or (max_frames is not None and len(trajectory) >= max_frames):
            success = info.get("terminal") == "food"
            break

    return trajectory, total_reward, success, int(info.get("steps", len(trajectory)))


def _evaluate_and_save_snapshot(
    agent: QLearningAgent,
    config: TrainConfig,
    episode_number: int,
    output_dir: Path,
):
    env = LineWorld(config.env)
    result = rollout_episode(agent, env, episode_label=f"train_ep{episode_number:04d}")
    return result["frames"], result


def train(config=None):
    config = config or TrainConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = LineWorld(config.env)
    agent = QLearningAgent(config.agent)

    rewards = []
    successes = []
    epsilon = config.agent.epsilon_start
    snapshot_paths = []
    training_reel_frames = []

    for episode in range(1, config.episodes + 1):
        trajectory, total_reward, success, steps = run_episode(
            env=env,
            agent=agent,
            epsilon=epsilon,
            learn=True,
            episode_label=str(episode),
        )
        rewards.append(total_reward)
        successes.append(1 if success else 0)
        epsilon = agent.decay_epsilon(epsilon)

        if episode % 50 == 0 or episode == 1:
            rolling = np.mean(successes[-config.rolling_window :]) if successes else 0.0
            print(
                f"Episode {episode:04d}/{config.episodes} | reward={total_reward:+.3f} | "
                f"steps={steps:02d} | success={int(success)} | epsilon={epsilon:.3f} | rolling_success={rolling:.3f}"
            )

        if episode % config.snapshot_interval == 0:
            frames, _ = _evaluate_and_save_snapshot(agent, config, episode, output_dir)
            training_reel_frames.extend(frames)
            snapshot_paths.append(str(output_dir / f"train_ep{episode:04d}"))
            print(f"Captured training snapshot rollout for episode {episode:04d}")

    q_table_path = output_dir / "q_table.npy"
    agent.save(q_table_path)

    training_video_path = save_video(training_reel_frames, output_dir / "train_reel", fps=config.video_fps)

    learning_curve_path = plot_learning_curves(
        rewards=rewards,
        successes=successes,
        path=output_dir / "learning_curve.png",
        rolling_window=config.rolling_window,
    )
    heatmap_path = plot_q_table_heatmap(agent.q_table, output_dir / "q_table_heatmap.png")

    print(f"Saved Q-table to {q_table_path}")
    print(f"Saved learning curve to {learning_curve_path}")
    print(f"Saved Q-table heatmap to {heatmap_path}")
    print(f"Saved combined training video to {training_video_path}")

    return {
        "q_table_path": str(q_table_path),
        "learning_curve_path": str(learning_curve_path),
        "heatmap_path": str(heatmap_path),
        "training_video_path": str(training_video_path),
        "rewards": rewards,
        "successes": successes,
        "snapshot_paths": snapshot_paths,
        "agent": agent,
    }


def main():
    train()


if __name__ == "__main__":
    main()
