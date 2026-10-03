from pathlib import Path

import numpy as np

from agent import QLearningAgent, QLearningConfig
from environment import LineWorld, LineWorldConfig
from visualize import rollout_episode, save_video


class TestConfig:
    def __init__(
        self,
        episodes=100,
        video_fps=3,
        output_dir="outputs",
        env=None,
        agent=None,
    ):
        self.episodes = episodes
        self.video_fps = video_fps
        self.output_dir = output_dir
        self.env = env if env is not None else LineWorldConfig(seed=123)
        self.agent = agent if agent is not None else QLearningConfig(seed=123)


def evaluate(q_table_path, config=None):
    config = config or TestConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = QLearningAgent.load(q_table_path, config.agent)
    env = LineWorld(config.env)

    rewards = []
    successes = []
    steps_taken = []
    test_reel_frames = []

    for episode in range(1, config.episodes + 1):
        result = rollout_episode(agent, env, episode_label="test_ep{0:02d}".format(episode))
        test_reel_frames.extend(result["frames"])
        rewards.append(result["total_reward"])
        successes.append(1 if result["success"] else 0)
        steps_taken.append(result["steps"])

    test_video_path = save_video(test_reel_frames, output_dir / "test_reel", fps=config.video_fps)

    success_rate = float(np.mean(successes))
    avg_steps = float(np.mean(steps_taken))
    avg_reward = float(np.mean(rewards))

    print("Evaluation over {0} episodes".format(config.episodes))
    print(f"Success rate: {success_rate:.3f}")
    print(f"Average steps: {avg_steps:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Saved combined test video to {test_video_path}")

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "test_video_path": str(test_video_path),
    }


def main():
    evaluate(Path("outputs/q_table.npy"))


if __name__ == "__main__":
    main()
