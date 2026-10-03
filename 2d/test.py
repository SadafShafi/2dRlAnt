from __future__ import division

from pathlib import Path

import numpy as np

from agent import QLearning2DAgent, QLearning2DConfig
from environment import GridWorld2D, GridWorld2DConfig
from visualize import rollout_episode, save_video


class TestConfig:
    def __init__(
        self,
        episodes=100,
        video_fps=6,
        output_dir="outputs",
        env=None,
        agent=None,
    ):
        self.episodes = int(episodes)
        self.video_fps = int(video_fps)
        self.output_dir = output_dir
        self.env = env if env is not None else GridWorld2DConfig(seed=123)
        self.agent = agent if agent is not None else QLearning2DConfig(seed=123)


def evaluate(q_table_path, config=None):
    config = config if config is not None else TestConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = QLearning2DAgent.load(q_table_path, config.agent)
    env = GridWorld2D(config.env)

    rewards = []
    successes = []
    steps_taken = []
    test_reel_frames = []

    for episode in range(1, config.episodes + 1):
        result = rollout_episode(agent, env, episode_label="test_ep{0:03d}".format(episode))
        test_reel_frames.extend(result["frames"])
        rewards.append(result["total_reward"])
        successes.append(1 if result["success"] else 0)
        steps_taken.append(result["steps"])

    test_video_path = save_video(test_reel_frames, output_dir / "test_reel", fps=config.video_fps)
    success_rate = float(np.mean(successes))
    avg_steps = float(np.mean(steps_taken))
    avg_reward = float(np.mean(rewards))

    print("Evaluation over {0} episodes".format(config.episodes))
    print("Success rate: {0:.3f}".format(success_rate))
    print("Average steps: {0:.3f}".format(avg_steps))
    print("Average reward: {0:.3f}".format(avg_reward))
    print("Saved combined test video to {0}".format(test_video_path))

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
