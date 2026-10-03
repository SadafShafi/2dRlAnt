from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from environment import ACTION_LEFT, ACTION_RIGHT, LineWorld


DEFAULT_SMELL_LAMBDA = 1.6
DEFAULT_FPS = 6


def _figure_to_frame(fig):
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _normalize(values):
    peak = float(np.max(values))
    if peak <= 0.0:
        return values
    return values / peak


def _smell_strip(env, smell_lambda=DEFAULT_SMELL_LAMBDA, width=900):
    x = np.linspace(-0.5, env.length - 0.5, width)
    food_strength = np.exp(-np.abs(x - env.food_pos) / float(smell_lambda))
    poison_strength = np.exp(-np.abs(x - env.poison_pos) / float(smell_lambda))

    # Food should visibly read as green and poison as red.
    image = np.full((1, width, 3), 0.08, dtype=np.float32)
    image[..., 0] += 0.88 * _normalize(poison_strength)
    image[..., 1] += 0.88 * _normalize(food_strength)
    image[..., 2] += 0.18 * np.maximum(_normalize(food_strength), _normalize(poison_strength))
    image = np.clip(image, 0.0, 1.0)
    return image


def render_frame(
    env,
    perception,
    action,
    reward,
    step_index,
    episode_label=None,
    q_values=None,
    smell_lambda=DEFAULT_SMELL_LAMBDA,
):
    """Render a single LineWorld frame.

    The top panel shows a 1D row of cells with a continuous smell gradient.
    The bottom panel shows the per-step overlay using the agent's discretized
    perception, the chosen action, and the received reward.
    """

    background = _smell_strip(env, smell_lambda=smell_lambda)

    fig = plt.figure(figsize=(13, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.1])

    ax = fig.add_subplot(gs[0])
    overlay = fig.add_subplot(gs[1])

    ax.imshow(
        background,
        extent=[-0.5, env.length - 0.5, 0.0, 1.0],
        aspect="auto",
        interpolation="bilinear",
        origin="lower",
    )

    for cell in range(env.length):
        ax.add_patch(
            Rectangle(
                (cell - 0.5, 0.0),
                1.0,
                1.0,
                fill=False,
                edgecolor=(0.1, 0.1, 0.1, 0.28),
                linewidth=1.1,
            )
        )

    ax.scatter([env.food_pos], [0.5], s=280, marker="s", color="#22c55e", edgecolors="#14532d", linewidths=1.4, zorder=5)
    ax.scatter([env.poison_pos], [0.5], s=280, marker="s", color="#ef4444", edgecolors="#7f1d1d", linewidths=1.4, zorder=5)
    ax.scatter([env.agent_pos], [0.5], s=180, marker="o", color="#2563eb", edgecolors="#111827", linewidths=1.6, zorder=6)

    ax.set_xlim(-0.5, env.length - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(env.length))
    ax.set_yticks([])
    ax.set_xlabel("LineWorld cells")
    ax.set_title("Food smells green, poison smells red")

    action_label = "none"
    if action == ACTION_LEFT:
        action_label = "left"
    elif action == ACTION_RIGHT:
        action_label = "right"

    overlay.axis("off")
    overlay_text = [
        "Episode: {0}".format(episode_label if episode_label is not None else "-"),
        "Step: {0}".format(step_index),
        "Perception (food_dir, poison_dir): {0}".format(tuple(perception)),
        "Action: {0}".format(action_label),
        "Reward: {0:+.3f}".format(float(reward)),
    ]
    if q_values is not None:
        overlay_text.append("Q(left, right): [{0:+.3f}, {1:+.3f}]".format(float(q_values[0]), float(q_values[1])))

    overlay.text(
        0.01,
        0.58,
        "\n".join(overlay_text),
        ha="left",
        va="center",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f8fafc", "edgecolor": "#94a3b8", "boxstyle": "round,pad=0.5"},
    )

    return _figure_to_frame(fig)


def save_video(frames, base_path, fps=DEFAULT_FPS):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = base_path.with_suffix(".mp4")
    try:
        with imageio.get_writer(mp4_path, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:
        raise RuntimeError(
            "MP4 video writing failed. Install ffmpeg and imageio-ffmpeg, then rerun."
        ) from exc
    return mp4_path


def rollout_episode(agent, env, episode_label=None, smell_lambda=DEFAULT_SMELL_LAMBDA):
    """Run one greedy rollout and return all rendered frames plus stats."""

    observation = env.reset()
    frames = []
    total_reward = 0.0
    steps = 0
    success = False

    while True:
        action = agent.select_action(observation, epsilon=0.0)
        next_observation, reward, done, info = env.step(action)
        steps += 1
        total_reward += float(reward)

        frames.append(
            render_frame(
                env=env,
                perception=observation,
                action=action,
                reward=reward,
                step_index=steps,
                episode_label=episode_label,
                q_values=agent.q_values(observation),
                smell_lambda=smell_lambda,
            )
        )

        observation = next_observation
        if done:
            success = info.get("terminal") == "food"
            break

    return {
        "frames": frames,
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
    }


def record_episode(agent, env, filename, fps=DEFAULT_FPS, smell_lambda=DEFAULT_SMELL_LAMBDA):
    """Run one greedy rollout and save the complete episode as a video."""

    episode_label = Path(filename).name
    result = rollout_episode(agent, env, episode_label=episode_label, smell_lambda=smell_lambda)
    frames = result["frames"]
    video_path = save_video(frames, filename, fps=fps)
    return {
        "video_path": str(video_path),
        "steps": result["steps"],
        "total_reward": result["total_reward"],
        "success": result["success"],
    }


def plot_learning_curves(rewards, successes, path, rolling_window):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    episode_ids = np.arange(1, len(rewards) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    axes[0].plot(episode_ids, rewards, color="#1d4ed8", alpha=0.35, linewidth=1.2, label="Episode reward")
    if len(rewards) >= rolling_window:
        reward_avg = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode="valid")
        axes[0].plot(episode_ids[rolling_window - 1 :], reward_avg, color="#1d4ed8", linewidth=2.2, label="{0}-episode mean".format(rolling_window))
    axes[0].set_title("Learning curve: reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].plot(episode_ids, successes, color="#059669", alpha=0.35, linewidth=1.2, label="Success (0/1)")
    if len(successes) >= rolling_window:
        success_avg = np.convolve(successes, np.ones(rolling_window) / rolling_window, mode="valid")
        axes[1].plot(episode_ids[rolling_window - 1 :], success_avg, color="#059669", linewidth=2.2, label="{0}-episode success rate".format(rolling_window))
    axes[1].set_title("Learning curve: rolling success rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_q_table_heatmap(q_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = ["(-1, -1)", "(-1, +1)", "(+1, -1)", "(+1, +1)"]
    actions = ["left", "right"]
    matrix = np.array(
        [
            q_table[0, 0],
            q_table[0, 1],
            q_table[1, 0],
            q_table[1, 1],
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(actions)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Q-table heatmap")
    ax.set_xlabel("Action")
    ax.set_ylabel("Perception state")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "{0:+.2f}".format(matrix[i, j]), ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
