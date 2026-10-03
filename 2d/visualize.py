from __future__ import division

from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from environment import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT


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


def render_frame(env, perception, action, reward, step_index, episode_label=None, q_values=None, info=None):
    food_field, poison_field = env.smell_fields()
    food_norm = _normalize(food_field)
    poison_norm = _normalize(poison_field)

    bg = np.full((env.height, env.width, 3), 0.08, dtype=np.float32)
    bg[..., 0] += 0.88 * poison_norm
    bg[..., 1] += 0.88 * food_norm
    bg[..., 2] += 0.16 * np.maximum(food_norm, poison_norm)
    bg = np.clip(bg, 0.0, 1.0)

    fig = plt.figure(figsize=(10.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.2])
    ax = fig.add_subplot(gs[0])
    overlay = fig.add_subplot(gs[1])

    ax.imshow(bg, interpolation="nearest", origin="upper")

    for y in range(env.height):
        for x in range(env.width):
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=(0.1, 0.1, 0.1, 0.25),
                    linewidth=1.0,
                )
            )

    fx, fy = env.food_pos
    px, py = env.poison_pos
    ax.scatter([fx], [fy], s=220, marker="s", color="#22c55e", edgecolors="#14532d", linewidths=1.6, zorder=4)
    ax.scatter([px], [py], s=220, marker="s", color="#ef4444", edgecolors="#7f1d1d", linewidths=1.6, zorder=4)

    ax_x, ax_y = env.agent_pos
    ax.scatter([ax_x], [ax_y], s=190, marker="o", color="#2563eb", edgecolors="#111827", linewidths=1.7, zorder=5)

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D GridWorld: food smell green, poison smell red")

    requested_action = None
    executed_action = action
    if info is not None:
        requested_action = info.get("action_requested")
        executed_action = info.get("action_executed", executed_action)

    def _action_name(value):
        if value == ACTION_UP:
            return "up"
        if value == ACTION_DOWN:
            return "down"
        if value == ACTION_LEFT:
            return "left"
        if value == ACTION_RIGHT:
            return "right"
        return "none"

    action_label = _action_name(executed_action)

    overlay.axis("off")
    text_lines = [
        "Episode: {0}".format(episode_label if episode_label is not None else "-"),
        "Step: {0}".format(step_index),
        "Perception (fx, fy, px, py): {0}".format(tuple(perception)),
        "Action: {0}".format(action_label),
        "Reward: {0:+.3f}".format(float(reward)),
    ]
    if info is not None and info.get("wall_bump"):
        text_lines.append("Wall bump: yes")
        if requested_action is not None and requested_action != executed_action:
            text_lines.append("Auto-switch: {0} -> {1}".format(_action_name(requested_action), action_label))
    if q_values is not None:
        text_lines.append(
            "Q(up, down, left, right): [{0:+.3f}, {1:+.3f}, {2:+.3f}, {3:+.3f}]".format(
                float(q_values[0]), float(q_values[1]), float(q_values[2]), float(q_values[3])
            )
        )
    overlay.text(
        0.01,
        0.58,
        "\n".join(text_lines),
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
    with imageio.get_writer(str(mp4_path), fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return mp4_path


def rollout_episode(agent, env, episode_label=None):
    obs = env.reset()
    frames = []
    total_reward = 0.0
    steps = 0
    success = False

    while True:
        action = agent.select_action(obs, epsilon=0.0)
        next_obs, reward, done, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        frames.append(
            render_frame(
                env=env,
                perception=obs,
                action=action,
                reward=reward,
                step_index=steps,
                episode_label=episode_label,
                q_values=agent.q_values(obs),
                info=info,
            )
        )
        obs = next_obs
        if done:
            success = info.get("terminal") == "food"
            break

    return {
        "frames": frames,
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
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

    fig.savefig(str(path), dpi=180)
    plt.close(fig)
    return path


def plot_q_table_heatmap(q_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reduce the 4D state by averaging over poison dirs for a readable plot.
    # Rows: food x-dir in {-1,0,1}, cols: food y-dir in {-1,0,1}; values are max Q over actions.
    matrix = np.zeros((3, 3), dtype=np.float32)
    for fx in range(3):
        for fy in range(3):
            state_slice = q_table[fx, fy, :, :, :]
            matrix[fx, fy] = np.mean(np.max(state_slice, axis=-1))

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["fy=-1", "fy=0", "fy=+1"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["fx=-1", "fx=0", "fx=+1"])
    ax.set_title("Q-table summary (food direction axes)")
    ax.set_xlabel("Food y direction")
    ax.set_ylabel("Food x direction")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "{0:+.2f}".format(matrix[i, j]), ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(str(path), dpi=180)
    plt.close(fig)
    return path
