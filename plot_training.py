"""Generate training curve from a saved checkpoint."""

import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
parser.add_argument("--out", default=None, help="Output image path (default: same dir as checkpoint)")
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")

history_updates = ckpt.get("history_updates", [])
history_win_rate = ckpt.get("history_win_rate", [])
history_actor_loss = ckpt.get("history_actor_loss", [])
history_critic_loss = ckpt.get("history_critic_loss", [])
history_eval_updates = ckpt.get("history_eval_updates", [])
history_eval_win_rate = ckpt.get("history_eval_win_rate", [])
phase = ckpt.get("phase", 0)
update = ckpt.get("update", 0)

CURRICULUM_NAMES = {0: "random", 1: "greedy", 2: "blocking"}
opponent = CURRICULUM_NAMES.get(phase, f"phase{phase}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(history_updates, history_win_rate, label=f"Win rate (vs {opponent})", color="tab:blue", alpha=0.6)
if history_eval_win_rate:
    axes[0].plot(history_eval_updates, history_eval_win_rate, "o-", label="Eval win rate", color="tab:orange")
axes[0].set_ylabel("Win Rate")
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].set_title(f"Training curve — phase {phase+1} (vs {opponent}) — update {update}")
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_updates, history_actor_loss, color="tab:red")
axes[1].set_ylabel("Actor Loss")
axes[1].grid(True, alpha=0.3)

axes[2].plot(history_updates, history_critic_loss, color="tab:green")
axes[2].set_ylabel("Critic Loss")
axes[2].set_xlabel("Update")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

if args.out:
    out_path = args.out
else:
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_path = os.path.join(ckpt_dir, "training_curve.png")

fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved to: {out_path}")
