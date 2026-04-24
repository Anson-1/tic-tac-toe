import os
import torch
import argparse
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episodes_vectorized, build_buffer
from super_tictactoe.ppo import ppo_update


def train(
    num_updates: int = 1000,
    episodes_per_update: int = 512,
    save_every: int = 100,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints',
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for update in range(1, num_updates + 1):
        print(f"Update {update:4d}/{num_updates} collecting...", end='\r', flush=True)
        episodes = collect_episodes_vectorized(episodes_per_update, model, device)
        buffer = build_buffer(episodes)
        losses = ppo_update(model, optimizer, buffer)

        wins = sum(1 for ep in episodes if ep[-1]['reward'] == 1.0)
        print(
            f"Update {update:4d}/{num_updates} | "
            f"actor={losses['actor_loss']:.4f} | "
            f"critic={losses['critic_loss']:.4f} | "
            f"steps={len(buffer['states'])} | "
            f"win%={wins/len(episodes):.0%}"
        )

        if update % save_every == 0:
            path = os.path.join(checkpoint_dir, f"model_{update:04d}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train(
        num_updates=args.updates,
        episodes_per_update=args.episodes,
        save_every=args.save_every,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
