import os
import torch
import argparse
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episodes_vectorized, build_buffer
from super_tictactoe.ppo import ppo_update
from super_tictactoe.evaluate import evaluate


def train(
    num_updates: int = 3000,
    episodes_per_update: int = 512,
    save_every: int = 100,
    eval_every: int = 100,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints',
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = ActorCritic().to(device)
    base_lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # Fixed reference model for best-checkpoint evaluation (random init, never updated)
    reference = ActorCritic()
    best_vs_reference = 0.0
    best_path = os.path.join(checkpoint_dir, 'model_best.pt')

    for update in range(1, num_updates + 1):
        # Linear learning rate decay: 3e-4 → 3e-5
        lr = base_lr * (1 - 0.9 * update / num_updates)
        for g in optimizer.param_groups:
            g['lr'] = lr

        print(f"Update {update:4d}/{num_updates} collecting...", end='\r', flush=True)
        episodes = collect_episodes_vectorized(episodes_per_update, model, device)
        buffer = build_buffer(episodes)
        losses = ppo_update(model, optimizer, buffer)

        wins = sum(1 for ep in episodes if ep[-1]['reward'] > 0.5)
        print(
            f"Update {update:4d}/{num_updates} | "
            f"lr={lr:.2e} | "
            f"actor={losses['actor_loss']:.4f} | "
            f"critic={losses['critic_loss']:.4f} | "
            f"steps={len(buffer['states'])} | "
            f"win%={wins/len(episodes):.0%}"
        )

        if update % save_every == 0:
            path = os.path.join(checkpoint_dir, f"model_{update:04d}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

        # Evaluate against reference model, save best
        if update % eval_every == 0:
            model.eval()
            results = evaluate(model, reference, num_games=100, device=device)
            win_rate = results['model1_wins']
            model.train()
            print(f"  vs reference: {win_rate:.0%} wins", end="")
            if win_rate > best_vs_reference:
                best_vs_reference = win_rate
                torch.save(model.state_dict(), best_path)
                print(f"  ← new best!")
            else:
                print()

    final_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Best: {best_vs_reference:.0%} vs reference. "
          f"Saved to {final_path} and {best_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train(
        num_updates=args.updates,
        episodes_per_update=args.episodes,
        save_every=args.save_every,
        eval_every=args.eval_every,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
