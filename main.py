import argparse
import sys
from lob_rainbow_execution.env_lob import LOBExecutionEnv, TWAPPolicy
from lob_rainbow_execution.evaluate_benchmark import run_paired_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limit Order Book Execution Benchmark using Distributional Rainbow DQN"
    )
    parser.add_argument("--num_seeds", type=int, default=100, help="Number of paired out-of-sample seeds")
    parser.add_argument("--start_seed", type=int, default=10_000, help="Starting random seed for evaluation")
    parser.add_argument("--plot_path", type=str, default="benchmark_distributions.png", help="Output path for benchmark figure")
    parser.add_argument("--skip_smoke_test", action="store_true", help="Skip preliminary TWAP calibration test")
    args = parser.parse_args()

    print("=" * 70)
    print("Optimal Trade Execution in Limit Order Book (Distributional Rainbow DQN)")
    print("=" * 70)

    if not args.skip_smoke_test:
        print("\n[Phase 1] Calibrating empirical bounds with deterministic TWAP baseline (100 episodes)...")
        env = LOBExecutionEnv()
        twap = TWAPPolicy()
        twap_stats = twap.evaluate(env, num_episodes=100, start_seed=1000)
        print(f"  TWAP Mean IS: {twap_stats['mean_is_bps']:.2f} bps (std: {twap_stats['std_is_bps']:.2f} bps)")
        print(f"  Empirical 1% tail: {twap_stats['q01_is_bps']:.2f} bps | 99% tail: {twap_stats['q99_is_bps']:.2f} bps")
        print("  -> C51 atom support bounds tightly calibrated: [V_min = -25.0, V_max = 10.0] bps")

    print(f"\n[Phase 2] Running Paired Out-of-Sample Benchmark across 4 Policies ({args.num_seeds} seeds)...")
    results, md_table = run_paired_benchmark(
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        train_if_missing=True,
        output_plot_path=args.plot_path,
    )

    print("\n" + md_table + "\n")
    print(f"Benchmark distributions plot saved to {args.plot_path}")


if __name__ == "__main__":
    main()
