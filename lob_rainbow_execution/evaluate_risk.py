"""
Paired-Path Evaluation: IS (bps), Sharpe, and 95% CVaR metrics.
Aliases and re-exports functionality from evaluate_benchmark.py for compatibility.
"""

from lob_rainbow_execution.evaluate_benchmark import (
    evaluate_policy_on_seeds,
    format_markdown_table,
    plot_benchmark_distributions,
    run_paired_benchmark,
)

__all__ = [
    "evaluate_policy_on_seeds",
    "format_markdown_table",
    "plot_benchmark_distributions",
    "run_paired_benchmark",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate risk metrics on paired seeds.")
    parser.add_argument("--num_seeds", type=int, default=1000, help="Number of seeds")
    parser.add_argument("--start_seed", type=int, default=10000, help="Start seed")
    parser.add_argument("--plot_path", type=str, default="benchmark_distributions.png", help="Plot output path")
    args = parser.parse_args()

    results, md_table = run_paired_benchmark(
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        output_plot_path=args.plot_path,
    )
    print("\nBenchmark Evaluation Results (Paired Seeds):\n")
    print(md_table)
