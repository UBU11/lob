"""Benchmark environment simulation throughput in steps per second."""
import time
import numpy as np
from lob_rainbow_execution.env_lob import LOBExecutionEnv

def profile_env(num_episodes: int = 50) -> dict[str, float]:
    env = LOBExecutionEnv()
    start_time = time.perf_counter()
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = int(total_steps % 5)
            obs, reward, term, trunc, _ = env.step(action)
            total_steps += 1
            done = term or trunc

    elapsed = time.perf_counter() - start_time
    fps = total_steps / elapsed
    return {"total_steps": total_steps, "elapsed_seconds": elapsed, "steps_per_sec": fps}

if __name__ == "__main__":
    res = profile_env(100)
    print(f"Total steps: {res['total_steps']}")
    print(f"Elapsed: {res['elapsed_seconds']:.2f}s")
    print(f"Throughput: {res['steps_per_sec']:.0f} steps/second")
