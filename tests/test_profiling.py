from scripts.profile_throughput import profile_env

def test_profiling_execution() -> None:
    res = profile_env(num_episodes=5)
    assert res["total_steps"] == 5 * 20
    assert res["steps_per_sec"] > 0.0
