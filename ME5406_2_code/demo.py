"""
demo.py — Evaluate SAC and TD3 in parallel (100 eps each).

Usage:
    python3 demo.py        # headless / DIRECT mode (default)
    python3 demo.py --gui  # GUI mode

Output:
    ./logs/demo_eval.csv   — reach & grasp rates (read by generate_images.py)
"""

import os
import sys
import time
import csv
import multiprocessing

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SAC_MODEL  = "./models/ur_robot_sac_20000_steps"
TD3_MODEL  = "./models/ur_robot_td3_benchmark"
N_EVAL_EPS = 100
DEMO_REACH_THRESHOLD_M = 0.02  # 2 cm success radius in demo


def _check_models():
    missing = []
    if not os.path.exists(SAC_MODEL + ".zip"):
        missing.append(f"SAC model not found: {SAC_MODEL}.zip")
    if not os.path.exists(TD3_MODEL + ".zip"):
        missing.append(f"TD3 model not found: {TD3_MODEL}.zip")
    if missing:
        for m in missing:
            print(f"  [ERROR] {m}")
        print("  Run  python3 main.py  first to train the models.")
        sys.exit(1)


# ── worker (one process per algorithm) ───────────────────────────────────────

def _worker(algo: str, model_path: str, label_color: list,
            n_episodes: int, headless: bool,
            result_queue: multiprocessing.Queue) -> None:
    """Runs in its own process. Opens a GUI window and evaluates n_episodes."""
    import pybullet as p
    from stable_baselines3 import SAC, TD3

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from ur5_env import UR5RobotiqEnv

    env    = UR5RobotiqEnv(
        headless=headless,
        reach_threshold_m=DEMO_REACH_THRESHOLD_M,
    )
    client = env.physics_client

    if not headless:
        p.addUserDebugText(
            f"[ {algo} ]",
            textPosition=[0.5, -0.55, 1.1],
            textColorRGB=label_color,
            textSize=3.0,
            physicsClientId=client,
        )

    model        = (SAC if algo == "SAC" else TD3).load(model_path)
    reach_count  = 0
    grasp_count  = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done, reached, grasped = False, False, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if not headless:
                time.sleep(1 / 300)
            obs, _, done, _, info = env.step(action)
            if info.get("reached", False):
                reached = True
            if info.get("grasp_success", False):
                grasped = True

        if reached:
            reach_count += 1
        if grasped:
            grasp_count += 1

        reach_rate = reach_count / ep * 100
        grasp_rate = grasp_count / ep * 100
        status     = "GRASP OK" if grasped else "FAILED  "
        print(f"  [{algo}] Ep {ep:3d}/{n_episodes}: {status}  "
              f"reach {reach_rate:.1f}%  grasp {grasp_rate:.1f}%")

    env.close()

    result_queue.put({
        "algo":       algo,
        "reach_rate": reach_count / n_episodes * 100,
        "grasp_rate": grasp_count / n_episodes * 100,
    })
    print(f"\n  [{algo}] Done — reach {reach_count/n_episodes*100:.1f}%  "
          f"grasp {grasp_count/n_episodes*100:.1f}%")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    _check_models()

    headless = "--fast" in sys.argv
    mode_str = "headless (DIRECT)" if headless else "GUI"

    print("\n" + "=" * 60)
    print(f"  SAC (red)  vs  TD3 (blue)  —  {N_EVAL_EPS} episodes each  [{mode_str}]")
    if not headless:
        print("  Two PyBullet windows will open — drag apart to view side by side.")
    print("  Press Ctrl-C to stop early.")
    print("=" * 60 + "\n")

    result_queue = multiprocessing.Queue()

    p_sac = multiprocessing.Process(
        target=_worker,
        args=("SAC", SAC_MODEL, [1.0, 0.2, 0.2],
              N_EVAL_EPS, headless, result_queue),
        name="SAC-window", daemon=True,
    )
    p_td3 = multiprocessing.Process(
        target=_worker,
        args=("TD3", TD3_MODEL, [0.2, 0.5, 1.0],
              N_EVAL_EPS, headless, result_queue),
        name="TD3-window", daemon=True,
    )

    p_sac.start()
    p_td3.start()

    try:
        p_sac.join()
        p_td3.join()
    except KeyboardInterrupt:
        print("\n  Stopping both windows ...")
        p_sac.terminate()
        p_td3.terminate()
        p_sac.join()
        p_td3.join()
        print("  Stopped early — no CSV saved.")
        return

    # Collect results from queue
    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        results[r["algo"]] = r

    if "SAC" not in results or "TD3" not in results:
        print("  Could not collect full results.")
        return

    sac, td3 = results["SAC"], results["TD3"]

    print("\n" + "━" * 60)
    print(f"  {'Metric':<14} {'SAC':>10}   {'TD3':>10}")
    print(f"  {'─' * 42}")
    for label, key in [("Reach rate", "reach_rate"), ("Grasp rate", "grasp_rate")]:
        sv, tv   = sac[key], td3[key]
        winner   = "  ◀ SAC" if sv > tv else ("  ◀ TD3" if tv > sv else "  TIE")
        print(f"  {label:<14} {sv:>9.1f}%  {tv:>9.1f}%{winner}")
    print("━" * 60)

    # Save to CSV for generate_images.py
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/demo_eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "reach_rate", "grasp_rate", "n_episodes"])
        writer.writerow(["SAC", sac["reach_rate"], sac["grasp_rate"], N_EVAL_EPS])
        writer.writerow(["TD3", td3["reach_rate"], td3["grasp_rate"], N_EVAL_EPS])
    print(f"\n  Results saved → ./logs/demo_eval.csv")
    print("  Run  python3 generate_images.py  to regenerate figure_c.\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main()
