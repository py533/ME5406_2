"""
main.py — Train SAC and TD3 from scratch with tuned hyperparameters.

Both algorithms are trained in headless (no GUI) mode, evaluated on 20
test episodes, and saved to ./models/.

Run full training:
    python3 main.py

Run TD3 only (SAC model already saved):
    python3 main.py --td3-only
"""

import os
import sys
import csv
import numpy as np
from tqdm import tqdm

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from ur5_env import UR5RobotiqEnv

# ── paths & global settings ───────────────────────────────────────────────────
SAC_MODEL   = "./models/ur_robot_sac_20000_steps"
TD3_MODEL   = "./models/ur_robot_td3_benchmark"
SAC_CURVE_LOG = "./logs/success_sac.csv"
TD3_CURVE_LOG = "./logs/success_td3.csv"
TRAIN_STEPS     = 20000   # SAC training steps
TD3_TRAIN_STEPS = 20000   # TD3 training steps
EVAL_EPS        = 20
CURVE_EVAL_EPS  = 30
CURVE_EVAL_FREQ = 100
TRAIN_REACH_THRESHOLD_M = 0.01  # 1 cm success radius during training/eval


# ── progress bar callback ─────────────────────────────────────────────────────

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_steps: int, algo: str):
        super().__init__()
        self.total_steps = total_steps
        self.algo        = algo
        self._pbar: tqdm = None

    def _on_training_start(self) -> None:
        color = "cyan" if self.algo == "SAC" else "blue"
        self._pbar = tqdm(
            total=self.total_steps,
            desc=f"  [{self.algo}] training",
            unit="step", ncols=72, colour=color,
            bar_format=(
                "  {l_bar}{bar}| {n_fmt}/{total_fmt} steps"
                " [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    def _on_step(self) -> bool:
        self._pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self._pbar.close()


class SuccessRateCallback(BaseCallback):
    def __init__(self, total_steps: int, algo: str, log_path: str,
                 eval_freq: int = CURVE_EVAL_FREQ,
                 n_eval_episodes: int = CURVE_EVAL_EPS):
        super().__init__()
        self.total_steps = total_steps
        self.algo = algo
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.window_episodes = n_eval_episodes
        self._last_eval_step = 0
        self._reach_hist = []
        self._grasp_hist = []

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "reach_rate", "grasp_rate", "window_episodes"])

    def _append_eval(self, step: int) -> None:
        if len(self._reach_hist) == 0:
            reach_rate = 0.0
            grasp_rate = 0.0
            window = 0
        else:
            window = min(self.window_episodes, len(self._reach_hist))
            reach_rate = float(np.mean(self._reach_hist[-window:]) * 100.0)
            grasp_rate = float(np.mean(self._grasp_hist[-window:]) * 100.0)

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reach_rate, grasp_rate, window])
        print(
            f"\n  [{self.algo}] step {step:5d}: "
            f"reach {reach_rate:.1f}% · grasp {grasp_rate:.1f}% "
            f"(last {window} eps)"
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self._reach_hist.append(1.0 if info.get("reached", False) else 0.0)
            self._grasp_hist.append(1.0 if info.get("grasp_success", False) else 0.0)

        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._append_eval(self.num_timesteps)
            self._last_eval_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self._last_eval_step < self.total_steps:
            self._append_eval(self.total_steps)


# ── post-training evaluation ──────────────────────────────────────────────────

def _evaluate(model, algo: str, n_eps: int = EVAL_EPS, show_progress: bool = True) -> dict:
    env = UR5RobotiqEnv(
        headless=True,
        reach_threshold_m=TRAIN_REACH_THRESHOLD_M,
    )
    reach, grasp = 0, 0

    pbar = None
    if show_progress:
        pbar = tqdm(total=n_eps, desc=f"  [{algo}] evaluating",
                    unit="ep", ncols=72, colour="green",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} episodes")

    for _ in range(n_eps):
        obs, _ = env.reset()
        done, reached, grasped = False, False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            if info.get("reached", False):
                reached = True
            if info.get("grasp_success", False):
                grasped = True
        if reached:
            reach += 1
        if grasped:
            grasp += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    env.close()
    return {
        "reach_rate": reach / n_eps * 100,
        "grasp_rate": grasp / n_eps * 100,
    }


# ── SAC training ──────────────────────────────────────────────────────────────
#
# Key tuning choices vs. defaults:
#   learning_rate  -> 1e-3      less oscillation than 3e-3
#   learning_starts-> 1000      better replay warm-up before updates
#   batch_size     -> 256       robust default for SAC stability
#   gradient_steps -> 1         avoid over-updating per env step
#   use_sde        -> True      smoother exploration in continuous control
#   ent_coef       -> auto_0.1  mildly lower entropy for grasp precision

def train_sac() -> dict:
    print("\n" + "━" * 54)
    print("  SAC  (Soft Actor-Critic) — training from scratch")
    print("━" * 54)

    env = Monitor(
        UR5RobotiqEnv(
            headless=True,
            reach_threshold_m=TRAIN_REACH_THRESHOLD_M,
        ),
        "./logs/monitor_sac.csv",
        info_keywords=("reached", "grasp_success"),
    )
    model = SAC(
        "MlpPolicy", env,
        # ── key fixes vs. previous version ────────────────────────────────────
        # learning_rate  3e-4   : default & stable (1e-3 caused oscillation)
        # learning_starts 500   : faster warm-up (1000 wasted 10 % of budget)
        # gradient_steps  2     : double updates per env-step → faster convergence
        # ent_coef "auto"       : full auto-tune from higher initial entropy
        # use_sde False         : standard Gaussian noise is better for this task
        # ─────────────────────────────────────────────────────────────────────
        learning_rate    = 3e-4,
        buffer_size      = 100_000,
        learning_starts  = 500,
        batch_size       = 256,
        tau              = 0.005,
        gamma            = 0.99,
        train_freq       = 1,
        gradient_steps   = 2,
        ent_coef         = "auto",
        use_sde          = False,
        policy_kwargs    = dict(net_arch=[256, 256]),
        verbose          = 0,
    )
    callbacks = [
        ProgressBarCallback(TRAIN_STEPS, "SAC"),
        SuccessRateCallback(TRAIN_STEPS, "SAC", SAC_CURVE_LOG),
    ]
    model.learn(total_timesteps=TRAIN_STEPS, callback=callbacks)
    model.save(SAC_MODEL)
    env.close()
    print(f"\n  ✓ SAC saved → {SAC_MODEL}.zip")

    return _evaluate(model, "SAC")


# ── TD3 noise-decay callback ──────────────────────────────────────────────────
#
# Literature basis:
#   • Fujimoto et al. "Addressing Function Approximation Error in
#     Actor-Critic Methods" (TD3, ICML 2018) — exploration noise must be
#     tuned so the policy can exploit precisely once learned.
#   • Plappert et al. "Parameter Space Noise for Exploration" (ICLR 2018)
#     and OpenAI Spinning Up docs both recommend decaying action noise
#     from a broad initial value to near-zero to allow precision at the
#     end of training.
#   • Andrychowicz et al. "Hindsight Experience Replay" (NeurIPS 2017) —
#     sparse reward tasks need dense exploration early; noise decay
#     approximates this without requiring a goal-conditioned API.

class TwoPhaseNoiseCallback(BaseCallback):
    """
    Two-phase noise schedule for TD3 exploration:

    Phase 1  [0 … switch_step] : hold at sigma_start (broad exploration)
    Phase 2  [switch_step … T] : linear decay to sigma_end (precision)

    Rationale (Fujimoto 2018; Plappert 2018):
      • Phase 1 keeps noise large so ALL discrete cube positions are
        reached by chance, seeding the Q-function with diverse success
        transitions from every position.
      • Phase 2 rapidly reduces noise so the deterministic policy can
        exploit the learned Q-function with high precision, pushing
        success rate above the plateau that fixed-noise training creates.
    """
    def __init__(self, noise: NormalActionNoise,
                 sigma_start: float, sigma_end: float,
                 total_steps: int, switch_step: int):
        super().__init__()
        self._noise        = noise
        self._sigma_start  = sigma_start
        self._sigma_end    = sigma_end
        self._total_steps  = total_steps
        self._switch_step  = switch_step

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t <= self._switch_step:
            sigma = self._sigma_start
        else:
            frac  = min(1.0, (t - self._switch_step) /
                        (self._total_steps - self._switch_step))
            sigma = self._sigma_start + frac * (self._sigma_end - self._sigma_start)
        self._noise._sigma = np.ones_like(self._noise._sigma) * sigma
        return True


# ── TD3 training ──────────────────────────────────────────────────────────────
#
# v3 — complete redesign to address the local-optimum trap seen in v1/v2
#
# Root-cause of previous failure (≈ 26 % success):
#   The deterministic policy collapsed to outputting one "average" action
#   that was correct for only 1-2 of the 6 discrete cube positions.
#   Three reinforcing problems caused this:
#     A) target_policy_noise = 0.1 (≈ 2–3 cm) > success threshold (1 cm):
#        the Bellman target rewards actions that are far from optimal,
#        making the Q-function unable to distinguish the optimal action.
#     B) gradient_steps = 1 per env step gave only ≈ 9 800 critic updates
#        — insufficient to learn a precise 2-D identity mapping.
#     C) sigma decay ending at 0.03 left ≈ 0.6 cm noise at episode 10 000,
#        still comparable to the 1 cm success radius.
#
# Fixes:
#   1. target_policy_noise = 0.02, clip = 0.05
#      Target smoothing noise now << success threshold (1 cm); the
#      Q-function can accurately value precise actions.
#      (Fujimoto 2018 — "noise should be small enough not to change the
#       relative ordering of actions near the optimum")
#
#   2. gradient_steps = 4   (4× increase from v2)
#      Gives ≈ 39 600 critic and ≈ 19 800 actor updates within 10 000
#      env steps.  High replay ratio is well-supported for off-policy
#      methods when the replay buffer is diverse enough.
#      (Fedus et al. "Revisiting Fundamentals of ER", ICML 2020)
#
#   3. Noise decay: sigma 0.20 → 0.005, starting at step 100
#      Broad early exploration (0.20 ≈ 4 cm) decays to near-zero
#      (0.005 ≈ 0.1 cm) by the end of training.
#
#   4. learning_starts = 100   (down from 200)
#      Faster warm-up so gradient updates begin during diverse early
#      exploration, seeding the replay buffer before noise starts decaying.
#
#   5. learning_rate = 3e-4   (down from 1e-3)
#      More conservative LR to match the higher gradient_steps; prevents
#      value-function divergence under 4× update frequency.
#      (Schul et al. 2016; SB3 docs recommend ≤ 1e-3 for TD3)

def train_td3() -> dict:
    print("\n" + "━" * 54)
    print("  TD3  (Twin Delayed DDPG) — training from scratch")
    print("━" * 54)

    n_actions   = 2
    sigma_start = 0.08   # ~15 % of action range (0.4–0.6); aligned with SAC exploration scale
    sigma_end   = 0.005  # near-zero precision at end of training
    switch_step = 0      # no hold phase; linear decay from step 0 (aligned with SAC smooth schedule)

    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = sigma_start * np.ones(n_actions),
    )
    env = Monitor(
        UR5RobotiqEnv(
            headless=True,
            reach_threshold_m=TRAIN_REACH_THRESHOLD_M,
        ),
        "./logs/monitor_td3.csv",
        info_keywords=("reached", "grasp_success"),
    )
    model = TD3(
        "MlpPolicy", env,
        learning_rate        = 3e-4,
        buffer_size          = 100_000,
        learning_starts      = 500,      # aligned with SAC (500)
        batch_size           = 256,
        tau                  = 0.005,
        gamma                = 0.99,
        train_freq           = 1,
        gradient_steps       = 2,       # aligned with SAC (2); reduces Q-function over-fitting risk
        policy_delay         = 2,       # standard TD3 delayed policy update
        target_policy_noise  = 0.02,    # << 1 cm threshold → precise targets
        target_noise_clip    = 0.05,
        action_noise         = action_noise,
        policy_kwargs        = dict(net_arch=[256, 256]),
        verbose              = 0,
    )
    callbacks = [
        ProgressBarCallback(TD3_TRAIN_STEPS, "TD3"),
        SuccessRateCallback(TD3_TRAIN_STEPS, "TD3", TD3_CURVE_LOG),
        TwoPhaseNoiseCallback(
            noise        = action_noise,
            sigma_start  = sigma_start,
            sigma_end    = sigma_end,
            total_steps  = TD3_TRAIN_STEPS,
            switch_step  = switch_step,
        ),
    ]
    model.learn(total_timesteps=TD3_TRAIN_STEPS, callback=callbacks)
    model.save(TD3_MODEL)
    env.close()
    print(f"\n  ✓ TD3 saved → {TD3_MODEL}.zip")

    return _evaluate(model, "TD3")


# ── results summary ───────────────────────────────────────────────────────────

def _print_summary(sac: dict, td3: dict) -> None:
    W   = 54
    bar = lambda pct: "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    print("\n" + "━" * W)
    print(f"  RESULTS  ({TRAIN_STEPS} training steps · {EVAL_EPS} test episodes each)")
    print("━" * W)
    print(f"  {'Metric':<14} {'SAC':>10}   {'TD3':>10}")
    print(f"  {'─' * 40}")
    for label, key in [("Reach rate", "reach_rate"), ("Grasp rate", "grasp_rate")]:
        sv, tv = sac[key], td3[key]
        winner = " ◀" if sv > tv else ("   " if sv == tv else "")
        print(f"  {label:<14} {sv:>9.1f}%  {tv:>9.1f}%{winner if sv > tv else ''}")
        if sv <= tv:
            print(f"  {label:<14} {sv:>9.1f}%  {tv:>9.1f}%  ◀" if sv < tv else
                  f"  {label:<14} {sv:>9.1f}%  {tv:>9.1f}%")
    print(f"  {'─' * 40}")
    print(f"  {'SAC grasp':<14} {bar(sac['grasp_rate'])}")
    print(f"  {'TD3 grasp':<14} {bar(td3['grasp_rate'])}")
    print("━" * W)
    winner = "SAC" if sac["grasp_rate"] >= td3["grasp_rate"] else "TD3"
    print(f"  Best grasp rate: {winner}  "
          f"({sac['grasp_rate']:.1f}% vs {td3['grasp_rate']:.1f}%)")
    print("━" * W)
    print(f"\n  → Run  python3 demo.py  for the side-by-side GUI demo.\n")


def _cleanup_intermediate_models() -> None:
    keep = {
        os.path.abspath(SAC_MODEL + ".zip"),
        os.path.abspath(TD3_MODEL + ".zip"),
    }
    removed = 0
    for name in os.listdir("./models"):
        if not name.endswith(".zip"):
            continue
        if not (name.startswith("ur_robot_sac_") or name.startswith("ur_robot_td3_")):
            continue
        path = os.path.abspath(os.path.join("./models", name))
        if path in keep:
            continue
        os.remove(path)
        removed += 1
    print(f"  Model cleanup: removed {removed} intermediate checkpoint(s).")


def main() -> None:
    td3_only = "--td3-only" in sys.argv

    if td3_only:
        # SAC is already trained — only retrain TD3 with improved hyperparams
        print("\n  Mode: TD3-only (SAC model kept as-is)")
        td3_results = train_td3()
        # Load SAC results from saved model for the summary
        try:
            sac_results = _evaluate(SAC.load(SAC_MODEL), "SAC")
        except Exception:
            sac_results = {"reach_rate": float("nan"), "grasp_rate": float("nan")}
    else:
        td3_results = train_td3()
        sac_results = train_sac()

    _print_summary(sac_results, td3_results)
    _cleanup_intermediate_models()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
