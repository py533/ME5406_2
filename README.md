# UR5 Robotic Grasping with Reinforcement Learning: SAC vs. TD3

This project investigates deep reinforcement learning for robotic grasping, training a **UR5 arm** equipped with a **Robotiq 85 gripper** to autonomously pick up objects in a PyBullet physics simulation. The robot learns entirely from scratch — no pre-programmed motion plans — relying only on a compact 4-D observation (cube position + end-effector position) and a dense distance-based reward signal to discover successful grasp strategies through trial and error.

Two state-of-the-art continuous-control algorithms are benchmarked head-to-head: **SAC (Soft Actor-Critic)**, an entropy-regularised off-policy method that naturally encourages exploration, and **TD3 (Twin Delayed DDPG)**, which addresses value overestimation through clipped double Q-learning and delayed policy updates. Both agents are trained for 20,000 environment steps and evaluated on reach success rate, grasp success rate, and sample efficiency, providing a direct comparison of their learning dynamics in a contact-rich manipulation task.

---

## Demo

<table>
  <tr>
    <th align="center">SAC</th>
    <th align="center">TD3</th>
  </tr>
  <tr>
    <td align="center">
      <video src="demo/SAC_demo.webm" controls width="420">SAC demo</video>
    </td>
    <td align="center">
      <video src="demo/TD3_demo.webm" controls width="420">TD3 demo</video>
    </td>
  </tr>
</table>

---

PyBullet physics simulation of a UR5 arm with a Robotiq 85 gripper learning to grasp objects, benchmarking **SAC (Soft Actor-Critic)** against **TD3 (Twin Delayed DDPG)**.

The robot observes the 2-D cube position and current end-effector position (4-D state), then outputs a 2-D target end-effector position. An episode succeeds when the gripper closes within 1 cm of the cube and lifts it above 0.80 m. Episodes run for up to 100 steps with a dense, distance-based reward.

---

## Main Files

| File | Purpose |
|---|---|
| `ur5_env.py` | Gymnasium environment — PyBullet physics, IK arm control, force-feedback gripper, reward computation |
| `main.py` | Train SAC and TD3; logs reach/grasp success rates every 100 steps; post-training evaluation |
| `demo.py` | Evaluate trained models (100 episodes each); supports headless or GUI mode |
| `generate_images.py` | Generate figures from training logs → `images/figure_a.png`, `figure_b.png`, `figure_c.png` (300 dpi) |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train SAC + TD3 (headless, ~40–60 min on CPU)
python main.py

# Retrain TD3 only (preserves existing SAC model)
python main.py --td3-only

# GUI evaluation — two side-by-side PyBullet windows (default)
python demo.py

# Headless evaluation — no display required
python demo.py --fast

# Generate comparison figures (requires completed training logs)
python generate_images.py
```

> **GUI on a headless server:** Run `Xvfb :1 -screen 0 1280x720x24 & export DISPLAY=:1` before `python demo.py`.

---

## Requirements

**Python 3.10+**

| Library | Version | Purpose |
|---|---|---|
| `pybullet` | 3.2.7 | Physics engine — robot simulation, IK, contact forces |
| `stable-baselines3` | 2.8.0 | SAC and TD3 implementations |
| `gymnasium` | 1.2.3 | RL environment API |
| `torch` | 2.10.0 | Neural network backend |
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.3.3 | Log file parsing |
| `matplotlib` | 3.10.8 | Figure generation |
| `scipy` | 1.8.0 | Curve smoothing |
| `tqdm` | 4.67.3 | Training progress bars |

No external simulation software is required — PyBullet is installed via pip. GUI mode requires OpenGL; on Ubuntu: `sudo apt install libgl1-mesa-glx`. CUDA is optional; CPU training is fully supported.
