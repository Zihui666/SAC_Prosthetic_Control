import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. Define Motor Fault Wrapper (Core Modification)
# ==========================================
class MotorDamageWrapper(gym.ActionWrapper):
    def __init__(self, env, weakness=1.0, bias=0.3):
        """
        Simulate realistic motor faults:
        weakness: Torque output attenuation (e.g., 1.0 means 100% force remains).
        bias: Zero-point drift/Constant offset (e.g., 0.3 means the motor outputs 0.3 force even with 0 command).
        """
        super().__init__(env)
        self.weakness = weakness
        self.bias = bias

    def action(self, action):
        # Copy the action to avoid modifying original data
        damaged_action = np.array(action, dtype=np.float32).copy()

        # Reacher typically has 2D actions [Joint1, Joint2].
        # We assume Joint1 (Shoulder) is healthy, but Joint2 (Elbow) is damaged.
        # This "asymmetrical fault" is challenging for neural networks and highly realistic.
        if len(damaged_action) >= 2:
            # Fault logic: Output = Input * Attenuation + Bias
            damaged_action[1] = (damaged_action[1] * self.weakness) + self.bias

        # Must clip to prevent physical values from going out of bounds
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "low"):
            damaged_action = np.clip(damaged_action, self.env.action_space.low, self.env.action_space.high)

        return damaged_action


# ==========================================
# 2. Recording Function
# ==========================================
def record_gif(model, env, filename, steps=300):
    obs = env.reset()
    images = []
    print(f"Recording {filename}...")
    for _ in range(steps):
        # deterministic=True shows the model's true capability (without random noise)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            images.append(frame)
    imageio.mimsave(filename, images, fps=30)
    print(f"Saved {filename}")


# ==========================================
# 3. Plotting Function (Fixed visualization issues)
# ==========================================
def plot_combined_learning_curve(log_dir):
    try:
        # Read Phase 1 (Normal) data
        df1 = pd.read_csv(os.path.join(log_dir, "monitor_normal.csv.monitor.csv"), skiprows=1)
        df1['cumsum_steps'] = df1['l'].cumsum()

        # Read Phase 3 (Adaptation) data
        df3 = pd.read_csv(os.path.join(log_dir, "monitor_adapt.csv.monitor.csv"), skiprows=1)
        last_step = df1['cumsum_steps'].iloc[-1]
        # Continue step count from where Phase 1 left off
        df3['cumsum_steps'] = df3['l'].cumsum() + last_step

        # === Key Modification: Smoothing Parameters ===
        # window=15: Moderate smoothing to retain trend visibility.
        # min_periods=1: Plot immediately to ensure the initial crash is not hidden.
        window_size = 15
        df1['reward_smooth'] = df1['r'].rolling(window=window_size, min_periods=1).mean()
        df3['reward_smooth'] = df3['r'].rolling(window=window_size, min_periods=1).mean()

        plt.figure(figsize=(12, 6))

        # Plot curves
        plt.plot(df1['cumsum_steps'], df1['reward_smooth'], label="Normal Brain (Phase 1)", color='blue', alpha=0.8)
        plt.plot(df3['cumsum_steps'], df3['reward_smooth'], label="Adapting Brain (Phase 3)", color='green', alpha=0.8)

        # Draw fault injection line
        plt.axvline(x=last_step, color='red', linestyle='--', linewidth=2, label="Motor Damage Incident")

        # Add text annotation
        plt.text(last_step + 5000, df1['reward_smooth'].min(), "Performance Drop\nDue to Bias", color='red')

        plt.xlabel("Total Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("Neuroplasticity Simulation: Recovering from Motor Bias & Weakness")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = "Motor_Damage_Result.png"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        print(f"Plotting failed: {e}")


# ==========================================
# 4. Environment Helper Function
# ==========================================
def make_env(env_id, log_dir, filename, wrapper_class=None, wrapper_kwargs=None, render_mode=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if wrapper_class:
            env = wrapper_class(env, **(wrapper_kwargs or {}))
        env = Monitor(env, os.path.join(log_dir, filename))
        return env

    return _init


# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 150k steps is sufficient for Reacher convergence
    TRAIN_STEPS = 150000

    # === Fault Definition ===
    # weakness=1.0: Normal strength (no weakness).
    # bias=0.3: Significant drift (this causes the major reward drop).
    DAMAGE_CONFIG = {"weakness": 1.0, "bias": 0.3}

    # ---------------------------------------
    # Phase 1: Normal Training (Establishing Baseline)
    # ---------------------------------------
    print("\n=== Phase 1: Normal Training ===")
    env_train_normal = DummyVecEnv([make_env("Reacher-v4", log_dir, "monitor_normal.csv", render_mode=None)])
    env_train_normal = VecNormalize(env_train_normal, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = SAC("MlpPolicy", env_train_normal, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=TRAIN_STEPS)

    model.save("brain_phase1")
    env_train_normal.save("vec_stats_phase1.pkl")
    env_train_normal.close()

    # Record Phase 1
    env_render = DummyVecEnv([make_env("Reacher-v4", log_dir, "dummy_1", render_mode="rgb_array")])
    env_render = VecNormalize.load("vec_stats_phase1.pkl", env_render)
    env_render.training = False;
    env_render.norm_reward = False
    record_gif(model, env_render, "1_Normal_Control.gif")
    env_render.close()

    # ---------------------------------------
    # Phase 2: Damage Evaluation (Direct testing, no training)
    # ---------------------------------------
    print(f"\n=== Phase 2: Damage Evaluation (Bias={DAMAGE_CONFIG['bias']}) ===")

    # Create environment with faults
    env_drift_base = DummyVecEnv([make_env("Reacher-v4", log_dir, "dummy_2",
                                           MotorDamageWrapper, DAMAGE_CONFIG,
                                           render_mode="rgb_array")])

    # Load statistics from the "Old Brain"
    env_drift = VecNormalize.load("vec_stats_phase1.pkl", env_drift_base)
    env_drift.training = False
    env_drift.norm_reward = False

    # Recording: Should see the arm drifting uncontrollably to one side
    record_gif(model, env_drift, "2_Motor_Damage_Effect.gif")
    env_drift.close()

    # ---------------------------------------
    # Phase 3: Adaptation Training (Rehabilitation)
    # ---------------------------------------
    print("\n=== Phase 3: Adaptation Training (Recovering...) ===")

    # 1. Load the "Old Brain" (pre-trained model)
    model = SAC.load("brain_phase1")

    # 2. Create training env: Must include the same fault configuration
    env_train_adapt = DummyVecEnv(
        [make_env("Reacher-v4", log_dir, "monitor_adapt.csv",
                  MotorDamageWrapper, DAMAGE_CONFIG,
                  render_mode=None)])

    # 3. Inherit Observation statistics from Phase 1
    env_train_adapt = VecNormalize.load("vec_stats_phase1.pkl", env_train_adapt)

    # 4. Enable training=True to adapt to the new Observation distribution
    # But keep norm_reward=False
    env_train_adapt.training = True
    env_train_adapt.norm_reward = False

    # 5. Inject environment and start training (Rehab)
    model.set_env(env_train_adapt)
    model.learn(total_timesteps=TRAIN_STEPS)

    model.save("brain_phase3_adapted")
    env_train_adapt.save("vec_stats_phase3.pkl")
    env_train_adapt.close()

    # Record Phase 3
    env_render_recover = DummyVecEnv(
        [make_env("Reacher-v4", log_dir, "dummy_3",
                  MotorDamageWrapper, DAMAGE_CONFIG,
                  render_mode="rgb_array")])

    env_render_recover = VecNormalize.load("vec_stats_phase3.pkl", env_render_recover)
    env_render_recover.training = False;
    env_render_recover.norm_reward = False

    record_gif(model, env_render_recover, "3_Recovered_Control.gif")
    env_render_recover.close()

    # ---------------------------------------
    # Result Visualization
    # ---------------------------------------
    print("\n=== Generating Final Plot ===")
    plot_combined_learning_curve(log_dir)