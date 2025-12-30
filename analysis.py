import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


# ==========================================
# 1. 定义核心损伤环境 (Motor Damage Wrapper)
# ==========================================
class MotorDamageWrapper(gym.ActionWrapper):
    def __init__(self, env, weakness=1.0, bias=0.3):
        """
        weakness: 力矩衰减系数 (1.0 = 正常, 0.5 = 只有50%力矩)
        bias: 电极漂移/固有偏差 (0.0 = 正常, 0.3 = 显著漂移)
        """
        super().__init__(env)
        self.weakness = weakness
        self.bias = bias

    def action(self, action):
        damaged_action = np.array(action, dtype=np.float32).copy()
        # 假设故障发生在第二个关节 (Elbow)
        if len(damaged_action) >= 2:
            damaged_action[1] = (damaged_action[1] * self.weakness) + self.bias

        # 确保动作不超出物理限制
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "low"):
            damaged_action = np.clip(damaged_action, self.env.action_space.low, self.env.action_space.high)
        return damaged_action


# ==========================================
# 2. 评估辅助函数
# ==========================================
def make_eval_env(env_id, wrapper_class=None, wrapper_kwargs=None, stats_path=None):
    """创建一个用于评估的环境，并加载对应的标准化参数"""
    env = gym.make(env_id, render_mode=None)
    if wrapper_class:
        env = wrapper_class(env, **(wrapper_kwargs or {}))

    env = DummyVecEnv([lambda: env])

    # 加载 VecNormalize 统计数据 (关键步骤，否则神经网络会看到错误的数据分布)
    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False  # 评估模式，不更新统计数据
        env.norm_reward = False  # 评估时通常看原始 Reward
    else:
        print(f"Warning: Stats file {stats_path} not found. Running raw environment.")

    return env


def evaluate_model(model, env, n_episodes=20):
    """运行 N 个 episode 并返回平均奖励和标准差"""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


# ==========================================
# 3. 主分析流程
# ==========================================
if __name__ == "__main__":
    # 配置
    env_id = "Reacher-v4"
    damage_config = {"weakness": 1.0, "bias": 0.3}  # Phase 2/3 使用的损伤参数
    n_episodes = 30  # 每个场景测试的次数

    # 文件路径
    p1_model = "brain_phase1.zip"
    p1_stats = "vec_stats_phase1.pkl"
    p3_model = "brain_phase3_adapted.zip"
    p3_stats = "vec_stats_phase3.pkl"

    results = []

    # --- 加载模型 ---
    print("Loading models...")
    model_phase1 = SAC.load(p1_model) if os.path.exists(p1_model) else None
    model_phase3 = SAC.load(p3_model) if os.path.exists(p3_model) else None

    if not model_phase1:
        print("Error: brain_phase1.zip not found!")
        exit()

    # -------------------------------------------------
    # 实验 1: 状态对比 (Normal vs Damaged vs Recovered)
    # -------------------------------------------------
    print("\n>>> Running Experiment 1: Quantitative Comparison")

    # 1. Normal State (Phase 1 Model + Normal Env)
    env_normal = make_eval_env(env_id, stats_path=p1_stats)
    mean, std = evaluate_model(model_phase1, env_normal, n_episodes)
    results.append({"Condition": "Normal (Phase 1)", "Reward": mean, "Std": std})
    env_normal.close()
    print(f"Normal: {mean:.2f} ± {std:.2f}")

    # 2. Damaged State (Phase 1 Model + Damaged Env)
    env_damaged = make_eval_env(env_id, MotorDamageWrapper, damage_config, stats_path=p1_stats)
    mean, std = evaluate_model(model_phase1, env_damaged, n_episodes)
    results.append({"Condition": "Damaged (Phase 1)", "Reward": mean, "Std": std})
    env_damaged.close()
    print(f"Damaged: {mean:.2f} ± {std:.2f}")

    # 3. Recovered State (Phase 3 Model + Damaged Env)
    if model_phase3:
        # 注意：这里使用 Phase 3 的 stats，因为它已经适应了新的数据分布
        env_recovered = make_eval_env(env_id, MotorDamageWrapper, damage_config, stats_path=p3_stats)
        mean, std = evaluate_model(model_phase3, env_recovered, n_episodes)
        results.append({"Condition": "Recovered (Phase 3)", "Reward": mean, "Std": std})
        env_recovered.close()
        print(f"Recovered: {mean:.2f} ± {std:.2f}")

    # 绘图 1：柱状图
    df = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df["Condition"], df["Reward"], yerr=df["Std"], capsize=10,
                   color=['#1f77b4', '#d62728', '#2ca02c'], alpha=0.8)
    plt.title("Impact of Motor Damage and Rehabilitation Recovery")
    plt.ylabel("Average Episode Reward")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig("quant_comparison.png", dpi=300)
    print("Saved quant_comparison.png")

    # -------------------------------------------------
    # 实验 2: 敏感性分析 (Ablation Study / Heatmap)
    # -------------------------------------------------
    print("\n>>> Running Experiment 2: Sensitivity Analysis (Bias vs Weakness)")

    # 定义扫描网格
    bias_values = [0.0, 0.1, 0.3, 0.5]
    weakness_values = [1.0, 0.8, 0.6, 0.4]  # 1.0 是健康，0.4 是严重无力

    heatmap_data = np.zeros((len(weakness_values), len(bias_values)))

    # 使用 Phase 1 模型来测试它对不同损伤的鲁棒性
    for i, w in enumerate(weakness_values):
        for j, b in enumerate(bias_values):
            config = {"weakness": w, "bias": b}
            # 临时环境
            env_sense = make_eval_env(env_id, MotorDamageWrapper, config, stats_path=p1_stats)
            mean, _ = evaluate_model(model_phase1, env_sense, n_episodes=5)  # 少量次数以加快速度
            heatmap_data[i, j] = mean
            env_sense.close()
            print(f"Eval W={w}, B={b} -> Reward: {mean:.1f}")

    # 绘图 2：热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=bias_values, yticklabels=weakness_values)
    plt.title("Sensitivity Analysis: Zero-Shot Performance on Damage")
    plt.xlabel("Bias (Drift Magnitude)")
    plt.ylabel("Weakness Factor (1.0 = Normal)")
    plt.savefig("sensitivity_heatmap.png", dpi=300)
    print("Saved sensitivity_heatmap.png")

    print("\nAll Done! Please check the generated PNG files.")