import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup Paths ---
log_dir = "./logs/"


def plot_clean_zoom():
    print("Generating Clean Zoomed Plot (No Arrows)...")
    try:
        # 1. Load Data
        file_normal = os.path.join(log_dir, "monitor_normal.csv.monitor.csv")
        file_adapt = os.path.join(log_dir, "monitor_adapt.csv.monitor.csv")

        df1 = pd.read_csv(file_normal, skiprows=1)
        df3 = pd.read_csv(file_adapt, skiprows=1)

        # 2. Construct Time Axis
        df1['cumsum_steps'] = df1['l'].cumsum()
        transition_point = df1['cumsum_steps'].iloc[-1]
        df3['cumsum_steps'] = df3['l'].cumsum() + transition_point

        # 3. Smoothing (Technique: Keep Phase 3 window small to prevent smoothing out the performance drop)
        df1['reward_smooth'] = df1['r'].rolling(window=5, min_periods=1).mean()
        df3['reward_smooth'] = df3['r'].rolling(window=5, min_periods=1).mean()

        # 4. Data Slicing (Zoom in on the 147k - 152k range)
        zoom_start = 147500
        zoom_end = 152500

        df1_zoom = df1[(df1['cumsum_steps'] >= zoom_start) & (df1['cumsum_steps'] <= zoom_end)]
        df3_zoom = df3[(df3['cumsum_steps'] >= zoom_start) & (df3['cumsum_steps'] <= zoom_end)]

        # ==========================================
        # Plotting (Clean Version)
        # ==========================================
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot lines
        plt.plot(df1_zoom['cumsum_steps'], df1_zoom['reward_smooth'],
                 label="Baseline", color='#1f77b4', linewidth=2.5)

        plt.plot(df3_zoom['cumsum_steps'], df3_zoom['reward_smooth'],
                 label="Adaptation", color='#2ca02c', linewidth=2.5)

        # Add fault injection line only
        plt.axvline(x=transition_point, color='red', linestyle='--', linewidth=2, alpha=0.8,
                    label="Bias Injection")

        # Set axis labels
        plt.xlabel("Total Timesteps", fontsize=12)
        plt.ylabel("Episode Reward", fontsize=12)
        plt.title("Response to Sensor Drift (Detailed View)", fontsize=14, fontweight='bold')

        # Place legend in upper left to avoid covering curves
        plt.legend(loc="upper left", frameon=True)

        # Enforce X-axis limits
        plt.xlim(zoom_start, zoom_end)

        # Save figure
        output_filename = "Calibration_Zoomed_Clean.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Success! Saved as '{output_filename}'")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    plot_clean_zoom()