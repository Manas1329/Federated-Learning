import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
})

# Paths
base_dir = r"d:\Codes\College_Projects\Major Project\Federated-Learning\results\experiments"
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
csv_path = os.path.join(base_dir, "final_experiment_results.csv")

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"{name}.png"), format='png', dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{name}.pdf"), format='pdf', bbox_inches='tight')

# Load data
df = pd.read_csv(csv_path)

# ==================================================
# FIGURE 1 — FINAL ACCURACY BY EXPERIMENT
# ==================================================
def plot_fig1():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_order = [
        'exp1_baseline',
        'exp2_one_straggler',
        'exp3_two_stragglers',
        'exp4_network_dropout',
        'exp5_recovery',
        'exp6_adaptive_off',
        'exp7_nonstationary'
    ]
    
    labels = ["Exp 1: Baseline", "Exp 2: One Straggler", "Exp 3: Two Stragglers", 
              "Exp 4: Network Dropout", "Exp 5: Recovery", "Exp 6: Adaptive OFF", "Exp 7: Nonstationary"]
    
    accuracies = []
    colors = []
    
    for exp in exp_order:
        row = df[df['experiment'] == exp].iloc[0]
        accuracies.append(row['final_accuracy'])
        if row['adaptive_dropout'] == 'ON':
            colors.append('#1f77b4') # Blue
        else:
            colors.append('#ff7f0e') # Orange
            
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', zorder=3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_title('Final Global Model Accuracy by Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Adaptive Dropout ON'),
                       Patch(facecolor='#ff7f0e', edgecolor='black', label='Adaptive Dropout OFF')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    save_fig(fig, 'fig1_final_accuracy')
    plt.close(fig)

# ==================================================
# FIGURE 2 — SUCCESSFUL VS FAILED/ABORTED CLIENT-ROUNDS
# ==================================================
def plot_fig2():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_order = [
        'exp1_baseline',
        'exp2_one_straggler',
        'exp3_two_stragglers',
        'exp4_network_dropout',
        'exp5_recovery',
        'exp6_adaptive_off',
        'exp7_nonstationary'
    ]
    
    labels = ["Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5", "Exp 6", "Exp 7"]
    
    successful = []
    failed = []
    aborted = []
    
    for exp in exp_order:
        row = df[df['experiment'] == exp].iloc[0]
        successful.append(row['successful_client_rounds'])
        failed.append(row['failed_client_rounds'])
        aborted.append(row['aborted_aggregations'])
        
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, successful, width, label='Successful Client-Rounds', color='#2ca02c', edgecolor='black', zorder=3)
    ax.bar(x, failed, width, label='Failed Client-Rounds', color='#d62728', edgecolor='black', zorder=3)
    ax.bar(x + width, aborted, width, label='Aborted Aggregations', color='#7f7f7f', edgecolor='black', zorder=3)
    
    ax.set_ylabel('Count')
    ax.set_title('Operational Outcomes per Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    save_fig(fig, 'fig2_client_round_outcomes')
    plt.close(fig)

# ==================================================
# FIGURE 3 — EXPERIMENT OUTCOMES
# ==================================================
def plot_fig3():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_order = [
        'exp1_baseline',
        'exp6_adaptive_off',
        'exp2_one_straggler',
        'exp3_two_stragglers',
        'exp4_network_dropout',
        'exp5_recovery',
        'exp7_nonstationary'
    ]
    
    labels = ["Exp 1\nBaseline", "Exp 6\nAdaptive OFF", "Exp 2\n1 Straggler", 
              "Exp 3\n2 Stragglers", "Exp 4\nNet Dropout", "Exp 5\nRecovery", "Exp 7\nNonstationary"]
    
    accuracies = []
    colors = []
    
    for exp in exp_order:
        row = df[df['experiment'] == exp].iloc[0]
        accuracies.append(row['final_accuracy'])
        if exp == 'exp1_baseline':
            colors.append('#2ca02c') # Green for baseline
        elif exp == 'exp6_adaptive_off':
            colors.append('#ff7f0e') # Orange for OFF
        else:
            colors.append('#1f77b4') # Blue for adverse scenarios
            
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', zorder=3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_title('Final Accuracy Under Tested Federated Learning Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Baseline (No adverse condition)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Adaptive OFF (Control)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Adaptive ON (Adverse Scenarios)')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    save_fig(fig, 'fig3_experiment_outcomes')
    plt.close(fig)

# ==================================================
# FIGURE 4 — NONSTATIONARY CLIENT BEHAVIOR
# ==================================================
def plot_fig4():
    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    obs_time = [34.6, 33.9, 106.3, np.nan, np.nan, 25.3, np.nan, np.nan, np.nan, np.nan]
    ema_time = [34.6, 34.4, 56.0, np.nan, np.nan, 46.8, np.nan, np.nan, np.nan, np.nan]
    pred_time = [34.6, 34.4, 34.4, 77.7, np.nan, np.nan, 71.2, np.nan, np.nan, np.nan]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rounds, obs_time, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8, label='Observed Completion Time')
    ax.plot(rounds, pred_time, marker='s', color='#ff7f0e', linestyle='--', linewidth=2, markersize=8, label='Predicted Completion (EMA + Dev)')
    ax.axhline(y=60, color='red', linestyle=':', linewidth=2, label='Hard Deadline (60s)')
    
    ax.scatter([4, 7], [1.2, 1.1], color='red', marker='X', s=150, zorder=5, label='Preemptively Dropped')
    
    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Exp 7: Adaptive Prediction of Nonstationary Delay (Hospital_C)')
    ax.set_xticks(rounds)
    ax.set_xlim(0.5, 10.5)
    
    ax.annotate('70s Delay\nInjected', xy=(3, 106.3), xytext=(2.5, 80),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    ax.annotate('70s Delay\nInjected\n(Preempted)', xy=(7, 71.2), xytext=(7.5, 90),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
                
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    save_fig(fig, 'fig4_nonstationary_hospital_c')
    plt.close(fig)

try:
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    
    # We remove the old files that were incorrectly named or duplicated
    import glob
    old_files = glob.glob(os.path.join(figures_dir, 'fig3_adaptive_on_vs_off.*')) + glob.glob(os.path.join(figures_dir, 'fig5_adaptive_prediction.*'))
    for f in old_files:
        try:
            os.remove(f)
        except:
            pass
            
    print("Figures generated successfully.")
except Exception as e:
    print(f"Error generating figures: {e}")

