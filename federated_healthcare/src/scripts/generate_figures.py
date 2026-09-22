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

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.src.paths import EXPERIMENTS_RESULTS_DIR, FIGURES_DIR

csv_path = EXPERIMENTS_RESULTS_DIR / "final_experiment_results.csv"
figures_dir = FIGURES_DIR

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
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_order = [
        'exp1_baseline',
        'exp2_one_straggler',
        'exp3_two_stragglers',
        'exp4_network_dropout',
        'exp5_recovery',
        'exp6_adaptive_off',
        'exp7_nonstationary'
    ]
    
    # Applying the aesthetic from the old Figure 3
    labels = ["Exp 1:\nBaseline", "Exp 2:\nOne Straggler", "Exp 3:\nTwo Stragglers", 
              "Exp 4:\nNetwork Dropout", "Exp 5:\nRecovery", "Exp 6:\nAdaptive OFF", "Exp 7:\nNonstationary"]
    
    accuracies = []
    colors = []
    
    for exp in exp_order:
        row = df[df['experiment'] == exp].iloc[0]
        accuracies.append(row['final_accuracy'])
        if exp == 'exp1_baseline':
            colors.append('#2ca02c') # Green
        elif row['adaptive_dropout'] == 'ON':
            colors.append('#1f77b4') # Blue
        else:
            colors.append('#ff7f0e') # Orange
            
    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', width=0.5, zorder=3)
    
    # Add values on top (styling from old Fig 3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_text = f'{height:.2f}%'
        if exp_order[i] == 'exp2_one_straggler':
            label_text += '\n(4/10 rounds)'
        
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_title('Final Accuracy Across Tested Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', edgecolor='black', label='Baseline (Reference)'),
                       Patch(facecolor='#1f77b4', edgecolor='black', label='Adaptive Dropout ON'),
                       Patch(facecolor='#ff7f0e', edgecolor='black', label='Adaptive Dropout OFF')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    save_fig(fig, 'fig1_final_accuracy')
    plt.close(fig)

# ==================================================
# FIGURE 2 — SUCCESSFUL VS FAILED/ABORTED CLIENT-ROUNDS (Unchanged)
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
# FIGURE 3 — ADAPTIVE DECISION RESPONSE (Replacement)
# ==================================================
def plot_fig3():
    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    obs_time = [34.6, 33.9, 106.3, np.nan, np.nan, 25.3, np.nan, np.nan, np.nan, np.nan]
    
    # We only plot predicted/safe completion where explicitly available
    pred_time = [np.nan, np.nan, np.nan, 77.7, np.nan, np.nan, 71.2, np.nan, np.nan, np.nan]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot observations
    ax.plot(rounds, obs_time, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8, label='Step 1: Observed Delay')
    
    # Plot predicted/safe values
    ax.plot(rounds, pred_time, marker='s', color='#ff7f0e', linestyle='', markersize=8, label='Step 2: Predicted Safe Completion')
    
    # Hard deadline
    ax.axhline(y=60, color='red', linestyle=':', linewidth=2, label='Step 3: Hard Deadline (60s)')
    
    # Mark dropped rounds
    ax.scatter([4, 7], [1.2, 1.1], color='red', marker='X', s=150, zorder=5, label='Step 4: Preemptive Drop Decision')
    
    # Adding an explicit visual flow arrow between R3 obs and R4 pred to show the mechanism
    ax.annotate('', xy=(4, 77.7), xytext=(3, 106.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, linestyle='--'), zorder=4)
                
    ax.annotate('', xy=(7, 71.2), xytext=(6, 25.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, linestyle='--'), zorder=4)
    
    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Completion / Predicted Safe Completion Time (s)')
    ax.set_title('Adaptive Decision Mechanism (Hospital_C)')
    ax.set_xticks(rounds)
    ax.set_xlim(0.5, 10.5)
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Text annotation as requested
    ax.text(0.5, -0.15, "Observed and predicted completion behavior for Hospital_C in Experiment 7", 
            ha='center', va='center', transform=ax.transAxes, fontsize=11, fontstyle='italic')
    
    save_fig(fig, 'fig3_adaptive_decision_response')
    plt.close(fig)

# ==================================================
# FIGURE 4 — NONSTATIONARY HOSPITAL C (Unchanged)
# ==================================================
def plot_fig4():
    # Exactly like the old plot_fig4_5
    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    obs_time = [34.6, 33.9, 106.3, np.nan, np.nan, 25.3, np.nan, np.nan, np.nan, np.nan]
    pred_time = [34.6, 34.4, 34.4, 77.7, np.nan, np.nan, 71.2, np.nan, np.nan, np.nan] # Kept original for Fig 4
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot observations
    ax.plot(rounds, obs_time, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8, label='Observed Completion Time')
    
    # Plot predicted/EMA
    ax.plot(rounds, pred_time, marker='s', color='#ff7f0e', linestyle='--', linewidth=2, markersize=8, label='Predicted Completion (EMA + Dev)')
    
    # Hard deadline
    ax.axhline(y=60, color='red', linestyle=':', linewidth=2, label='Hard Deadline (60s)')
    
    # Mark dropped rounds
    ax.scatter([4, 7], [1.2, 1.1], color='red', marker='X', s=150, zorder=5, label='Preemptively Dropped')
    
    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Exp 7: Adaptive Prediction of Nonstationary Delay (Hospital_C)')
    ax.set_xticks(rounds)
    ax.set_xlim(0.5, 10.5)
    
    # Annotate artificial delays
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
    
    print("Figures generated successfully.")
except Exception as e:
    print(f"Error generating figures: {e}")
