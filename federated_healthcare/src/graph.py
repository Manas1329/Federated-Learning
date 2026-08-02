# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # -------------------------------
# # Paths
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# METRICS_FILE = os.path.join(BASE_DIR, "dashboard", "metrics.csv")
# OUTPUT_DIR = os.path.join(BASE_DIR, "dashboard", "plots")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -------------------------------
# # Read CSV
# # -------------------------------
# df = pd.read_csv(METRICS_FILE)

# # Convert accuracy to percentage if stored as decimal
# if df["Accuracy"].max() <= 1:
#     df["Accuracy"] *= 100

# print(df)

# # -------------------------------
# # Professional Style
# # -------------------------------
# plt.style.use("ggplot")

# plt.figure(figsize=(12,7))

# plt.plot(
#     df["Round"],
#     df["Accuracy"],
#     color="royalblue",
#     marker="o",
#     markersize=9,
#     linewidth=3,
#     label="Global Accuracy"
# )

# # Show values on each point
# for x, y in zip(df["Round"], df["Accuracy"]):
#     plt.annotate(
#         f"{y:.2f}%",
#         (x, y),
#         textcoords="offset points",
#         xytext=(0,10),
#         ha="center",
#         fontsize=10,
#         fontweight="bold"
#     )

# # Highlight best accuracy
# best_idx = df["Accuracy"].idxmax()

# plt.scatter(
#     df.loc[best_idx, "Round"],
#     df.loc[best_idx, "Accuracy"],
#     s=180,
#     color="red",
#     label="Best Accuracy"
# )

# plt.title(
#     "Global Model Accuracy Across Federated Learning Rounds",
#     fontsize=18,
#     fontweight="bold"
# )

# plt.xlabel(
#     "Federated Learning Round",
#     fontsize=14,
#     fontweight="bold"
# )

# plt.ylabel(
#     "Accuracy (%)",
#     fontsize=14,
#     fontweight="bold"
# )

# plt.xticks(df["Round"], fontsize=11)
# plt.yticks(fontsize=11)

# plt.ylim(70,100)

# plt.grid(True, linestyle="--", alpha=0.6)

# plt.legend(fontsize=12)

# plt.tight_layout()

# plt.savefig(
#     os.path.join(OUTPUT_DIR, "accuracy_new.png"),
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# print("\nAccuracy graph saved successfully!")
# print("Location :", OUTPUT_DIR)

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_file = os.path.join(BASE_DIR, "dashboard", "metrics.csv")

plots_dir = os.path.join(BASE_DIR, "dashboard", "plots")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(csv_file)

df["Accuracy"] = df["Accuracy"] * 100

# ---------------- Accuracy Graph ----------------
plt.figure(figsize=(8,5))

plt.plot(df["Round"], df["Accuracy"], marker="o", linewidth=2)

for x, y in zip(df["Round"], df["Accuracy"]):
    plt.text(x, y+0.3, f"{y:.2f}%", ha="center", fontsize=8)

best_idx = df["Accuracy"].idxmax()

plt.scatter(df.loc[best_idx, "Round"],
            df.loc[best_idx, "Accuracy"],
            s=120)

plt.title("Global Model Accuracy")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.savefig(os.path.join(plots_dir, "accuracy.png"), dpi=300)

plt.close()


# ---------------- Loss Graph ----------------
plt.figure(figsize=(8,5))

plt.plot(df["Round"], df["Loss"], marker="o", linewidth=2)

for x, y in zip(df["Round"], df["Loss"]):
    plt.text(x, y+0.01, f"{y:.3f}", ha="center", fontsize=8)

plt.title("Global Model Loss")
plt.xlabel("Communication Round")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig(os.path.join(plots_dir, "loss.png"), dpi=300)

plt.close()


# ---------------- Combined Graph ----------------
plt.figure(figsize=(9,5))

plt.plot(df["Round"], df["Accuracy"], marker="o", label="Accuracy (%)")
plt.plot(df["Round"], df["Loss"]*100, marker="s", label="Loss ×100")

plt.title("Accuracy vs Loss")
plt.xlabel("Communication Round")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(plots_dir, "accuracy_loss.png"), dpi=300)

plt.close()

print("Graphs generated successfully.")