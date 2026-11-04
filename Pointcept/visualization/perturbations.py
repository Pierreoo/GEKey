"""
Plot jitter, downsampling and corresponding IoU values across different thresholds for the Airplane category
"""

import seaborn as sns
import matplotlib.pyplot as plt


colors = sns.color_palette("deep", 10)

steps_jitter = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
values_jitter = [87.0, 84.0, 80.4, 82.0, 78.7, 83.4, 80.0, 82.1, 81.3, 85.6, 65.9]
values_jitter_9 = [81.9, 78.8, 72.9, 76.2, 72.2, 77.1, 74.3, 74.8, 74.7, 78.8, 58.5]
values_jitter_8 = [75.6, 72.5, 63.6, 69.8, 64.8, 69.8, 67.9, 66.1, 68.2, 69.0, 50.4]
values_jitter_7 = [67.3, 65.2, 53.9, 62.9, 57.8, 62.3, 61.1, 57.5, 61.0, 57.2, 41.5]
values_jitter_6 = [58.2, 57.4, 44.3, 55.1, 50.7, 54.9, 53.8, 49.8, 54.4, 46.3, 32.9]
values_jitter_5 = [47.6, 49.7, 35.3, 48.3, 44.4, 47.7, 46.4, 42.3, 47.5, 36.7, 24.8]
values_jitter_4 = [38.0, 43.3, 27.7, 42.1, 38.2, 41.4, 38.4, 34.9, 41.2, 28.9, 16.6]
values_jitter_3 = [29.5, 36.1, 21.0, 35.2, 31.1, 35.0, 29.9, 28.1, 33.4, 22.0, 10.0]
values_jitter_2 = [18.9, 25.3, 13.6, 24.5, 20.0, 24.6, 18.2, 19.7, 21.1, 14.4, 5.4]
values_jitter_1 = [7.6, 11.2, 5.2, 9.8, 6.9, 10.7, 6.2, 8.9, 8.0, 5.9, 2.1]

steps_downsample = [0, 2, 4, 8, 16, 32]
values_downsample = [87.0, 86.4, 84.8, 81.1, 60.5, 73.7]
values_downsample_9 = [81.9, 80.1, 78.2, 73.7, 52.8, 66.2]
values_downsample_8 = [75.6, 71.9, 70.2, 64.8, 42.2, 57.4]
values_downsample_7 = [67.3, 62.7, 61.6, 55.4, 31.2, 46.0]
values_downsample_6 = [58.2, 52.2, 51.8, 46.7, 21.2, 34.5]
values_downsample_5 = [47.6, 41.2, 42.1, 38.0, 13.8, 24.0]
values_downsample_4 = [38.0, 30.4, 33.1, 28.8, 8.6, 14.5]
values_downsample_3 = [29.5, 21.0, 23.7, 19.3, 4.9, 7.0]
values_downsample_2 = [18.9, 11.5, 14.0, 10.3, 2.3, 2.7]
values_downsample_1 = [7.6, 3.5, 5.3, 3.8, 1.0, 1.0]

all_jitter_values = [
    values_jitter, values_jitter_9, values_jitter_8, values_jitter_7,
    values_jitter_6, values_jitter_5, values_jitter_4, values_jitter_3,
    values_jitter_2, values_jitter_1
]

all_downsample_values = [
    values_downsample, values_downsample_9, values_downsample_8, values_downsample_7,
    values_downsample_6, values_downsample_5, values_downsample_4, values_downsample_3,
    values_downsample_2, values_downsample_1
]

# Create figure with 2 rows, 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Top plot: Gaussian Noise
for i, values in enumerate(all_jitter_values):
    alpha = 1.0 if i == 0 else 1.0
    ax1.plot(steps_jitter, values, marker='o', color=colors[i], label=f'0.{i:02d}', alpha=alpha)

ax1.set_xlabel('Gaussian Noise Scale', fontsize=20)
ax1.set_ylabel('mIoU (%)', fontsize=20)
ax1.set_xticks(steps_jitter)
ax1.set_xticklabels(steps_jitter, fontsize=15, rotation=45, ha='right')
ax1.set_yticks(range(0, 101, 20))
ax1.set_yticklabels(range(0, 101, 20), fontsize=15)
ax1.set_ylim(0, 100)
ax1.grid(True)
ax1.set_facecolor('#E0E0E0')

# Bottom plot: Downsampling
indices = list(range(len(steps_downsample)))
for i, values in enumerate(all_downsample_values):
    alpha = 1.0 if i == 0 else 1.0
    ax2.plot(indices, values, marker='o', color=colors[i], label=f'0.{i:02d}', alpha=alpha)

ax2.set_xlabel('Downsampling Ratio', fontsize=20)
ax2.set_ylabel('mIoU (%)', fontsize=20)
ax2.set_xticks(indices)
ax2.set_xticklabels(steps_downsample, fontsize=15)
ax2.set_yticks(range(0, 101, 20))
ax2.set_yticklabels(range(0, 101, 20), fontsize=15)
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.set_facecolor('#E0E0E0')

# Add a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
thresholds = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
labels = [f"{t:.2f}" for t in thresholds]
fig.legend(handles, labels, fontsize=14, title="Threshold", title_fontsize=16,
           loc='upper center', bbox_to_anchor=(0.55, 1.02), ncol=5, frameon=True)
fig.legend(handles, labels, fontsize=14, title="Threshold", title_fontsize=16,
           loc='upper center', bbox_to_anchor=(0.55, 0.47), ncol=5, frameon=True)
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.8)
plt.show()
fig.savefig("perturbations.svg", format="svg", bbox_inches='tight')
