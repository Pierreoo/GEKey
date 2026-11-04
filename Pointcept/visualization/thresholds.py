"""
Plot IoU values across different thresholds for each category in the KeyPointNet dataset
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


blue, green, orange, purple, red, brown, pink, gray, cyan, magenta, yellow, lime, teal = (
    '#000099', '#006619', '#FF9900', '#660099', '#CC0000',
    '#996633', '#FF66CC', '#999999', '#00CCCC', '#CC00CC',
    '#FFD700', '#32CD32', '#008080'
)
colors = [blue, green, orange, purple, red, brown, pink, gray, cyan, magenta, yellow, lime, teal]

thresholds = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

categories = ['Airplane', 'Bed', 'Bottle', 'Cap', 'Car', 'Chair', 'Guitar',
              'Helmet', 'Knife', 'Motorcycle', 'Mug', 'Table', 'Vessel']

category_values = [
    [86.5, 81.0, 73.7, 65.8, 58.4, 51.1, 44.1, 37.0, 28.7, 15.0],  # airplane
    [51.9, 48.8, 44.9, 40.3, 36.1, 32.3, 28.5, 23.9, 19.0, 12.4],  # bed
    [46.1, 38.4, 31.2, 24.9, 18.9, 13.4,  7.9,  4.2,  2.1,  1.0],  # bottle
    [32.1, 26.9, 22.6, 18.5, 15.4, 12.7,  9.2,  7.1,  3.4,  1.3],  # cap
    [43.9, 33.6, 23.8, 15.3,  9.1,  5.3,  2.9,  1.3,  0.6,  0.2],  # car
    [65.6, 60.5, 54.9, 49.6, 44.4, 38.5, 32.3, 25.3, 16.7,  8.7],  # chair
    [71.6, 65.6, 58.3, 50.9, 43.5, 36.0, 29.1, 22.7, 16.9,  9.7],  # guitar
    [36.5, 30.9, 24.6, 18.9, 14.1, 10.0,  7.1,  4.1,  2.3,  1.1],  # helmet
    [53.0, 49.9, 46.2, 42.5, 38.2, 33.2, 28.1, 23.0, 17.3, 10.9],  # knife
    [66.2, 58.4, 50.1, 42.3, 33.9, 25.2, 18.2, 13.6,  9.9,  5.4],  # motorcycle
    [28.6, 23.1, 19.0, 14.8, 11.5,  8.3,  5.7,  3.8,  2.3,  1.3],  # mug
    [70.7, 66.8, 62.5, 58.3, 53.5, 48.2, 41.9, 33.4, 23.1, 12.0],  # table
    [55.8, 47.9, 39.3, 30.8, 22.9, 15.9, 11.0,  7.6,  5.3,  3.7],  # vessel
]

# Main plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, (category, values) in enumerate(zip(categories, category_values)):
    ax.plot(thresholds, values, marker='o', color=colors[i], label=category, linewidth=2)

ax.set_xlabel('Threshold', fontsize=20)
ax.set_ylabel('mIoU (%)', fontsize=20)
ax.set_xticks(thresholds)
ax.set_xticklabels([f'{t:.2f}' for t in thresholds], fontsize=15, rotation=45, ha='right')
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels(range(0, 101, 20), fontsize=15)
ax.set_ylim(0, 100)
ax.invert_xaxis()
ax.grid(True)
ax.set_facecolor('#E0E0E0')

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    fontsize=12,
    title="Category",
    title_fontsize=14,
    loc='center left',
    bbox_to_anchor=(0.82, 0.665),
    ncol=1,
    frameon=True
)

fig.savefig("threshold.svg", format="svg", bbox_inches='tight')