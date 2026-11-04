from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tensorboard.backend.event_processing import event_accumulator


# (experiment folder, plot label) pairs, e.g. ("../exp/shapenet/scratch_airplane", "GEKey")
event_files = [("../exp/<dataset>/<exp_name>", "<label>")]

tags = ['val/mIoU-0.10', 'val/mIoU-0.01', 'val/DAS']
data = {tag: {} for tag in tags}

for event_file, label in event_files:
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    scalars = ea.Tags().get('scalars', [])

    for tag in tags:
        if tag not in scalars:
            print(f"Tag '{tag}' not found in {event_file}")
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag][label] = (steps, values)


bright_blue, bright_green, bright_red = '#3299CD', '#CD3232', '#32CD32'
custom_colors = cycle([bright_blue, bright_green, bright_red])

for tag in tags:
    plt.figure(figsize=(11, 7))
    for label, (steps, values) in data[tag].items():
        ylabel = tag.split('/')[-1]
        if tag.startswith("val/mIoU"):
            values = [v * 100 for v in values]

        plt.plot(steps, values, label=label, color=next(custom_colors))
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.tick_params(axis='both', labelsize=20)
    plt.gca().xaxis.set_major_locator(MultipleLocator(20))
    plt.grid(False)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.show()
