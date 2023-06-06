import matplotlib.pyplot as plt
import numpy as np

calculation_method = ("GPU with\nsaving fields", "GPU without\nsaving fields", "CPU with\nsaving fields", "CPU without\nsaving fields")
calculation_times = {
    'Imaging Region 100x150': (115.0, 115.0, 307.0, 310),
    'Imaging Region 350x180': (115.0, 113.0, 309.5, 308.3),
}

x = np.arange(len(calculation_method))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0.5

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in calculation_times.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Calculation time (s)')
ax.set_title('Wave solver calculation times')
ax.set_xticks(x + width, calculation_method)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 380)
ax.grid(linestyle='--', linewidth=0.5)

plt.show()
