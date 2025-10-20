import matplotlib.pyplot as plt
import numpy as np

# Thresholds and corresponding evaluation metrics
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
accuracy =   [0.8300, 0.8325, 0.8400, 0.8425, 0.8500, 0.8600, 0.8600, 0.8575]
precision =  [0.7519, 0.7548, 0.7636, 0.7665, 0.7756, 0.7880, 0.7903, 0.7895]
recall =     [0.9850, 0.9850, 0.9850, 0.9850, 0.9850, 0.9850, 0.9800, 0.9750]
f1 =         [0.8528, 0.8547, 0.8603, 0.8621, 0.8678, 0.8756, 0.8750, 0.8725]

# Create figure
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6), dpi=200)

# Plot performance metrics across thresholds
plt.plot(thresholds, accuracy, marker='o', label='Accuracy', color="#5F81C8", linewidth=2)
plt.plot(thresholds, precision, marker='s', label='Precision', color="#66BDB3", linewidth=2)
plt.plot(thresholds, recall, marker='^', label='Recall', color="#427764", linewidth=2)
plt.plot(thresholds, f1, marker='D', label='F1 Score', color="#77A2DD", linewidth=2)

# Highlight the best threshold (0.65)
best_threshold = 0.65
best_f1 = 0.8756
plt.scatter(best_threshold, best_f1, color='#7A5DC7', s=80, zorder=5, label='Optimal Threshold')
plt.text(best_threshold + 0.005, best_f1 + 0.005, 'F1 = 0.8756', fontsize=10, color='#7A5DC7')

# Style settings
plt.title('Threshold Optimization Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0.7, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout(pad=3)

# Display figure
plt.show()
