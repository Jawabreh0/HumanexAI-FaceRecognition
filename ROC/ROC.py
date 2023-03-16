import matplotlib.pyplot as plt

# Define the data
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
tpr = [150/178, 154/178, 164/178, 171/178, 174/178, 176/178, 173/178]
fpr = [28/178, 24/178, 14/178, 7/178, 4/178, 1/178, 5/178]

# Create a figure with two axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the TPR on the top axis
ax1.plot(thresholds, tpr, marker='o', markersize=5, color="blue")
ax1.set_ylabel('True Positive Rate')

# Plot the FPR on the bottom axis
ax2.plot(thresholds, fpr, marker='o', markersize=5, color="red")
ax2.set_xlabel('Threshold')
ax2.set_ylabel('False Positive Rate')

# Show the plot
plt.show()
