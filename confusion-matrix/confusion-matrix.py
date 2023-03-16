import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the confusion matrix
confusion_matrix = [[49, 0, 0, 1], [0, 34, 0, 0], [0, 0, 44, 0], [0, 0, 0, 50]]

# Define the class labels
labels = ["Ahmad", "Milena", "Ravilya", "Unknown"]

# Create a DataFrame from the confusion matrix and labels
df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

# Plot the heatmap using Seaborn
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="d")

# Add axis labels and a title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for HumanexAI Face Recognition System")

# Show the plot
plt.show()
