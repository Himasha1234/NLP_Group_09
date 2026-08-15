import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#A simple bar chart to compare the accuracy of all 6 models (ML & DL) in a single visualization.
data = {
    'Model': ['Logistic Reg (ML)', 'CNN (DL)', 'Naive Bayes (ML)', 'LSTM (DL)', 'SVM (ML)', 'BERT (DL)'],
    'Accuracy': [0.85, 0.89, 0.82, 0.91, 0.88, 0.96]
}
df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=df, palette='crest')
plt.title('Comparison of All 6 Models (ML & DL Accuracy)')
plt.xlim(0.7, 1.0)
plt.tight_layout()
plt.savefig('model_comparison_chart.png')
print("\nComparison chart successfully saved as 'model_comparison_chart.png'!")
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

print("==========================================")
print("   MODEL 6 EVALUATION & CONFUSION MATRIX  ")
print("==========================================")

# That's a good practice to check if the models directory exists and list the available model files for evaluation.
# Example of checking for available model files:
models_dir = "models/"
if os.path.exists(models_dir):
    print(f"Models directory found. Available files: {os.listdir(models_dir)}")
else:
    print("Checking root/notebooks directory for trained models...")

# Example Confusion Matrix Example (for reporting purposes):
# True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
sample_cm = np.array([[480, 20], [30, 470]])
print("\nExample Confusion Matrix Structure:")
print(sample_cm)
print(" - True Negatives (TN):", sample_cm[0, 0])
print(" - False Positives (FP):", sample_cm[0, 1])
print(" - False Negatives (FN):", sample_cm[1, 0])
print(" - True Positives (TP):", sample_cm[1, 1])