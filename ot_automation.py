import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc

# Paths
project_root = r'C:\Jupyter Notebook\UOB labs\DSMP\Sapienza_data\progetto_bristol'
dataset_path = os.path.join(project_root, 'dataset')
hr_source_path = os.path.join(project_root, 'hr files', 'test')
classified_path = os.path.join(project_root, 'classified_dataset.csv')
model_dir = os.path.join(project_root, 'models')
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

# Hydropathy scales
scales = [
    "hr_ARGP820101.txt", "hr_BLAS910101.txt", "hr_CASG920101.txt", "hr_CIDH920105.txt",
    "hr_EISD840101.txt", "hr_EISD860102.txt", "hr_ENGD860101.txt", "hr_FASG890101.txt",
    "hr_GOLD730101.txt", "hr_HOPT810101.txt", "hr_JOND750101.txt", "hr_JURD980101.txt",
    "hr_KIDA850101.txt", "hr_KUHL950101.txt", "hr_KYTJ820101.txt", "hr_LEVM760101.txt",
    "hr_NADH010101.txt", "hr_NADH010102.txt", "hr_NADH010103.txt", "hr_NADH010104.txt",
    "hr_NADH010105.txt", "hr_NADH010106.txt", "hr_NADH010107.txt", "hr_PONP930101.txt",
    "hr_PRAM900101.txt", "hr_WOLR790101.txt", "hr_ZIMJ680101.txt", "hr_L_hydrophobicity_scale.txt"
]

def read_and_norm(dataset_path, type_, file_name):
    with open(f'{dataset_path}/{type_}/{file_name}.txt', 'rb') as f:
        matrix = [[float(x) for x in line.split()] for line in f]
    matrix = np.array(matrix)
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())

def load_test_dataset():
    classification = np.loadtxt(f'{dataset_path}/test/classification.txt').reshape(-1,1)

    with open(f'{dataset_path}/test/hr.txt', "r") as file:
        hr = [list(map(float, line.split())) for line in file if len(line.split()) != 9]
    hr = np.array(hr)

    shape = read_and_norm(dataset_path, 'test', 'shape')
    el = read_and_norm(dataset_path, 'test', 'el')
    dist = read_and_norm(dataset_path, 'test', 'dist')

    data_X = np.array([p for p in zip(shape, dist, el, hr)])
    data_X = data_X.reshape(data_X.shape[0], data_X.shape[1], data_X.shape[2], 1)

    return data_X, classification

all_metrics = []

# Load classified dataset once
classified_df = pd.read_csv(classified_path)

for scale in scales:
    scale_name = scale.replace(".txt", "")
    print(f"\nEvaluating {scale_name}...")

    # Copy the corresponding hr.txt
    shutil.copy(os.path.join(hr_source_path, scale), os.path.join(dataset_path, 'test', 'hr.txt'))

    # Load model
    model_path = os.path.join(model_dir, f"my_model_{scale_name}.keras")
    if not os.path.exists(model_path):
        print(f"Model not found for {scale_name}")
        continue
    model = load_model(model_path)

    # Load data
    testX, testy = load_test_dataset()
    pred_probs = model.predict(testX)

    # ROC & Optimal Threshold
    fpr, tpr, thresholds = roc_curve(testy, pred_probs)
    roc_auc = auc(fpr, tpr)
    J_scores = tpr - fpr
    optimal_idx = np.argmax(J_scores)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (pred_probs >= optimal_threshold).astype(int)

    # Accuracy and F1
    accuracy = accuracy_score(testy, pred_labels)
    f1 = f1_score(testy, pred_labels)

    # Save ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Threshold = {optimal_threshold:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {scale_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{scale_name}_roc.png"))
    plt.close()

    # Merge with classified dataset
    df = pd.concat([classified_df.copy(), pd.DataFrame({
        'True_Label': testy.flatten(),
        'Probability': pred_probs.flatten(),
        'Predicted_Label': pred_labels.flatten()
    })], axis=1)
    
    df["residue_pair"] = df["resA_type"] + "-" + df["resB_type"]

    # Compute F1 per pair
    f1_scores = df.groupby("residue_pair").apply(
        lambda g: f1_score(g["True_Label"], g["Predicted_Label"], zero_division=0)
    ).reset_index(name="F1-score")

    # ... (everything before remains unchanged)

    # Save F1-barplot and F1 CSV
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=f1_scores, x="residue_pair", y="F1-score", palette="tab20")
    plt.title(f"F1-score by Residue Pair - {scale_name}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{scale_name}_f1_bar.png"))
    plt.close()

    # Save F1-scores as CSV for PCA
    f1_csv_path = os.path.join(results_dir, f"{scale_name}.csv")
    f1_scores.to_csv(f1_csv_path, index=False)


    # Append metrics for this scale
    all_metrics.append({
        'Scale': scale_name,
        'Optimal_Threshold': optimal_threshold,
        'Accuracy': accuracy,
        'F1_Score': f1
    })

# Save all metrics to CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(results_dir, "hydropathy_scale_metrics.csv"), index=False)

print("\nAll scales evaluated. Metrics and plots saved in 'results' folder.")
