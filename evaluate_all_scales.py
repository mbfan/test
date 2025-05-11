import os
import shutil
import pandas as pd
from BRISTOLcnn import run_experiment

# Paths setup
project_root = r'C:\Jupyter Notebook\UOB labs\DSMP\Sapienza_data\progetto_bristol'
dataset_path = os.path.join(project_root, 'dataset')
hr_source_path = os.path.join(project_root, 'hr files')
csv_path = os.path.join(project_root, 'hydropathy_scale_results.csv')

# List of 27 hydropathy scale names
scales = [
    "hr_ARGP820101.txt", "hr_BLAS910101.txt", "hr_CASG920101.txt", "hr_CIDH920105.txt",
    "hr_EISD840101.txt", "hr_EISD860102.txt", "hr_ENGD860101.txt", "hr_FASG890101.txt",
    "hr_GOLD730101.txt", "hr_HOPT810101.txt", "hr_JOND750101.txt", "hr_JURD980101.txt",
    "hr_KIDA850101.txt", "hr_KUHL950101.txt", "hr_KYTJ820101.txt", "hr_LEVM760101.txt",
    "hr_NADH010101.txt", "hr_NADH010102.txt", "hr_NADH010103.txt", "hr_NADH010104.txt",
    "hr_NADH010105.txt", "hr_NADH010106.txt", "hr_NADH010107.txt", "hr_PONP930101.txt",
    "hr_PRAM900101.txt", "hr_WOLR790101.txt", "hr_ZIMJ680101.txt", "hr_L_hydrophobicity_scale.txt"
]

# Load existing results if available
if os.path.exists(csv_path):
    results_df = pd.read_csv(csv_path)
    done_scales = set(results_df['Scale'])
    results = results_df.to_dict('records')
else:
    done_scales = set()
    results = []

for scale in scales:
    if scale in done_scales:
        print(f"Skipping {scale}, already evaluated.")
        continue

    print(f"Running CIRNet with scale: {scale}")

    # Source paths
    train_hr = os.path.join(hr_source_path, 'train', scale)
    test_hr = os.path.join(hr_source_path, 'test', scale)

    # Destination paths
    train_dest = os.path.join(dataset_path, 'train', 'hr.txt')
    test_dest = os.path.join(dataset_path, 'test', 'hr.txt')

    try:
        shutil.copy(train_hr, train_dest)
        shutil.copy(test_hr, test_dest)

        model_filename = os.path.join(project_root, f"my_model_{scale.replace('.txt', '')}.keras")
        accuracy = run_experiment(model_filename=model_filename)


        results.append({'Scale': scale, 'Accuracy': accuracy})
        print(f"{scale}: Accuracy = {accuracy}")

    except Exception as e:
        print(f"Failed to evaluate {scale}: {e}")
        results.append({'Scale': scale, 'Accuracy': 'Error'})

    # Save progress after each scale
    pd.DataFrame(results).to_csv(csv_path, index=False)

print(f"All done! Final results saved to {csv_path}")

