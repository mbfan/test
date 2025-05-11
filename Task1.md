Code Explanation: 
Task I:
Filename: BRISTOLcnn.py
Purpose:
This script defines and trains a Convolutional Neural Network (CNN) using TensorFlow/Keras to predict core interacting residues in protein-protein interactions. It processes multiple types of biochemical and structural features and saves the trained model.
What the Code Does:
1. Data Loading and Preprocessing
• Function: read_and_norm()
Reads a .txt matrix (such as shape, electrostatics, or distance), then 
normalizes the values to the [0, 1] range.
• Function: load_full_dataset(type_)
Loads the features and labels from either the train or test folder:
o Features: shape, distance, electrostatics (el), and hydropathy values 
(hr)
o Labels: binary classification (interacting or not interacting)
o It also removes rows with errors (specifically, rows in hr.txt with 9 
columns, which indicate corrupt or invalid data).
• The features are formatted as a 4D array: (samples, features, values, 1) to match the input shape required by the 2D CNN.
2. Model Architecture
• Function: evaluate_model_2dconv(...)
o Architecture:
▪ Two 2D convolution layers with ReLU activation and dropout layers to prevent overfitting
▪ Two dense layers with 30 neurons each and dropout
▪ Final output layer uses a sigmoid activation function for binary 
classification
o Training:
▪ Loss function: binary crossentropy
▪ Optimizer: Adam
▪ Trained for up to 300 epochs with early stopping based on 
validation accuracy
o Evaluation:
▪ Evaluates the model on the test set and returns accuracy
▪ Saves prediction results (predicted label, probability, true label) 
in a DataFrame
▪ Optionally saves the trained model if a filename is provided
3. Model Execution
• Function: run_experiment()
Runs the training and evaluation process:
o Loads the training and testing datasets
o Trains the model for the specified number of repetitions
o Reports the test accuracy for each run and summarizes the results with 
the mean and standard deviation
• Main Execution Block
When the script is run directly, it executes run_experiment().
Output
• Printed training progress and accuracy scores for each run
• Summary of mean and standard deviation of test accuracy
• Optionally saves a trained model file if configured
Filename: evaluate_all_scales.py
Purpose:
This script automates the evaluation of CIRNet across 27 different hydropathy scales. It copies the corresponding hr.txt files into the main dataset folder, runs the model, and logs the accuracy of each scale into a CSV file for later comparison.
What the Code Does:
1. Path Setup
• Defines the key paths:
o project_root: Main project directory.
o dataset_path: Points to the dataset folder containing train and test
subfolders.
o hr_source_path: Folder containing all the hydropathy scales, with 
separate train and test subfolders.
o csv_path: Destination file for logging results 
(hydropathy_scale_results.csv).
2. Hydropathy Scale Management
• Maintains a list of all 27 hydropathy scale filenames to be evaluated.
• Checks if results have already been recorded in the CSV. If so, it avoids re￾evaluating already completed scales.
3. Model Evaluation Loop
For each hydropathy scale:
• Step 1: Copies the corresponding train/hr.txt and test/hr.txt files into the 
dataset folders used by CIRNet.
• Step 2: Calls the run_experiment() function from BRISTOLcnn to train and evaluate CIRNet with the new scale.
o The trained model is saved as a .keras file, uniquely named by scale.
• Step 3: Appends the scale name and resulting accuracy to a results list.
4. Error Handling
• If a scale causes an exception (e.g., missing files or evaluation errors), the error is caught and logged in the CSV with Accuracy marked as 'Error'.
5. Save Results
• After each scale is evaluated, the current results are saved to the CSV to ensure progress is not lost if the script is interrupted.
Output
• Terminal messages indicating which scales are evaluated or skipped
• A CSV file (hydropathy_scale_results.csv) containing the scale name and its 
corresponding test accuracy (or error)
• Saved model files for each scale in the project root directory
Dependencies
• os and shutil for file operations
• pandas for data handling and CSV output
• run_experiment imported from BRISTOLcnn.py, which handles model training and evaluation
Filename: ot.ipynb
Purpose:
This script loads a trained CIRNet model, evaluates it on the test dataset, calculates the optimal threshold using the ROC curve, and then computes per-class F1-scores grouped by chemical residue pair types. It also generates visualizations and saves results for further analysis.
What the Code Does:
1. Model and Dataset Loading
• Loads a pre-trained CIRNet model from the specified path.
• Reads and normalizes the test dataset from the dataset/test folder, including:
o Shape matrix
o Electrostatic potential (el)
o Distance matrix
o Hydropathy values from hr.txt
2. Model Prediction and Threshold Optimization
• Predicts interaction probabilities using the trained model.
• Uses the ROC curve and Youden's J statistic (TPR - FPR) to determine the optimal classification threshold.
• Plots the ROC curve, marking the optimal threshold.
3. Performance Evaluation
• Applies the optimal threshold to convert probabilities into binary predictions.
• Computes:
o Accuracy
o F1-score
• Saves a CSV (prediction_results.csv) with the true labels, predicted labels, and probabilities.
4. Residue Pair Classification
• Reads residue names from dataset/test/dataset.txt.
• Maps amino acids to chemical types:
o H = Hydrophobic
o P = Polar
o C = Charged
• Creates new columns resA_type and resB_type for the mapped types and 
saves to classified_dataset.csv.
5. Per Residue-Pair F1-score Analysis
• Combines the classification and prediction results using pandas.
• Creates a residue_pair column (e.g., "H-P").
• Calculates the F1-score for each unique residue pair type using a helper function.
• Stores these scores in a DataFrame and plots them as a bar chart.
Output:
• Visuals:
o ROC curve showing model performance and optimal threshold.
o Bar plot showing F1-scores for each residue pair type.
• Files:
o prediction_results.csv: Contains probabilities and labels for all test samples.
o classified_dataset.csv: Includes residue classification (H, P, C) for each protein pair.
• Console Output:
o Optimal threshold
o Overall accuracy and F1-score
o Per-pair F1-score table
Dependencies:
• NumPy, Pandas, Matplotlib, Seaborn
• TensorFlow/Keras for model loading and inference
• Scikit-learn for metrics like ROC curve and F1-score
Filename: ot_automation.py
Purpose:
To automatically evaluate CIRNet's performance on multiple hydropathy scales, computing optimal thresholds, global accuracy/F1, and per-residue-pair F1 scores. It saves metrics, ROC curves, and bar plots for each scale.
Script Workflow:
1. Setup
• Sets root directories for:
o Dataset, classified dataset, model files, and output results.
• Defines the list of hydropathy scales (28 total).
• Ensures a clean results/ directory exists to store outputs.
2. Loop Over Hydropathy Scales
For each hydropathy scale:
• Step 1: Copy the current hr.txt file to dataset/test/ so CIRNet reads the correct features.
• Step 2: Load the corresponding trained model.
o If the model file doesn't exist, it skips to the next scale.
• Step 3: Load and normalize the test dataset.
• Step 4: Predict interaction probabilities with CIRNet.
• Step 5: Compute ROC curve and optimal threshold using Youden's J statistic.
• Step 6: Classify predictions and compute:
o Accuracy
o Global F1-score
• Step 7: Merge predictions with the classified residue pairs (loaded once at the 
top).
• Step 8: Compute F1-score for each residue pair type (H-H, H-P, etc.).
• Step 9: Save:
o ROC curve plot
o Per-pair F1 bar plot
o CSV of F1-scores
o Aggregate metrics for this scale
3. Save Aggregated Results
• Creates a final CSV summarizing all scales:
o Optimal threshold
o Accuracy
o F1-score
Output Files (per scale, in results/):
• ScaleName_roc.png: ROC curve
• ScaleName_f1_bar.png: Bar plot of F1 per residue pair
• ScaleName.csv: Per-pair F1-scores for use in PCA
• hydropathy_scale_metrics.csv: Summary metrics for all 28 scales
Key Advantages:
• Scalable: Easily extendable to new scales or models.
• Reproducible: Ensures consistency in evaluation.
• Analysis-Ready: Directly feeds PCA or comparison studies.
• Clean and Modular: Well-organized code with reusable functions.
