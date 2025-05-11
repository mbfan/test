Task II:
PCA Script Summary
Filename: comparison.ipynb
Summary of Code and Outputs
This code performs a comprehensive evaluation of 27 hydropathy scales based on their performance in predicting core interacting residues using CIRNet. The evaluation includes performance aggregation, visualization, and ranking to identify potentially superior hydropathy scales.
1. Loading and Preparing Data
• The script starts by loading per-scale F1-score CSV files from the resultsdirectory.
• Each CSV contains F1-scores for different residue-residue pairs predicted by CIRNet using a specific hydropathy scale.
• All data is combined into a single dataframe for further analysis.
2. Principal Component Analysis (PCA)
• PCA is applied to the F1-scores of all residue pairs across all scales.
• The aim is to reduce dimensionality and visualize patterns or clusters among the scales based on prediction behavior.
• The output is a 2D scatter plot (pca_hydropathy_scales.png) where each point represents a scale, allowing us to observe similarity or divergence among them.
3. F1 Score Bar Plot
• A bar chart (f1_score_per_scale.png) is created to visualize the maximum F1 score achieved by each scale.
• This allows quick identification of scales with the best single-class 
performance.
4. ROC Curve Visualization
• All ROC curve images saved during earlier evaluations are displayed in a grid (all_roc_curves_grid.png).
• This visual overview helps assess how well each scale distinguishes between positive and negative classes across residue pairs.
5. Heatmap by Residue Chemical Class
• The F1-scores are grouped based on chemical classes (e.g. Hydrophobic Hydrophobic, Polar-Charged, etc.).
• A heatmap (f1_by_chemical_class_heatmap.png) shows how each hydropathy scale performs across these broader categories.
• This helps answer whether some scales are more suited to identifying 
interactions between chemically similar or distinct residue types.
6. Ranking Hydropathy Scales Using Composite Score
• A separate CSV (hydropathy_scale_metrics.csv) containing global metrics like accuracy, F1-score, and optimal threshold is loaded.
• For each scale, the maximum per-class F1-score is also added.
• All metrics are normalized and averaged to compute a Composite_Score
reflecting overall performance.
• The top 5 scales based on this score are visualized in a bar chart 
(top5_composite_scores.png).
• The full ranked list is saved as ranked_hydropathy_scales.csv.
Purpose 
This pipeline supports scale comparison both visually and quantitatively. PCA reveals high-level differences in behavior. The ROC curves and F1 score heatmaps 
allow class-wise performance insight, and the composite score gives an overall performance ranking. Together, these outputs help identify hydropathy scales that 
are particularly effective for CIRNet’s task and reveal whether certain scales are better at predicting specific types of residue interactions.
Metrics Used for Evaluating Hydropathy Scales
To assess the performance of CIRNet across different hydropathy scales, we selected several evaluation metrics that together provide a well-rounded view of each scale's effectiveness:
1. Accuracy
Measures the overall proportion of correct predictions. It serves as a general baseline for model performance.
2. F1 Score (Overall)
The harmonic mean of precision and recall. It is especially valuable in 
situations with class imbalance, which is common in protein interaction data.
3. Optimal Threshold
Represents the threshold value at which CIRNet achieves its best F1-score. It provides insight into the decision boundary sensitivity of the model for each scale.
4. Maximum Class-Wise F1 Score
The highest F1 score achieved for any specific residue class pair (e.g., 
Hydrophobic-Hydrophobic, Charged-Polar, etc.). This helps identify if a scale is particularly effective for distinguishing interactions within or between specific residue types.
Rationale Behind Metric Selection
• Comprehensive evaluation: No single metric fully captures performance. A combination of global and class-specific metrics offers both breadth and detail.
• Suitability for imbalance: F1 Score provides a better balance of false positives 
and false negatives, which is important when class distribution is skewed.
• Sensitivity and class-specific insight: Optimal threshold and max per-class F1 allow us to see which scales help the model make confident and class￾sensitive predictions.
Composite Score Calculation
To rank hydropathy scales consistently, we calculated a composite score that combines the normalized values of the key performance metrics:
Composite Score = Mean of Normalized Accuracy, Normalized F1 Score, and 
Normalized Max Class-Wise F1 Score
• All metrics were min-max normalized to a common [0, 1] scale to ensure comparability.
• Equal weighting was used to avoid bias toward any particular metric.
• This single score reflects overall model quality and scale-specific strength.
The resulting ranked list was saved as a CSV and visualized with a bar chart showing the top 5 performing hydropathy scales.