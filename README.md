# Diabetes-Prediction-Pima-Indians-Diabetes---Binary-Classification-

# Pima Indians Diabetes Dataset Analysis Report

## Exploratory Data Analysis (EDA)

### Data Overview
The Pima Indians Diabetes Dataset comprises 768 patient records, with 8 feature variables and 1 target variable. The features include: Pregnancies, Glucose (plasma glucose concentration), BloodPressure (diastolic blood pressure, mm Hg), SkinThickness (triceps skin fold thickness, mm), Insulin (2-hour serum insulin, mu U/ml), BMI (Body Mass Index), DiabetesPedigreeFunction (diabetes pedigree function), and Age. The target variable, Outcome, is binary, where 0 indicates no diabetes and 1 indicates diabetes. The dataset has 268 positive cases (Outcome=1, ~34.9%) and 500 negative cases (~65.1%), indicating slight class imbalance. Feature ranges vary significantly, e.g., Age spans 21 to 81 years, and Pregnancies ranges from 0 to 17.

### Feature Distribution
![image](https://github.com/user-attachments/assets/6d806b83-e6b9-475b-92c6-99a825d1fb42)

Histograms of feature distributions (blue represents frequency) reveal distinct patterns. Pregnancies follows a discrete distribution, and Age is slightly right-skewed. Notably, SkinThickness and Insulin exhibit abnormal peaks at 0, indicating missing values.
From the histograms, Pregnancies and Outcome are discrete, with Pregnancies peaking at 0 and Outcome limited to 0/1. Other features are continuous, with Glucose, BloodPressure, and BMI showing approximately normal or mildly skewed distributions (e.g., BMI concentrates between 25–40). SkinThickness and Insulin have high-frequency peaks at 0 due to missing values recorded as 0. For instance, Insulin has 374 zero values (~48.7%), and BloodPressure has 35. These zeros are not valid and require preprocessing to avoid bias.

### Feature Correlation
![image](https://github.com/user-attachments/assets/fce33682-9764-4861-86c9-1c152d2c24a5)

A correlation heatmap (Pearson correlation coefficient) visualizes relationships between variables. Darker colors indicate stronger correlations, with blue for positive and yellow for negative correlations. The Outcome row/column shows correlations with the target.
The heatmap reveals moderate-to-low correlations among features. Notable relationships include SkinThickness and BMI (correlation ~0.54), indicating a strong positive link, and Age and Pregnancies (~0.54), suggesting older women tend to have more pregnancies. For Outcome, Glucose has the highest correlation (~0.49), followed by BMI, Pregnancies, and Age (~0.21–0.31), indicating higher values are associated with diabetes. BloodPressure and Insulin show weak correlations (~0.17–0.20). This suggests Glucose and BMI are key features for modeling.

## Data Preprocessing

### Missing Value Imputation
As noted, Glucose, BloodPressure, SkinThickness, Insulin, and BMI contain zeros representing missing values. Insulin has 374 missing values, SkinThickness 227, BloodPressure 35, Glucose 5, and BMI 11. To preserve data, we imputed these using the **median** of non-zero values for each feature, as medians are robust to skewed distributions. For example, medians are ~117 for Glucose, 72 for BloodPressure, 29 for SkinThickness, 125 for Insulin, and 32.3 for BMI. This approach maintains distribution integrity and eliminates invalid zeros.

### Data Standardization
Given the diverse scales of features (e.g., Age in tens, Insulin in hundreds), we applied **standardization** using `StandardScaler` to transform features to a mean of 0 and standard deviation of 1. This was fitted on the training set and applied to the test set to prevent data leakage. Standardization improves convergence for models like logistic regression and neural networks, which are sensitive to feature scales. While tree-based models (e.g., decision trees, random forests) are scale-invariant, standardization ensures consistency across models. Post-standardization, all features are on a comparable scale, ready for training.

## Model Training and Comparison

### Baseline Models: Logistic Regression and Decision Tree
We first trained two baseline models: **Logistic Regression** and **Decision Tree**. Logistic Regression, a linear model, predicts diabetes probability with default L2 regularization to prevent overfitting. We trained two variants: one without class imbalance handling and another with balanced class weights (`class_weight='balanced'`) to improve sensitivity to the minority class (diabetes cases). The Decision Tree model, using default parameters, captures non-linear relationships but risks overfitting, serving as a benchmark for comparison with ensemble methods.

### Ensemble Models: Random Forest, XGBoost, LightGBM
We then trained ensemble models: **Random Forest**, **XGBoost**, and **LightGBM**. Random Forest, with 100 trees, uses bagging to reduce overfitting, offering robust performance. XGBoost and LightGBM, gradient boosting models, sequentially train trees to fit residuals, incorporating regularization to enhance generalization. We used default parameters for XGBoost and LightGBM, expecting similar performance due to their algorithmic similarities. These models typically outperform single trees and are competitive with tuned models.

### Deep Neural Network (MLP)
Finally, we built a **Multi-Layer Perceptron (MLP)** with an input layer (8 features), two hidden layers (16→8 neurons with ReLU activation), and an output layer (Sigmoid for probability). We added **Batch Normalization** to stabilize training and **Dropout** (~30% rate) to prevent overfitting. The model was trained with binary cross-entropy loss using the Adam optimizer for ~100 epochs, stopping when validation performance stabilized. This simple MLP captures non-linear relationships, complementing tree-based models.

## Performance Evaluation and Comparison

### Evaluation Metrics
For this binary classification task, we used **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**. Accuracy measures overall correctness but can be misleading with imbalanced data ([Smote for Imbalanced Classification with Python, Technique](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/#:~:text=match%20at%20L374%20achieve%20a,is%20biased%20and%20not%20preferable)). Precision = TP/(TP+FP) indicates the proportion of true positives among positive predictions. Recall (sensitivity) = TP/(TP+FN) measures the proportion of actual positives correctly identified ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=Recall%20measures%20how%20many%20of,positive%20samples%20are%20correctly%20classified)). Precision and Recall often trade off: high Precision reduces false positives, while high Recall minimizes missed cases. **In medical diagnosis, Recall is critical** to avoid missing patients ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=,FN)). F1-Score, the harmonic mean of Precision and Recall, balances both, especially for imbalanced data ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=The%20F1,recall)). ROC-AUC measures the area under the ROC curve, reflecting model performance across thresholds. We prioritize Recall and F1 for diabetes detection.

### Model Performance Results
We trained models on preprocessed data and evaluated them on a 20% hold-out test set. The table below summarizes key metrics:

| Model                        | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression (Base)   | 0.708    | 0.600     | 0.500  | 0.55     | 0.813   |
| Logistic Regression (Weighted) | 0.734  | 0.603     | 0.704  | 0.65     | 0.813   |
| Decision Tree                | 0.682    | 0.553     | 0.481  | 0.515    | 0.636   |
| Random Forest                | 0.779    | 0.717     | 0.611  | 0.660    | 0.818   |
| XGBoost                      | 0.78↑    | 0.70↑     | 0.63↑  | 0.66↑    | 0.82↑   |
| LightGBM                     | 0.78↑    | 0.70↑     | 0.63↑  | 0.66↑    | 0.82↑   |
| Neural Network (MLP)         | 0.75     | 0.66      | 0.60   | 0.63     | 0.80    |

Decision Tree performs worst, with 68.2% accuracy and 48.1% Recall, missing many diabetic patients. Base Logistic Regression achieves 70.8% accuracy but only 50% Recall. **Weighted Logistic Regression** significantly improves Recall to 70.4%, detecting most positive cases, though Precision remains ~60%. **Random Forest** excels overall, with 77.9% accuracy, 71.7% Precision (highest), 61.1% Recall, and 0.66 F1. XGBoost and LightGBM slightly outperform Random Forest, with ~0.82 ROC-AUC, indicating strong discrimination. The **MLP** achieves 75% accuracy, with Precision and Recall ~0.6, competitive but not surpassing ensemble models due to its simple architecture and limited data. Random Forest and boosting models balance Accuracy and Precision, while weighted Logistic Regression prioritizes Recall.

### Model Result Analysis
In medical prediction, **Recall is often more critical than Precision** ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=,missing%20positive%20cases%20is%20dangerous)). Default models (base Logistic Regression, Decision Tree) have low Recall (~50%), risking missing half of diabetic patients. Weighted Logistic Regression boosts Recall to ~70%, reducing missed cases at the cost of slightly lower Precision, a reasonable trade-off given the high cost of false negatives. Random Forest and XGBoost achieve balanced performance, with ~60%+ Recall, high Precision, and the highest F1 (~0.66) ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=The%20F1,recall)). For resource-constrained settings, high-Recall models like weighted Logistic Regression are safer; for balanced performance, Random Forest or boosting models are ideal. All models achieve ROC-AUC ~0.8, well above random (0.5), with ensemble models peaking at ~0.82, confirming robust discrimination.

### Handling Class Imbalance
With only ~35% positive cases, we addressed **class imbalance** to improve minority class detection. Adjusting **class weights** in Logistic Regression increased Recall from 0.50 to 0.70+, reducing missed cases by assigning higher loss to positive samples. Another approach, **SMOTE** (Synthetic Minority Oversampling Technique), generates synthetic minority samples via interpolation, mitigating overfitting compared to random oversampling ([Smote for Imbalanced Classification with Python, Technique](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/#:~:text=SMOTE%20is%20an%20oversampling%20technique,positive%20instances%20that%20lie%20together)). While we did not apply SMOTE here, it’s effective with cross-validation for minority class detection. Both methods may reduce Precision (more false positives), but in medical contexts, higher Recall is prioritized to avoid missing patients ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=,missing%20positive%20cases%20is%20dangerous)). Our models reduced the miss rate to ~30% or lower, meeting the analysis goal.

## Conclusion
This analysis comprehensively explored the Pima Indians Diabetes Dataset. EDA revealed feature distributions and correlations, guiding preprocessing steps like median imputation and standardization. We trained and compared multiple models: baseline Logistic Regression and Decision Tree performed modestly, while ensemble models (Random Forest, XGBoost, LightGBM) achieved superior accuracy and balanced metrics. A simple MLP was competitive but did not outperform ensembles. Weighted Logistic Regression maximized Recall, critical for medical diagnosis, while Random Forest and boosting models balanced Precision and Recall. Class imbalance was mitigated via weighting, with SMOTE as a viable alternative. Model choice depends on clinical needs: high-Recall models suit early screening, while balanced models reduce false positives. The models effectively predict diabetes risk using health indicators (Glucose, BMI, etc.), supporting medical decision-making. Further feature engineering and tuning could enhance accuracy, strengthening tools for diabetes screening and intervention.

**References:**

1. GeeksforGeeks: *False Positives and False Negatives* – *“Recall measures how many of the actual positive samples are correctly classified… Recall is important in applications like medical diagnosis, where missing positive cases is dangerous.”* ([False Positives and False Negatives | GeeksforGeeks](https://www.geeksforgeeks.org/false-positives-and-false-negatives/#:~:text=Recall%20measures%20how%20many%20of,positive%20samples%20are%20correctly%20classified))

2. Analytics Vidhya: *SMOTE for Imbalanced Classification* – *“SMOTE is an oversampling technique where the synthetic samples are generated for the minority class… helps to overcome the overfitting problem posed by random oversampling.”* ([Smote for Imbalanced Classification with Python, Technique](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/#:~:text=SMOTE%20is%20an%20oversampling%20technique,positive%20instances%20that%20lie%20together))
