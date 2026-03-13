# Insurance Fraud Detection Under Realistic Data Challenges

This project explores how machine learning pipelines perform on a real-world insurance fraud detection task under common data challenges such as class imbalance and missing values. Using the Kaggle Car Insurance Fraud dataset, the goal was to build and evaluate binary classification models that can identify fraudulent claims while remaining robust to imperfect data conditions. 

## Project Overview

Insurance fraud detection is a practical binary classification problem where the minority class is often the most important one to detect. In this project, I evaluated how preprocessing choices, resampling strategies, and classifier selection affect performance when the data is moderately imbalanced and contains incomplete values. To better simulate real-world conditions, additional missingness was artificially introduced into selected categorical and numerical variables before training. 

The dataset contains 13,760 insurance claim records. After cleaning and feature selection, the final modeling dataset used 19 features describing policy information, claimant characteristics, and accident details. The target variable indicates whether a claim was reported as fraudulent. 

## Methods

The workflow was built as a full machine learning pipeline using `ColumnTransformer` and scikit-learn tools. Different preprocessing paths were applied depending on feature type:

- **Numerical features** were imputed with the median and scaled.
- **Skewed numerical features** used `RobustScaler`.
- **Ordinal categorical features** were encoded with an explicit ordering.
- **Nominal categorical features** were imputed and one-hot encoded. 

To address class imbalance, the project compared multiple resampling approaches including no sampling, random oversampling, SMOTE, and random undersampling. Model selection then evaluated several classifiers, including:

- Logistic Regression  
- Perceptron  
- K-Nearest Neighbors  
- Random Forest 

A nested cross-validation framework with `RandomizedSearchCV` was used to compare complete pipeline configurations, including optional PCA, resampling choice, and classifier hyperparameters. Final evaluation focused on metrics better suited for imbalanced classification, especially **F1-score** and **balanced accuracy**, rather than plain accuracy. 

## Results

Among the tested configurations, **Random Forest combined with RandomUnderSampler** achieved the strongest overall performance. The best pipeline reached:

- **Mean cross-validated F1-score:** 0.528  
- **Test F1-score:** about 0.516  
- **Test balanced accuracy:** about 0.67 

The results showed that handling class imbalance had a larger impact on performance than increasing pipeline complexity. Confusion matrix, ROC, and precision-recall analysis also showed that fraud detection improved compared with simpler baselines, but the problem remained difficult, with both false positives and false negatives still present. 

## Key Takeaways

- Class imbalance strongly affects fraud detection performance.
- F1-score and balanced accuracy are more informative than standard accuracy for this task.
- Random Forest remained a strong baseline for this medium-sized tabular dataset.
- Random undersampling improved minority-class detection more than more complex pipeline additions.
- Learning curves suggested some overfitting, though hyperparameter tuning helped reduce it. 

## Future Improvements

Possible next steps for this project include:

- testing boosted tree models such as **XGBoost**
- adding explainability tools such as **SHAP**
- exploring threshold tuning to better balance precision and recall
- trying richer feature engineering for claim and accident patterns :contentReference[oaicite:9]{index=9}

## Tools and Libraries

This project uses:

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- matplotlib
- missingno

## Dataset

The project is based on the **Kaggle Car Insurance Fraud** dataset and focuses on predicting whether an insurance claim is fraudulent. 
