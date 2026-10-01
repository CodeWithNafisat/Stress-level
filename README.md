# Stress Level Classification

I built a classifier that sorts people into three stress categories, No Stress, Eustress (positive stress) and Distress (negative stress), from 20 psychological, physiological, environmental, academic and social indicators. The best single models, Logistic Regression and a Decision Tree, each reached 0.90 accuracy on a held-out test set, and a hard-voting ensemble of three models scored 0.89.

The more interesting result is what the comparison showed: on this data, combining models did not beat the best individual ones.

## The Problem

Stress shows up through many signals at once, such as anxiety, sleep, self-esteem, workload, noise and social support. The goal was to see how well a model could turn those indicators into a clear stress category, and which indicators matter most for that prediction.

## The Data

The dataset has 1,100 records and 21 columns: 20 features and the `stress_level` target. The features include `anxiety_level`, `self_esteem`, `mental_health_history`, `depression`, `sleep_quality`, `blood_pressure`, `noise_level`, `study_load`, `bullying`, `social_support` and others. There were no missing values and no duplicates.

The three classes are almost perfectly balanced: 373 No Stress, 369 Distress and 358 Eustress. That kept the evaluation simple, since accuracy and macro-averaged metrics are meaningful here and no resampling was needed.


## What I Did

- Checked for missing values and duplicates, and looked at boxplots for outliers. They were few, so I winsorized every numeric column at 5% per tail.
- Standardized the numeric features and ordinal-encoded the target.
- Split the data 80/20 with stratification, giving 880 training and 220 test records.
- Selected features using a Decision Tree with `SelectFromModel` and a median importance threshold. That kept 10 of the 20 features: `anxiety_level`, `self_esteem`, `depression`, `blood_pressure`, `sleep_quality`, `breathing_problem`, `noise_level`, `basic_needs`, `study_load` and `bullying`.
- Trained three different models on those features: Logistic Regression (L2), a Decision Tree and KNN. Then I combined them in a hard-voting ensemble.

## Results

On the 220-record test set, Logistic Regression and the Decision Tree both reached 0.90 accuracy and 0.90 macro F1. KNN reached 0.87. The voting ensemble reached 0.89 accuracy, with macro precision of 0.90, macro recall of 0.89 and macro F1 of 0.89.


The models also differ in where they are strongest. Logistic Regression was most precise on Eustress (0.96), while the Decision Tree was most precise on No Stress (0.96) but recalled fewer of those cases (0.86).

With 220 test records, one prediction is about 0.45 percentage points of accuracy. The 0.89 against 0.90 gap between the ensemble and the best single models is therefore about two predictions, which is within normal noise for a split this size. I read the ensemble as matching the best single models here, not improving on them.

## Key Takeaways

- Ten features were enough for about 0.90 accuracy, so a subset of the indicators carries most of the usable signal. I did not test the full 20-feature set against it, so I can't say how much the other ten add.
- Two simple models, Logistic Regression and a Decision Tree, matched the ensemble. More complexity did not buy accuracy on this dataset.
- Balanced classes made the evaluation straightforward, so I could rely on accuracy alongside macro precision, recall and F1.

## Tech Stack

Python, pandas, NumPy, scikit-learn (LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, VotingClassifier, SelectFromModel, StandardScaler, OrdinalEncoder), SciPy (winsorization), seaborn, Matplotlib and Jupyter Notebook.

## Project Structure

```
├── Stress_Level.ipynb
├── StressLevelDataset.csv
├── images
└── README.md
```
