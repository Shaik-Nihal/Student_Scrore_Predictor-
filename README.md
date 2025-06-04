# Student Exam Score Predictor

## 🚀 Overview

This project aims to predict student exam scores based on various personal habits, academic background, and engagement factors. By leveraging machine learning techniques, the model provides insights into factors influencing academic performance and can serve as a tool for early identification of students who might need additional support.

## 🎯 Problem Statement

Educational institutions and students are keen to understand and predict academic performance. An accurate prediction of student exam scores can help:
* Identify at-risk students early for timely interventions.
* Enable educators to tailor support and resources more effectively.
* Provide students with insights into how their habits might affect their scores.

The core challenge is to build a reliable model that accurately predicts a student's exam score based on a given set of input features.

## 📊 Dataset

The project uses a dataset named `student_habits_performance.csv` (not included in this README, but should be in your repository or its source mentioned). This dataset contains information about students, including:

* **Demographics & Background:** `age`, `gender`, `parental_education_level`
* **Study Habits:** `study_hours_per_day`, `attendance_percentage`
* **Lifestyle Factors:** `social_media_hours`, `sleep_hours`, `exercise_frequency`, `diet_quality`, `part_time_job`
* **Well-being:** `mental_health_rating`
* **Other (Initially considered but dropped):** `student_id`, `internet_quality`, `netflix_hours`
* **Target Variable:** `exam_score`

## ✨ Features

The final set of features used for modeling after preprocessing includes:
* `age`
* `study_hours_per_day`
* `social_media_hours`
* `attendance_percentage`
* `sleep_hours`
* `exercise_frequency`
* `mental_health_rating`
* One-hot encoded features for:
    * `gender` (e.g., `gender_Male`, `gender_Female`, `gender_Other`)
    * `part_time_job` (e.g., `part_time_job_Yes`, `part_time_job_No`)
    * `diet_quality` (e.g., `diet_quality_Good`, `diet_quality_Fair`, `diet_quality_Poor`)
    * `parental_education_level` (e.g., `parental_education_level_Bachelor`, `parental_education_level_High School`, etc.)
    * `extracurricular_participation` (e.g., `extracurricular_participation_Yes`, `extracurricular_participation_No`)

## 🛠️ Methodology

The project follows these key steps:

1.  **Data Loading & Initial Exploration:**
    * Load the dataset using Pandas.
    * Initial inspection of data types, missing values, and basic statistics.
    * Visualizations to understand feature distributions and relationships (e.g., age vs. study hours, gender distribution).
2.  **Data Preprocessing:**
    * **Handling Missing Values:** Filled missing `parental_education_level` with 'Unknown'.
    * **Dropping Irrelevant Columns:** Removed `student_id`, `internet_quality`, and `netflix_hours`.
    * **Encoding Categorical Features:** Applied one-hot encoding to categorical columns (`gender`, `part_time_job`, `diet_quality`, `parental_education_level`, `extracurricular_participation`) using `pd.get_dummies()`.
    * **Data Type Conversion:** Ensured all features are converted to integer type for modeling.
3.  **Model Training & Selection:**
    * **Splitting Data:** Divided the dataset into training (80%) and testing (20%) sets using `train_test_split` from scikit-learn.
    * **Model Implementation:**
        * Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
        * Linear Regression (`sklearn.linear_model.LinearRegression`)
    * **Training:** Fitted the models on the training data.
4.  **Model Evaluation:**
    * Predicted scores on the test set.
    * Evaluated model performance using the following metrics:
        * Mean Squared Error (MSE)
        * R-squared (R²) Score
        * Mean Absolute Error (MAE)
        * Root Mean Squared Error (RMSE)
5.  **Prediction Function:**
    * Developed a function `predict_exam_score_flexible` that takes raw student input (categorical and numerical), preprocesses it according to the training pipeline, and returns a predicted exam score.

## 💻 Technologies & Libraries Used

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn (sklearn):** For machine learning tasks (model implementation, train-test split, metrics).
* **NumPy:** For numerical computations.
* **Matplotlib & Seaborn:** For data visualization.
* **Google Colab / Jupyter Notebook:** As the development environment.

## 📈 Results

The models were evaluated on the test set. Key performance metrics for the **Linear Regression model** are (update with your final script values):
* **Mean Absolute Error (MAE):** [e.g., 4.6059]
* **Mean Squared Error (MSE):** [e.g., 32.5609]
* **Root Mean Squared Error (RMSE):** [e.g., 5.7062]
* **R-squared (R²):** [e.g., 0.8754]

Performance metrics for the **Random Forest Regressor model** are (update with your final script values):
* **Mean Squared Error (MSE):** [e.g., 28.79]
* **R-squared (R²):** [e.g., 0.8899]

The `predict_exam_score_flexible` function allows for predicting the exam score for a new student profile. For instance, given:
```python
student_input = {
    'age': 22, 'study_hours_per_day': 4, 'social_media_hours': 2,
    'attendance_percentage': 95, 'sleep_hours': 7, 'exercise_frequency': 4,
    'mental_health_rating': 8, 'gender': 'Male',
    'extracurricular_participation': 'Yes', 'part_time_job': 'No',
    'parental_education_level': 'Bachelor', 'diet_quality': 'Good'
}
