# Food Coma Predictor 🍔💤

**🎮 [Click here to use the live Web App!](https://SayandeepGit.github.io/Food-Coma-ML-Project/Lets_play.html)**

Have you ever wondered why you feel incredibly sleepy after a heavy lunch? This project analyzes the classic "Food Coma" phenomenon using real-world survey data, Machine Learning models, and SHAP explainability to determine exactly what triggers post-meal drowsiness. 

As a bonus, it includes a fun, interactive web app where you can predict your own fate before your next meal!

## 📁 Project Files
* **`FinalCode_v7.py`**: The main Python script containing the entire ML pipeline (Data cleaning, EDA, Preprocessing, Model Training, Evaluation, and SHAP visualizations).
* **`Food Coma Study_ ... .csv`**: The raw dataset collected via Google Forms from real respondents.
* **`Lets_play.html`**: A fun, lightweight frontend UI to test if you'll survive your next meal without taking a nap.

## 📊 The Dataset
The dataset consists of real responses detailing people's habits, physical attributes, and meals. 
* **Target Variable**: `Did you feel drowsy after your meal?` (Yes/No)
* **Key Features Used**: 
  * Meal Size, Carb Content, Protein Content, Meal Heaviness
  * Sleep Quality, Stress Level
  * Last Meal Type (Breakfast/Lunch/Dinner/Snacks)
  * Pre-Meal Physical Activity
* **Data Cleaning Highlight**: The script includes custom parsers to clean highly irregular user inputs for Age, Height (mixed cm/feet/inches), and Weight, followed by dynamic BMI calculation.

## **Exploratory Data Analysis**:
* How many people felt drowsy after eating: ![EDA_1](<EDA_1_How many people felt drowsy after eating.png>)
* Selected Features vs Drowsiness: ![EDA_2](<EDA_2_Selected features vs Drowsiness.png>)

## 🧠 Machine Learning Pipeline
The project utilizes a robust `scikit-learn` pipeline with `ColumnTransformer` to handle Numeric, Ordinal, and Nominal data perfectly. 

Three models were trained and compared (using class-weight balancing to handle class imbalances):
1. **Random Forest Classifier** 🌲
2. **Support Vector Machine (SVM)** 📈
3. **Logistic Regression** 📊

### Evaluation & Explainability
The script automatically generates beautiful, publication-ready visual insights:
* **Exploratory Data Analysis (EDA)**: 8 custom charts showing the relationship between habits and drowsiness.
* **Model Evaluation**: Confusion Matrices, ROC-AUC curves, and a comprehensive Metric Comparison chart (Accuracy, Precision, Recall, F1).
* **Feature Importance**: Gini/MDI for Random Forest and Permutation Importance for all models.
* **SHAP Analysis**: Global feature importance and Beeswarm plots to explain *how* specific features (like high carbs or low sleep) drive the model's predictions.

* ## 📈 Results & Performance
The pipeline evaluates three different models: **Random Forest**, **Support Vector Machine (SVM)**, and **Logistic Regression**. Because the dataset has a slight class imbalance (more people feel drowsy than not), all models utilize `class_weight='balanced'`.

Here is the performance comparison across our key metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.731 | 0.750 | 0.882 | 0.811 | 0.771 |
| **SVM** | 0.808 | 0.833 | 0.882 | 0.857 | 0.830 |
| **Logistic Regression** | 0.769 | 0.824 | 0.824 | 0.824 | 0.778 |

### 🔍 Visualizing the Results
The script automatically generates several plots to evaluate the models and understand the data. Here are a few highlights:

**1. Model Performance (ROC Curves & Confusion Matrices)**
*The Logistic Regression and Random Forest models consistently show strong ability to distinguish between Food Coma vs. Alert states.*
> ![ROC Curves](<ROC Curves All models Compared.png>)
> ![Confusion Matrices](<Confusion Matrices for all the Models.png>)

**2. What causes a Food Coma? (Feature Importance)**
*Using SHAP and Permutation Importance, we can see exactly which features drive drowsiness. Meal Size, Carb Content, and Sleep Quality are the heaviest hitters.*
> ![SHAP Beeswarm Plot](<SHAP Beeswarm Plot.png>)
> ![Feature Importance](<Permutation Feature Importance all models.png>)

## 🎮 The Web App ("Could u survive this hour?")
Don't want to run the python code? Just open the web app!
`Lets_play.html` is a sleek, interactive frontend where you input your Meal Size, Stress Level, and Sleep Quality to get a brutally honest prediction about your impending food coma. 

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/SayandeepGit/[your-repo-name].git
cd [your-repo-name]
```

### 2. Play with the Web App
Simply double-click the `Lets_play.html` file in your file explorer to open it in any web browser. No server required!

### 3. Run the Machine Learning Script
Ensure you have Python installed, then install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

Run the pipeline:
```bash
python FinalCode_v7.py
```
*(Note: The script will output the cleaning process in the console and pop up the EDA, evaluation, and SHAP plots one by one).*

## 🤝 Contributing
Did you find a new feature that predicts food comas better? Feel free to open an issue or submit a pull request!
