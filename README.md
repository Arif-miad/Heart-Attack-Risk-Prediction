 

---

# Heart Attack Risk Prediction  

![Dataset Analysis](https://img.shields.io/badge/Status-Complete-green)  
![Python](https://img.shields.io/badge/Language-Python-blue)  
![License](https://img.shields.io/badge/License-MIT-orange)  

## 📖 Overview  
This project focuses on analyzing and predicting heart attack risks using a dataset containing patient demographic, clinical, and lifestyle features. The workflow involves exploratory data analysis (EDA), preprocessing, model training, evaluation, and visualization of results using advanced machine learning techniques.  

---

## 🗂️ Dataset Description  
The dataset consists of the following columns:  
- **Age**: Age of the patient.  
- **Sex**: Gender of the patient (M/F).  
- **ChestPainType**: Type of chest pain (e.g., ATA, NAP, ASY).  
- **RestingBP**: Resting blood pressure (mm Hg).  
- **Cholesterol**: Serum cholesterol in mg/dl.  
- **FastingBS**: Fasting blood sugar (1 = true, 0 = false).  
- **RestingECG**: Resting electrocardiographic results.  
- **MaxHR**: Maximum heart rate achieved.  
- **ExerciseAngina**: Exercise-induced angina (Y/N).  
- **Oldpeak**: ST depression induced by exercise.  
- **ST_Slope**: Slope of the peak exercise ST segment.  
- **HeartDisease**: Target variable (1 = disease, 0 = no disease).  

---

## 🚀 Workflow  

### 1️⃣ **Exploratory Data Analysis (EDA)**  
- **Univariate Analysis**: Histograms, count plots, and KDE plots for individual features.  
- **Multivariate Analysis**: Pairwise scatter plots, heatmaps for correlations, and bar plots for feature relationships.  

### 2️⃣ **Data Preprocessing**  
- Handled missing values (if any).  
- Applied label encoding and one-hot encoding to categorical features.  
- Scaled numerical features using MinMaxScaler.  

### 3️⃣ **Model Training**  
Trained and evaluated 15 classification models, including:  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machines (SVM)  
- XGBoost, AdaBoost, and more.  

### 4️⃣ **Model Evaluation**  
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC were calculated for each model.  
- Visualized ROC Curves and confusion matrices for performance comparison.  

### 5️⃣ **Feature Importance Analysis**  
- Identified important features using Random Forest feature importance plot.  

---

## 🧪 Results and Insights  
- The **Random Forest Classifier** and **XGBoost** models demonstrated the highest accuracy and ROC-AUC scores.  
- Feature importance analysis highlighted key factors influencing heart attack risks, such as **MaxHR**, **Cholesterol**, and **Oldpeak**.  

---

## 🛠️ Tools and Libraries  
- **Python**: Core programming language.  
- **Libraries**:  
  - Data Manipulation: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`, `xgboost`  

---

## 📊 Visualizations  
1. **ROC Curves**: Showcasing model performance.  
2. **Feature Importance Plot**: Highlighting the top predictors for heart disease.  
3. **Heatmap**: Depicting correlations between features.  

---

## 💡 Key Features  
- Comprehensive workflow for heart disease risk analysis.  
- Comparison of multiple machine learning models.  
- Clear and interpretable visualizations.  

---

## 📥 Usage  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/username/heart-attack-risk-prediction.git  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the notebook to reproduce results.  

---

## 📝 License  
This project is licensed under the [MIT License](LICENSE).  

---

