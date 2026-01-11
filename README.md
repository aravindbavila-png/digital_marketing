Digital Marketing Campaign Conversion Prediction

Project Overview:
   In today’s competitive digital marketing landscape, predicting which customers are likely to convert is crucial for optimizing ad spend and improving ROI.
This project uses Machine Learning techniques to predict customer conversion behavior based on demographic, engagement, and campaign-related features.

The system helps marketing teams target high-potential customers, reduce unnecessary ad spend, and maximize ROAS.

Objectives:
Build a machine learning model to predict customer conversions accurately
Perform Exploratory Data Analysis (EDA) to uncover key insights
Compare multiple ML algorithms and select the best-performing model
Identify important features influencing conversions
Deploy a user-friendly Streamlit prediction app

Dataset Information:
Records: ~8,000
Features: ~20
Data Types: Demographic, Engagement, Campaign & Behavioral
Target Variable: Conversion
  0 → No Conversion
  1 → Conversion

Exploratory Data Analysis (EDA):
No missing or duplicate values found
Removed irrelevant columns (CustomerID, AdvertisingPlatform, AdvertisingTool)
High correlation observed with:
  Email Opens
  Website Visits
  Loyalty Points
  Previous Purchases
Right-skewed distributions for Income and Ad Spend

Data Preprocessing:
One-Hot Encoding for categorical variables
Feature scaling using StandardScaler
Stratified Train-Test split (80% / 20%)
Handled class imbalance using scale_pos_weight

Models Used:
Decision Tree
Random Forest
Gradient Boosting
XGBoost
Support Vector Machine (SVM)
Best Model: XGBoost Classifier

Tech Stack:
Python
Pandas, NumPy
Scikit-learn
XGBoost
Matplotlib, Seaborn
Streamlit

Streamlit Prediction App:
Users input customer details
Real-time conversion probability prediction
Optimized decision threshold using F1-score
Simple and interpretable feature selection
      Key Features Used:
           Age
           Gender
           Income
           Ad Spend
           Website Visits
           Time on Site

Author:::: Aravind S
Aspiring Data Scientist | Machine Learning Enthusiast          
