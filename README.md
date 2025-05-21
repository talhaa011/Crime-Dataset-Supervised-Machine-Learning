# 🕵️‍♂️ Crime Case Closure Prediction – India 🇮🇳

This project aims to analyze and predict the closure status of crime cases across various cities in India using machine learning techniques. By understanding patterns in the data, this work helps identify factors influencing crime resolution and offers valuable insights for law enforcement agencies.

---

## 📌 Problem Statement

Crime is a persistent concern for societies and governments. In this project, we explore a real-world dataset of reported crimes in Indian cities to develop a predictive model that determines whether a crime case will be **closed or not** based on several features such as:

- City where the crime occurred
- Crime description and domain
- Victim details (gender, age)
- Weapons used
- Case closure status (target variable)

---

## 🎯 Project Objectives

1. **Explore and visualize** the patterns in the crime data.
2. **Preprocess** the data (handling missing values, encoding categorical data).
3. **Train a machine learning model** (Random Forest Classifier) to predict case closure.
4. **Evaluate model performance** using accuracy, confusion matrix, and classification report.
5. **Optimize the model** using hyperparameter tuning (GridSearchCV).
6. **Extract actionable insights** to support crime investigation and policymaking.

---

## 📂 Dataset

The dataset used in this project is [`crime_dataset_india.csv`](crime_dataset_india.csv), which contains crime reports from various cities in India. Key features include:

- `City`
- `Crime Head`, `Crime Subhead`
- `Victim Age`, `Victim Gender`
- `Weapon Used`
- `Case Closed` (Target)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (Data Visualization)
- Scikit-learn (Modeling & Evaluation)
- Jupyter Notebook

---

## 🧪 Model Development

The primary model used is a **Random Forest Classifier**. Steps include:

- Encoding categorical variables with `LabelEncoder`
- Splitting dataset into training and test sets
- Training the Random Forest model
- Evaluating with accuracy, confusion matrix, classification report
- Hyperparameter tuning using `GridSearchCV`

---

## 📊 Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

---

## 📌 Key Insights

- Certain cities and crime types have lower closure rates.
- Victim demographics (age, gender) influence the likelihood of case closure.
- The use of weapons in crimes shows a trend with unresolved cases.

---

## 📌 Future Improvements

- ✅ **Integrate the model into a web application** using frameworks like **Flask** or **Django** to make it accessible to users via a web interface.

- 🚀 **Experiment with advanced regression techniques** such as:
  - **XGBoost**
  - **LightGBM**
  - **CatBoost**
  
  These can potentially improve model performance and handle non-linear relationships better.

- ☁️ **Deploy the application** using:
  - **Streamlit** for quick and interactive UI development.
  - Cloud platforms like **Heroku**, **AWS**, or **Render** for real-world accessibility.

- 📦 **Create a REST API** to allow external systems to access predictions.

- 📊 **Add interactive visualizations** for data insights using Plotly or Bokeh.

## 📬 Contact

For any inquiries or collaboration, feel free to reach out:

- 📧 **Email**: muhammadtalha3589@gmail.com
- 💼 **LinkedIn**: (https://www.linkedin.com/in/muhammad-talha-sial/)
