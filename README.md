# 🧠 ML Models Hub

A collection of **6 interactive Machine Learning web applications** built with [Streamlit](https://streamlit.io/). Each app trains its model on startup and provides a premium, dark-themed UI for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Screenshots

| Diabetes Prediction | Heart Disease Prediction |
|:---:|:---:|
| ![Diabetes](screenshots/diabetes.png) | ![Heart](screenshots/heart.png) |

| Loan Prediction | Spam Mail Detector |
|:---:|:---:|
| ![Loan](screenshots/loan.png) | ![Spam](screenshots/spam.png) |

| Movie Recommender | Customer Churn |
|:---:|:---:|
| ![Movie](screenshots/movie.png) | ![Churn](screenshots/churn.png) |

---

## 🚀 Models Included

| # | Model | Algorithm | Features | Accuracy |
|---|-------|-----------|----------|----------|
| 1 | **🩺 Diabetes Prediction** | SVM (Linear Kernel) | Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, DPF, Age | ~78% |
| 2 | **❤️ Heart Disease Prediction** | Logistic Regression | 13 clinical features (age, sex, chest pain, cholesterol, etc.) | ~85% |
| 3 | **🏦 Loan Status Prediction** | SVM (Linear Kernel) | Gender, Income, Loan Amount, Credit History, Property Area, etc. | ~80% |
| 4 | **📧 Spam Mail Detector** | Logistic Regression + TF-IDF | Email text content | ~96% |
| 5 | **🎬 Movie Recommender** | TF-IDF + Cosine Similarity | Genres, Keywords, Tagline, Cast, Director | N/A (similarity) |
| 6 | **📊 Customer Churn Prediction** | Random Forest | 19 telecom features (tenure, services, charges, etc.) | ~79% |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS (glassmorphism, gradients, animations)
- **ML**: [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [imbalanced-learn](https://imbalanced-learn.org/)
- **Data**: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ml-models-hub.git
cd ml-models-hub
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Launch the Hub (main page)

```bash
streamlit run app_main.py
```

### Launch Individual Models

```bash
# Diabetes Prediction
streamlit run "Diabetes prediction/app_diabetes.py"

# Heart Disease Prediction
streamlit run "Heart Disease Prediction/app_heart.py"

# Loan Status Prediction
streamlit run "Loan Prediction/app_loan.py"

# Spam Mail Detector
streamlit run "spam mail prediciton/app_spam.py"

# Movie Recommender
streamlit run "Movie Recommendation/app_movie.py"

# Customer Churn Prediction
streamlit run "Customer Churn Prediction/app_churn.py"
```

---

## 📁 Project Structure

```
ML_projects-1/
├── app_main.py                          # 🧠 Hub page (links to all models)
├── requirements.txt
├── README.md
├── .gitignore
│
├── Diabetes prediction/
│   ├── app_diabetes.py                  # 🩺 Streamlit UI
│   ├── diabetes.csv                     # Dataset
│   └── Diabetes_Prediction.ipynb        # Jupyter notebook
│
├── Heart Disease Prediction/
│   ├── app_heart.py                     # ❤️ Streamlit UI
│   ├── heart_disease_data.csv           # Dataset
│   └── Heart_Disease_Prediction.ipynb   # Jupyter notebook
│
├── Loan Prediction/
│   ├── app_loan.py                      # 🏦 Streamlit UI
│   ├── loan.csv                         # Dataset
│   └── Loan_Status_Prediction.ipynb     # Jupyter notebook
│
├── spam mail prediciton/
│   ├── app_spam.py                      # 📧 Streamlit UI
│   ├── mail_data.csv                    # Dataset
│   └── Spam_Mail_Prediction_...ipynb    # Jupyter notebook
│
├── Movie Recommendation/
│   ├── app_movie.py                     # 🎬 Streamlit UI
│   ├── movies.csv                       # Dataset
│   └── Movie_Recommendation_...ipynb    # Jupyter notebook
│
└── Customer Churn Prediction/
    ├── app_churn.py                     # 📊 Streamlit UI
    ├── WA_Fn-UseC_-Telco-...csv         # Dataset
    └── Customer_Churn_...ipynb          # Jupyter notebook
```

---

## ✨ Features

- 🎨 **Premium Dark UI** — Glassmorphism cards, smooth gradients, and micro-animations
- ⚡ **Instant Predictions** — Models train on startup and cache results
- 📊 **Accuracy Display** — Training and test accuracy shown for each model
- 📱 **Responsive Layout** — Works on different screen sizes
- 🔗 **Central Hub** — One main page to access all 6 models

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -m 'Add new model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Streamlit & scikit-learn
</p>
# Mlproject
