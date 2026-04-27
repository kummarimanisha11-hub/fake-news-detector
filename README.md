# fake-news-detector
# 📰 AI Fake News Detection System

## 📌 Project Overview
This project is an AI-based Fake News Detection system that classifies news articles as *Real* or *Fake* using Machine Learning and Natural Language Processing (NLP).

## 🚀 Features
- 🔍 Real-time news prediction
- 📊 Accuracy evaluation
- 📉 Confusion matrix visualization
- 🌐 Web app using Streamlit

## 🧠 Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## ⚙️ How it Works
1. News text is converted into numerical format using *TF-IDF Vectorizer*
2. A *Logistic Regression model* is trained on real dataset
3. The model predicts whether the news is Real or Fake

## 📁 Dataset
- Fake.csv
- True.csv  
(Used for training the model)

##Live Demo:
https://fake-news-detector-7crqk7qwvsac5owpjc4vly.streamlit.app/

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
