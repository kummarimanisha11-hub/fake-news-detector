import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("📰 AI Fake News Detector")

# Sample dataset
data = {
    "text": [
        "Government announces new policy",
        "Aliens landed in India yesterday",
        "Stock market hits record high",
        "Drinking bleach cures diseases"
    ],
    "label": [1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

# Input from user
user_input = st.text_area("Enter News Text:")

# Button
if st.button("Check News"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success("✅ This looks like REAL news")
        else:
            st.error("❌ This looks like FAKE news")
    else:
        st.warning("Please enter some text")