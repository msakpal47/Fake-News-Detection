import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK stopwords once
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and preprocess dataset
@st.cache_data
def load_and_prepare_data(fake_file, real_file):
    if fake_file is None or real_file is None:
        st.error("Please upload both CSV files.")
        st.stop()

    fake_df = pd.read_csv(fake_file)
    real_df = pd.read_csv(real_file)

    fake_df['label'] = 1  # Fake
    real_df['label'] = 0  # Real

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Choose the column to use (text or title)
    if 'text' in df.columns:
        df = df[['text', 'label']].dropna()
    elif 'title' in df.columns:
        df = df[['title', 'label']].rename(columns={'title': 'text'}).dropna()
    else:
        st.error("Dataset must have a 'text' or 'title' column.")
        st.stop()

    # Apply preprocessing
    df['text'] = df['text'].apply(preprocess_text)
    return df

# Train and return model, vectorizer, and accuracy
@st.cache_resource
def train_model(df, model_type="Logistic"):
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    if model_type == "Logistic":
        model = LogisticRegression()
    else:
        model = MultinomialNB()

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, accuracy

# Streamlit UI
def main():
    st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
    st.title("📰 Fake News Detector")
    st.write("Detect whether a news article is **Real** ✅ or **Fake** ❌ using machine learning.")

    model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Naive Bayes"])
    model_type = "Logistic" if model_choice == "Logistic Regression" else "NaiveBayes"

    # File Upload
    fake_file = st.file_uploader("Upload Fake News CSV", type=["csv"])
    real_file = st.file_uploader("Upload Real News CSV", type=["csv"])

    if fake_file is not None and real_file is not None:
        with st.spinner("🔄 Loading and training model..."):
            df = load_and_prepare_data(fake_file, real_file)
            model, vectorizer, accuracy = train_model(df, model_type)

        st.success(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

        st.subheader("🔍 Enter News Text to Predict")
        user_input = st.text_area("Paste news article or headline:")

        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text to predict.")
            else:
                cleaned_input = preprocess_text(user_input)
                transformed_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(transformed_input)[0]
                result = "❌ Fake News" if prediction == 1 else "✅ Real News"
                st.subheader(f"Prediction: {result}")

if __name__ == "__main__":
    main()
