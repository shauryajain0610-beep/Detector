import streamlit as st
import pandas as pd
import joblib
import os
import urllib.parse

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Detect whether a news headline or article is Real or Fake, and get reasoning, advice, and related sources.")

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "models.pkcls")
TEST_PATH = os.path.join(BASE_DIR, "data", "test_headlines_20")

# -----------------------------
# LOAD PRE-TRAINED MODEL
# -----------------------------
@st.cache_data
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------
# LOAD TEST DATA (optional)
# -----------------------------
@st.cache_data
def load_test_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not load test data: {e}")
        return None

test_df = load_test_data(TEST_PATH)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def classify_text(text):
    if not model:
        return "Model not loaded"
    
    try:
        prediction = model.predict([text])  # sklearn-like interface
        return prediction[0]
    except Exception as e:
        return f"Error during prediction: {e}"

# -----------------------------
# REASONING FUNCTION
# -----------------------------
def generate_reasoning(text, prediction):
    fake_keywords = ["click", "shocking", "secret", "you won't believe", "breaking"]
    real_keywords = ["reports", "official", "announced", "confirmed", "statement"]

    text_lower = text.lower()
    if prediction == "Fake":
        reasons = [kw for kw in fake_keywords if kw in text_lower]
        return "The headline/article contains sensational or misleading terms: " + ", ".join(reasons) if reasons else "The text seems suspicious or exaggerated."
    else:
        reasons = [kw for kw in real_keywords if kw in text_lower]
        return "The headline/article contains credible, official, or verified terms: " + ", ".join(reasons) if reasons else "The text appears factual and credible."

# -----------------------------
# ADVICE FUNCTION
# -----------------------------
def generate_advice(prediction):
    if prediction == "Fake":
        return "Be cautious before sharing. Verify with trusted sources and avoid spreading misinformation."
    else:
        return "You can trust this news, but always cross-check with official outlets if needed."

# -----------------------------
# EXTERNAL SOURCES FUNCTION
# -----------------------------
def generate_sources(text):
    # Simple example: search the headline on Google News
    query = urllib.parse.quote(text)
    url = f"https://news.google.com/search?q={query}"
    return url

# -----------------------------
# USER INPUT
# -----------------------------
option = st.radio("Choose input type:", ("Headline", "Full Article"))

input_text = st.text_area(f"Enter your {option.lower()} here:")

if st.button("Check News"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify!")
    else:
        result = classify_text(input_text)
        st.success(f"Prediction: **{result}**")

        reasoning = generate_reasoning(input_text, result)
        advice = generate_advice(result)
        source_url = generate_sources(input_text)

        st.subheader("Reasoning")
        st.write(reasoning)

        st.subheader("Advice")
        st.write(advice)

        st.subheader("Related Sources")
        st.markdown(f"[Click here to check related news]({source_url})", unsafe_allow_html=True)

# -----------------------------
# OPTIONAL: Show sample test headlines
# -----------------------------
if st.checkbox("Show sample test headlines"):
    if test_df is not None:
        st.dataframe(test_df.head(20))
    else:
        st.info("Test dataset not available.")
