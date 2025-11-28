import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Choose input type and analyze whether the content seems Real or Fake.")

# -----------------------------
# SIMPLE RULE-BASED CLASSIFIER
# -----------------------------
def classify_news(text):
    text = text.lower()

    fake_keywords = [
        "shocking", "secret", "breaking!!!", "miracle",
        "unbelievable", "banned", "hidden truth", "exposed",
        "100% guarantee", "cure", "conspiracy"
    ]

    score = sum(word in text for word in fake_keywords)

    if score >= 2:
        return "Fake"
    elif score == 1:
        return "Possibly Fake"
    else:
        return "Real"


# -----------------------------
# STREAMLIT INPUT AREA
# -----------------------------
st.header("📝 Select Input Type")
choice = st.radio("Select what you want to enter:", ["Headline Only", "Full Article"])

headline = ""
article = ""

if choice == "Headline Only":
    headline = st.text_input("Enter News Headline")
else:
    headline = st.text_input("Headline (optional)")
    article = st.text_area("Enter Full Article Text", height=180)

if st.button("Analyze"):
    combined_text = (headline + " " + article).strip()

    if not combined_text:
        st.warning("⚠ Please enter some text first.")
    else:
        prediction = classify_news(combined_text)

        # -----------------------------
        # DYNAMIC REASONING & ADVICE
        # -----------------------------
        if prediction == "Fake":
            reasoning = "The text contains several sensational or misleading keywords, which indicate a high likelihood of misinformation."
            advice = """
            ❌ **Advice if Fake:**  
            - Immediately verify using trusted fact-checking sites  
            - Do NOT share this information unless verified  
            - Look for official government or credible news sources  
            """

            sources = """
            - [Alt News](https://www.altnews.in/)  
            - [BOOM Fact Check](https://www.boomlive.in/)  
            - [Factly](https://factly.in/)  
            """

        elif prediction == "Possibly Fake":
            reasoning = "At least one suspicious keyword is detected. It may or may not be accurate, but requires verification."
            advice = """
            ⚠ **Advice if Possibly Fake:**  
            - Cross-check with multiple reliable news outlets  
            - Check publication time and author credibility  
            - Search if reputed media outlets covered the same story  
            """

            sources = """
            - [Google Fact Check Explorer](https://toolbox.google.com/factcheck/explorer)  
            - [Snopes](https://www.snopes.com/)  
            """

        else:
            reasoning = "There are no common signals of misinformation or exaggerated keywords detected."
            advice = """
            ✔ **Advice if Real:**  
            - Still check original source for any updates  
            - Share responsibly from official/reputed outlets  
            - Verify facts from authentic government or national agencies  
            """

            sources = """
            - [Reuters Official News](https://www.reuters.com/)  
            - [BBC News](https://www.bbc.com/)  
            - [The Hindu](https://www.thehindu.com/)  
            """

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("🔍 Prediction")

        if prediction == "Fake":
            st.error("❌ FAKE NEWS")
        elif prediction == "Possibly Fake":
            st.warning("⚠️ POSSIBLY FAKE")
        else:
            st.success("✔ REAL NEWS")

        st.subheader("🧠 Reasoning")
        st.write(reasoning)

        st.subheader("💡 Smart Advice")
        st.write(advice)

        st.subheader("🔗 Trusted Verification Sources")
        st.markdown(sources)

        st.info("This rule-based model is for demonstration. Connect ML model for real accuracy.")

