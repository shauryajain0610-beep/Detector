import streamlit as st
import pickle
import urllib.parse

from Orange.data import Table, Domain, StringVariable
from Orange.preprocess.text import preprocess_strings

# ------------------------------------------------------------
# LOAD ORANGE MODEL
# ------------------------------------------------------------
with open("orange_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Matching domain used during training
domain = Domain([StringVariable("text")], class_vars=StringVariable("label"))

# Prediction function
def orange_predict(text):
    processed = preprocess_strings([text])
    test_table = Table(domain, processed)
    pred = model(test_table)
    return str(pred[0])


# ------------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Header + Logo
st.image("logo.png", width=120)
st.title("📰 Fake News Detector using ORANGE AI")
st.write("Analyze any headline or full article to detect if it's Fake or Real.")


# ------------------------------------------------------------
# SMART REASONING
# ------------------------------------------------------------
def generate_reasoning(prediction):
    if prediction == "FAKE":
        return (
            "The news looks suspicious and contains signs often seen in misinformation:\n"
            "- Sensational or emotional tone\n"
            "- Missing credible verification sources\n"
            "- Exaggerated or unrealistic statements\n"
            "- Designed to provoke strong reactions\n"
        )
    else:
        return (
            "The news appears more balanced and trustworthy:\n"
            "- Language is factual and controlled\n"
            "- Statements appear verifiable\n"
            "- Context seems realistic and coherent\n"
        )

# ------------------------------------------------------------
# SMART ADVICE
# ------------------------------------------------------------
def generate_advice(prediction):
    if prediction == "FAKE":
        return (
            "❗ **Advice for FAKE News:**\n"
            "- Do NOT forward or share immediately.\n"
            "- Verify using trusted fact-checking resources.\n"
            "- Check whether other reliable media outlets reported the same story.\n"
        )
    else:
        return (
            "✔ **Advice for REAL News:**\n"
            "- Still verify from the official media websites.\n"
            "- Avoid sharing unverified screenshots or forwards.\n"
        )

# ------------------------------------------------------------
# EXTERNAL VERIFICATION SOURCES
# ------------------------------------------------------------
def generate_links(query):
    encoded = urllib.parse.quote(query)
    return {
        "Google News Search": f"https://news.google.com/search?q={encoded}",
        "BBC Search": f"https://www.bbc.co.uk/search?q={encoded}",
        "Alt News Fact Check": f"https://www.altnews.in/?s={encoded}",
        "BOOMLive Fact Check": f"https://www.boomlive.in/search?query={encoded}"
    }


# ------------------------------------------------------------
# INPUT AREA
# ------------------------------------------------------------
st.subheader("Choose Input Type")

choice = st.radio(
    "Select what you want to analyze:",
    ["News Headline", "Full Article"]
)

if choice == "News Headline":
    user_input = st.text_input("Enter your headline below:")
else:
    user_input = st.text_area("Enter your full article below:", height=200)


# ------------------------------------------------------------
# ACTION BUTTON
# ------------------------------------------------------------
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠ Please enter some text first.")
    else:
        # Model Prediction
        prediction = orange_predict(user_input)

        # Result
        st.subheader("🧪 Prediction Result")
        if prediction == "FAKE":
            st.error("❌ This news appears to be FAKE.")
        else:
            st.success("✔ This news appears to be REAL.")

        # Reasoning
        st.subheader("🧠 Detailed Reasoning")
        st.write(generate_reasoning(prediction))

        # Advice
        st.subheader("💡 Smart Advice")
        st.info(generate_advice(prediction))

        # External References
        st.subheader("🌍 Fact Verification Links")
        links = generate_links(user_input)
        for name, url in links.items():
            st.write(f"- [{name}]({url})")

        st.success("🙏 THANK YOU FOR USING THIS APP! Stay aware, stay smart, stay safe ❤️")
