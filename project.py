import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

def predict_complaint(model, embedder, complaint_text):
    X = embedder.encode([complaint_text])
    return model.predict(X)[0]

st.set_page_config(page_title="Smart Complaint Classifier", layout="centered")
st.title("📝 Smart Public Complaint Classifier ✍️")

@st.cache_resource
def setup_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    model_cat = joblib.load("models/category_model.pkl")
    model_pri = joblib.load("models/priority_model.pkl")
    label_map_cat = joblib.load("models/category_model_labels.pkl")
    label_map_pri = joblib.load("models/priority_model_labels.pkl")
    df = pd.read_csv("citizen_complaints.csv")
    return model_cat, label_map_cat, model_pri, label_map_pri, embedder, df

model_cat, label_map_cat, model_pri, label_map_pri, embedder, df = setup_models()

with st.form("complaint_form"):
    complaint = st.text_area("✍️ Enter your complaint:")
    submitted = st.form_submit_button("🔍 Enter")

    if submitted and complaint.strip():
        department_code = predict_complaint(model_cat, embedder, complaint)
        priority_code = predict_complaint(model_pri, embedder, complaint)

        department = label_map_cat[department_code]
        priority = label_map_pri[priority_code]

        st.success(f"🏢 Predicted Department: {department}")
        st.info(f"⚡ Predicted Priority: {priority}")

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "complaint": complaint,
            "department": department,
            "priority": priority
        })

# Show history and sample data after first prediction
if "history" in st.session_state and st.session_state.history:
    st.markdown("### 🕘 Previous Predictions")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

    with st.expander("📄 See Example Data"):
        st.dataframe(df[["complaint_text", "category", "priority"]].sample(10), use_container_width=True)
