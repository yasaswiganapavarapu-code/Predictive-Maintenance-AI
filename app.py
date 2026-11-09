import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "best_predictive_maintenance_model.joblib"

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f8ff 0%, #edf4f8 100%); }
    .block-container { padding-top: 2rem; max-width: 1150px; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white; margin-bottom: 1.3rem;
    }f
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #e4f4fb; font-size: 1.05rem; margin-bottom: 0; }
    .small-note { color: #526579; font-size: .9rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #dce7ef;
        padding: 1rem; border-radius: 14px;
    }
    div.stButton > button {
        width: 100%; border-radius: 12px; min-height: 3rem;
        font-weight: 650;
    }
    /* Fix text visibility */
.stApp,
.stApp p,
.stApp label,
.stApp h2,
.stApp h3 {
    color: #263445 !important;
}

.stNumberInput input {
    color: #ffffff !important;
}

.stSelectbox div[data-baseweb="select"] {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚙️ Predictive Maintenance AI</h1>
  <p>Classify possible industrial machine failure types from operating conditions.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter a machine's operating readings below. The trained classification pipeline "
    "will estimate the most likely class from the model's learned patterns."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model file not found. Put "
        "'best_predictive_maintenance_model.joblib' in the same folder as app.py. "
        "It was created by your training script."
    )
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.markdown("### Machine operating conditions")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=0,
                                    help="L = low, M = medium, H = high product quality category in this dataset.")
        air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0,
                                   value=298.1, step=0.1)
    with col2:
        process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=400.0,
                                       value=308.6, step=0.1)
        rotational_speed = st.number_input("Rotational speed (rpm)", min_value=0, max_value=10000,
                                           value=1500, step=10)
    with col3:
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0,
                                 value=40.0, step=0.5)
        tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=500,
                                    value=10, step=1)
    submitted = st.form_submit_button("🔍 Predict failure type")

if submitted:
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])
    try:
        prediction = model.predict(input_data)[0]
        st.markdown("---")
        st.markdown("### Prediction result")
        if str(prediction).lower() == "no failure":
            st.success(f"Predicted class: **{prediction}**")
            st.write("The model classified these readings as **No Failure**.")
        else:
            st.warning(f"Predicted class: **{prediction}**")
            st.write(
                "The model classified these readings as a possible failure type. "
                "Treat this as a screening result, not a confirmed diagnosis of equipment condition."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({
                "Failure class": classes,
                "Model score (%)": (probabilities * 100).round(2),
            }).sort_values("Model score (%)", ascending=False)
            st.markdown("#### Model scores by class")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.caption(
                "These are model probability estimates, not calibrated guarantees of real-world failure risk."
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the saved model was created from the same dataset columns "
            "and that the feature names in app.py match the training script."
        )

st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Project type", "Multiclass classification")
m2.metric("Model", "Decision Tree")
m3.metric("Input features", "6")
st.markdown(
    '<p class="small-note">Academic demonstration only. This model was trained on historical dataset examples; '
    'it is not safety-certified and should not be used as the sole basis for industrial maintenance decisions.</p>',
    unsafe_allow_html=True,
)
