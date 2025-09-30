# app.py
# Streamlit app for interactive prediction and simple EDA for the Iris dataset

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

MODEL_PATH = "iris_rf_model.joblib"

st.set_page_config(page_title="Iris Classifier", layout="wide")

@st.cache_data
def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data["model"], data["feature_names"], data["target_names"]

@st.cache_data
def load_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    return df, iris

model, feature_names, target_names = load_model()
df, iris = load_dataset()

# Sidebar controls
st.sidebar.title("Controls")
mode = st.sidebar.radio("Choose mode", ("Prediction", "Data exploration"))

st.title("🌸 Iris Classification — Interactive App")
st.markdown(
    "This demo lets you explore the Iris dataset and make predictions using a trained Random Forest model.\n"
    "Use the sidebar to switch between Prediction and Data exploration modes."
)

if mode == "Prediction":
    st.header("Make a prediction")

    with st.container():
        col1, col2 = st.columns(2)

        # Build input widgets dynamically from feature names
        inputs = {}
        with col1:
            st.subheader("Input features")
            for fname in feature_names[:2]:
                min_val = float(df[fname].min())
                max_val = float(df[fname].max())
                mean_val = float(df[fname].mean())
                inputs[fname] = st.slider(
                    label=f"{fname}",
                    min_value=round(min_val, 2),
                    max_value=round(max_val, 2),
                    value=round(mean_val, 2),
                    step=0.01,
                    help=f"{fname}: range {min_val:.2f} — {max_val:.2f}"
                )
        with col2:
            st.subheader("Input features (cont.)")
            for fname in feature_names[2:]:
                min_val = float(df[fname].min())
                max_val = float(df[fname].max())
                mean_val = float(df[fname].mean())
                inputs[fname] = st.slider(
                    label=f"{fname}",
                    min_value=round(min_val, 2),
                    max_value=round(max_val, 2),
                    value=round(mean_val, 2),
                    step=0.01,
                    help=f"{fname}: range {min_val:.2f} — {max_val:.2f}"
                )

    # Arrange for prediction
    input_df = pd.DataFrame([inputs])

    if st.button("Predict"):
        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]

        label = target_names[pred_class]
        conf = pred_proba[pred_class]

        # Color coding
        color_map = {0: "#6EE7B7", 1: "#FDE68A", 2: "#FCA5A5"}
        col = color_map.get(pred_class, "white")

        st.markdown(f"### Prediction: <span style='color:black;background:{col};padding:6px;border-radius:6px'>{label}</span>", unsafe_allow_html=True)
        st.write(f"Confidence: {conf*100:.2f}%")

        proba_df = pd.DataFrame({"class": target_names, "probability": pred_proba})
        st.bar_chart(proba_df.set_index("class"))

        st.subheader("Model explanation (feature values used)")
        st.table(input_df.T.rename(columns={0: "value"}))

else:
    st.header("Data exploration")
    st.markdown("Use the controls below to change which features are shown.")

    # Sidebar options for EDA
    plot_type = st.sidebar.radio("Select plot type", ("Scatter plot", "Histogram"))

    if plot_type == "Scatter plot":
        feature_x = st.sidebar.selectbox("X feature", feature_names, index=0)
        feature_y = st.sidebar.selectbox("Y feature", feature_names, index=1)
        show_grid = st.sidebar.checkbox("Show grid on plot", value=True)

        st.subheader(f"Scatter: {feature_x} vs {feature_y}")
        fig, ax = plt.subplots()
        for i, tname in enumerate(iris.target_names):
            subset = df[df['target'] == i]
            ax.scatter(subset[feature_x], subset[feature_y], label=tname, s=50, alpha=0.7)
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.legend()
        if show_grid:
            ax.grid(True)
        st.pyplot(fig)

    elif plot_type == "Histogram":
        hist_feature = st.sidebar.selectbox("Histogram feature", feature_names, index=0)
        show_grid = st.sidebar.checkbox("Show grid on plot", value=True)

        st.subheader(f"Histogram: {hist_feature}")
        fig2, ax2 = plt.subplots()
        ax2.hist([df[df['target'] == i][hist_feature] for i in range(3)], bins=15, label=iris.target_names, stacked=False)
        ax2.set_xlabel(hist_feature)
        ax2.set_ylabel("Count")
        ax2.legend()
        if show_grid:
            ax2.grid(True)
        st.pyplot(fig2)

    # Show data table
    with st.expander("Show raw dataset"):
        st.dataframe(df)

# Footer / tips
st.markdown("---")
st.caption("Tip: To use your pre-saved model file instead of training again, place `iris_rf_model.joblib` in the same folder as this app.")
