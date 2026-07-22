import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import joblib
import os

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="StrokeInsight | Stroke Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.main{
background-color:#f5f7fb;
}

h1{
color:#0F4C81;
font-family:Segoe UI;
}

h2{
color:#1F2937;
}

.metric-card{
background:white;
padding:15px;
border-radius:10px;
box-shadow:0px 3px 8px rgba(0,0,0,0.15);
}

.stPlotlyChart{
background:white;
padding:10px;
border-radius:10px;
box-shadow:0px 3px 8px rgba(0,0,0,0.15);
}

</style>

""",unsafe_allow_html=True)

# ================= TITLE =================

st.title("🧠 StrokeInsight - Stroke Prediction Dashboard")

st.markdown(
"### Professional Stroke Analytics & Machine Learning Dashboard"
)

# ================= LOAD DATA =================

@st.cache_data
def load_data():

    df = pd.read_csv(r"D:\Mlproject\Stroke Detection\healthcare-dataset-stroke-data.csv")

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    return df

df = load_data()

# ================= SIDEBAR =================

st.sidebar.header("Navigation")

section = st.sidebar.radio(
"Go To",

[
"📊 Overview",

"📈 Exploratory Analysis",

"🔬 Statistical Tests",

"🤖 Machine Learning",

"🔮 Live Prediction"
]

)

# ===================================================
# OVERVIEW
# ===================================================

if section=="📊 Overview":

    st.header("Dataset Overview")

    col1,col2,col3,col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Patients",
            len(df)
        )

    with col2:

        st.metric(
            "Stroke Cases",
            df["stroke"].sum()
        )

    with col3:

        st.metric(
            "Average Age",
            round(df["age"].mean(),1)
        )

    with col4:

        rate = df["stroke"].mean()*100

        st.metric(
            "Stroke Rate",
            f"{rate:.2f}%"
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Information")

    c1,c2=st.columns(2)

    with c1:

        st.write("Shape")

        st.success(df.shape)

        st.write("Missing Values")

        st.write(df.isnull().sum())

    with c2:

        st.write("Data Types")

        st.write(df.dtypes)

        st.write("Summary Statistics")

        st.dataframe(df.describe())

# ===================================================
# NEXT PART PLACEHOLDER
# ===================================================


# ===================================================
# EXPLORATORY DATA ANALYSIS
# ===================================================

elif section == "📈 Exploratory Analysis":

    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Distributions", "Correlation", "Stroke Analysis"]
    )

    # -------------------------------
    # Distribution Plots
    # -------------------------------
    with tab1:

        variable = st.selectbox(
            "Select Numerical Variable",
            ["age", "avg_glucose_level", "bmi"]
        )

        fig = px.histogram(
            df,
            x=variable,
            color="stroke",
            marginal="box",
            nbins=30,
            title=f"{variable} Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        fig = px.box(
            df,
            y=variable,
            color="stroke",
            title=f"{variable} Box Plot"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Correlation
    # -------------------------------
    with tab2:

        st.subheader("Correlation Heatmap")

        numeric = df[
            [
                "age",
                "hypertension",
                "heart_disease",
                "avg_glucose_level",
                "bmi",
                "stroke"
            ]
        ]

        corr = numeric.corr()

        fig = px.imshow(
            corr.round(2),
            text_auto=True,
            color_continuous_scale="RdBu"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Stroke Analysis
    # -------------------------------
    with tab3:

        st.subheader("Stroke Distribution")

        fig = px.pie(
            df,
            names="stroke",
            title="Stroke Cases"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Stroke by Gender")

        fig = px.histogram(
            df,
            x="gender",
            color="stroke",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Stroke by Work Type")

        fig = px.histogram(
            df,
            x="work_type",
            color="stroke",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# STATISTICAL TESTS
# ===================================================

elif section == "🔬 Statistical Tests":

    st.header("Statistical Hypothesis Testing")

    test = st.selectbox(

        "Choose Test",

        [

            "T-Test : BMI by Gender",

            "ANOVA : Age by Work Type",

            "Chi-Square : Stroke vs Hypertension"

        ]

    )

    # ---------------------------------
    # T-Test
    # ---------------------------------

    if test == "T-Test : BMI by Gender":

        male = df[df["gender"] == "Male"]["bmi"]

        female = df[df["gender"] == "Female"]["bmi"]

        t_stat, p = stats.ttest_ind(

            male,

            female,

            nan_policy="omit"

        )

        st.success(f"T Statistic : {t_stat:.4f}")

        st.success(f"P Value : {p:.6f}")

        if p < 0.05:

            st.info("There is a significant difference between Male and Female BMI.")

        else:

            st.warning("No significant difference between Male and Female BMI.")

    # ---------------------------------
    # ANOVA
    # ---------------------------------

    elif test == "ANOVA : Age by Work Type":

        groups = [

            group["age"].values

            for _, group in df.groupby("work_type")

        ]

        f_stat, p = stats.f_oneway(*groups)

        st.success(f"F Statistic : {f_stat:.4f}")

        st.success(f"P Value : {p:.6f}")

        if p < 0.05:

            st.info("Age differs significantly across work types.")

        else:

            st.warning("No significant difference among work types.")

    # ---------------------------------
    # Chi Square
    # ---------------------------------

    else:

        contingency = pd.crosstab(

            df["hypertension"],

            df["stroke"]

        )

        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        st.success(f"Chi Square : {chi2:.4f}")

        st.success(f"P Value : {p:.6f}")

        st.dataframe(contingency)

        if p < 0.05:

            st.info("Stroke is associated with Hypertension.")

        else:

            st.warning("No significant association detected.")

            # ===================================================
# MACHINE LEARNING
# ===================================================

elif section == "🤖 Machine Learning":

    st.header("Machine Learning Models")

    model_type = st.radio(
        "Select Model",
        [
            "Logistic Regression",
            "Random Forest"
        ]
    )

    # -------------------------
    # Data Preprocessing
    # -------------------------

    data = df.copy()

    data = pd.get_dummies(
        data,
        columns=[
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status"
        ],
        drop_first=True
    )

    X = data.drop(["id", "stroke"], axis=1)

    y = data["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ==========================================
    # Logistic Regression
    # ==========================================

    if model_type == "Logistic Regression":

        model = LogisticRegression(

            max_iter=1000,

            random_state=42

        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Accuracy",

                f"{accuracy:.4f}"

            )

        with col2:

            roc = roc_auc_score(

                y_test,

                pred

            )

            st.metric(

                "ROC AUC",

                f"{roc:.4f}"

            )

        st.subheader("Classification Report")

        st.text(

            classification_report(

                y_test,

                pred

            )

        )

        cm = confusion_matrix(

            y_test,

            pred

        )

        fig = px.imshow(

            cm,

            text_auto=True,

            color_continuous_scale="Blues",

            labels=dict(

                x="Predicted",

                y="Actual"

            )

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        if st.button("💾 Save Logistic Regression Model"):

            joblib.dump(

                model,

                "stroke_model.pkl"

            )

            joblib.dump(

                scaler,

                "scaler.pkl"

            )

            joblib.dump(

                X.columns.tolist(),

                "features.pkl"

            )

            st.success("Model Saved Successfully!")

    # ==========================================
    # Random Forest
    # ==========================================

    else:

        model = RandomForestClassifier(

            n_estimators=200,

            random_state=42

        )

        model.fit(

            X_train,

            y_train

        )

        pred = model.predict(

            X_test

        )

        accuracy = accuracy_score(

            y_test,

            pred

        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Accuracy",

                f"{accuracy:.4f}"

            )

        with col2:

            roc = roc_auc_score(

                y_test,

                pred

            )

            st.metric(

                "ROC AUC",

                f"{roc:.4f}"

            )

        st.subheader("Classification Report")

        st.text(

            classification_report(

                y_test,

                pred

            )

        )

        cm = confusion_matrix(

            y_test,

            pred

        )

        fig = px.imshow(

            cm,

            text_auto=True,

            color_continuous_scale="Greens"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        importance = pd.DataFrame(

            {

                "Feature": X.columns,

                "Importance": model.feature_importances_

            }

        )

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        )

        st.subheader("Top 10 Important Features")

        fig = px.bar(

            importance.head(10),

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        if st.button("💾 Save Random Forest Model"):

            joblib.dump(

                model,

                "stroke_model.pkl"

            )

            joblib.dump(

                scaler,

                "scaler.pkl"

            )

            joblib.dump(

                X.columns.tolist(),

                "features.pkl"

            )

            st.success("Random Forest Model Saved Successfully!")

            # ===================================================
# LIVE STROKE PREDICTION
# ===================================================

elif section == "🔮 Live Prediction":

    st.header("🧠 Live Stroke Risk Prediction")
try:
    model = joblib.load("stroke_model.pkl")
    feature_columns = joblib.load("features.pkl")

except Exception as e:
    st.error(e)
    st.stop()


    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        age = st.slider(
            "Age",
            1,
            100,
            40
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0,1]
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0,1]
        )

        ever_married = st.selectbox(
            "Ever Married",
            ["Yes","No"]
        )

    with col2:

        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )

        residence = st.selectbox(
            "Residence Type",
            [
                "Urban",
                "Rural"
            ]
        )

        glucose = st.number_input(
            "Average Glucose Level",
            value=100.0
        )

        bmi = st.number_input(
            "BMI",
            value=25.0
        )

        smoking = st.selectbox(
            "Smoking Status",
            [
                "formerly smoked",
                "never smoked",
                "smokes",
                "Unknown"
            ]
        )

    if st.button("🚀 Predict Stroke Risk", type="primary"):

        input_data = pd.DataFrame({

            "gender":[gender],
            "age":[age],
            "hypertension":[hypertension],
            "heart_disease":[heart_disease],
            "ever_married":[ever_married],
            "work_type":[work_type],
            "Residence_type":[residence],
            "avg_glucose_level":[glucose],
            "bmi":[bmi],
            "smoking_status":[smoking]

        })

        input_data = pd.get_dummies(

            input_data,

            columns=[
                "gender",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status"
            ],

            drop_first=True

        )

        input_data = input_data.reindex(

            columns=feature_columns,

            fill_value=0

        )

        input_scaled = input_data

        prediction = model.predict(input_scaled)

        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error("⚠ High Risk of Stroke")

        else:

            st.success("✅ Low Risk of Stroke")

        st.metric(

            "Stroke Probability",

            f"{probability*100:.2f}%"

        )

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={"text":"Stroke Risk %"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"red"},

                    "steps":[

                        {"range":[0,30],"color":"green"},

                        {"range":[30,60],"color":"yellow"},

                        {"range":[60,100],"color":"red"}

                    ]

                }

            )

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.info("""

### Recommendation

✅ Maintain a healthy diet

✅ Exercise regularly

✅ Control blood pressure

✅ Monitor blood glucose

✅ Avoid smoking

✅ Consult a healthcare professional if symptoms persist.

""")