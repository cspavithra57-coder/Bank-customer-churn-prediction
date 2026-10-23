
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bank Customer Churn Prediction")
st.write(
    "Upload a bank customer CSV file to explore the data, train machine-learning "
    "models, compare their performance, and generate customer churn risk scores."
)

REQUIRED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
OPTIONAL_UNUSED_COLUMNS = ["Year", "CustomerId", "Surname"]


def prepare_data(raw_df):
    """Validate and prepare the uploaded dataset."""
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing)
        )

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["Exited"]).copy()

    # Convert target safely and keep only binary target records.
    df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")
    df = df.dropna(subset=["Exited"]).copy()
    df = df[df["Exited"].isin([0, 1])].copy()
    df["Exited"] = df["Exited"].astype(int)

    if df.empty:
        raise ValueError("No valid records remain after checking the Exited column.")

    # Remove identifiers and time field when present, to avoid using them as predictors.
    df = df.drop(columns=[c for c in OPTIONAL_UNUSED_COLUMNS if c in df.columns],
                 errors="ignore")

    # Numeric columns expected by this project.
    expected_numeric = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    for col in expected_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric missing values with median and categorical values with mode.
    for col in expected_numeric:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_value) else median_value)

    for col in ["Geography", "Gender"]:
        df[col] = df[col].astype("string")
        df[col] = df[col].replace(["<NA>", "nan", "None"], np.nan)
        if df[col].isna().any():
            modes = df[col].mode(dropna=True)
            df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "Unknown")
        df[col] = df[col].astype(str)

    # Feature engineering used in the original notebook.
    salary_denominator = df["EstimatedSalary"].replace(0, np.nan)
    df["Balance_Salary_Ratio"] = (
        df["Balance"] / salary_denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product_Interaction"] = (
        df["IsActiveMember"] * df["NumOfProducts"]
    )
    df["Age_Tenure_Interaction"] = df["Age"] * df["Tenure"]

    return df


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }


def train_and_evaluate(df, test_size, random_state):
    X = df.drop(columns=["Exited"]).copy()
    y = df["Exited"].copy()

    if y.nunique() < 2:
        raise ValueError(
            "The Exited column must contain both 0 (stayed) and 1 (churned) "
            "to train and evaluate classification models."
        )
    if y.value_counts().min() < 2:
        raise ValueError(
            "Each Exited class needs at least two records for a stratified train/test split."
        )

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = build_models()
    trained_models, predictions, probabilities, rows = {}, {}, {}, []

    progress = st.progress(0, text="Training models...")
    for idx, (name, model) in enumerate(models.items(), start=1):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        trained_models[name] = pipeline
        predictions[name] = y_pred
        probabilities[name] = y_prob
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })
        progress.progress(idx / len(models), text=f"Trained {name}")
    progress.empty()

    results_df = pd.DataFrame(rows).sort_values(
        "ROC-AUC", ascending=False
    ).reset_index(drop=True)

    return X, y, X_test, y_test, trained_models, predictions, probabilities, results_df


def make_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


uploaded_file = st.file_uploader(
    "Upload your bank customer dataset (CSV)",
    type=["csv"],
    help="The file should contain fields such as CreditScore, Geography, Gender, Age, and Exited."
)

with st.sidebar:
    st.header("Model settings")
    test_size_percent = st.slider(
        "Test set size (%)", min_value=10, max_value=40, value=20, step=5
    )
    random_state = st.number_input(
        "Random state", min_value=0, max_value=99999, value=42, step=1
    )
    st.caption("A fixed random state makes the split reproducible.")

if uploaded_file is None:
    st.info("Upload your CSV file to start. The app does not require Google Colab.")
    st.markdown("**Expected target column:** `Exited` (0 = stayed, 1 = churned).")
    st.markdown("**Required fields:** " + ", ".join(REQUIRED_COLUMNS))
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file)
    data = prepare_data(raw_data)
except Exception as exc:
    st.error(f"Could not read or prepare the dataset: {exc}")
    st.stop()

st.success("Dataset uploaded and validated successfully.")

tab_data, tab_eda, tab_models, tab_risk = st.tabs([
    "Dataset", "Exploratory Analysis", "Model Results", "Churn Risk Scores"
])

with tab_data:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", f"{len(data):,}")
    c2.metric("Columns", f"{data.shape[1]:,}")
    c3.metric("Overall churn rate", f"{data['Exited'].mean() * 100:.2f}%")

    st.write("Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.write("Missing values")
    missing_df = data.isna().sum().rename("Missing Values").to_frame()
    st.dataframe(missing_df, use_container_width=True)

    st.download_button(
        "Download cleaned and feature-engineered dataset",
        data=make_csv_download(data),
        file_name="European_Bank_Feature_Engineered.csv",
        mime="text/csv"
    )

with tab_eda:
    st.subheader("Exploratory data analysis")
    churn_counts = data["Exited"].map({0: "Stayed", 1: "Exited"}).value_counts()
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_counts.plot(kind="bar", ax=ax)
        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Customer status")
        ax.set_ylabel("Number of customers")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        if "IsActiveMember" in data.columns:
            active = data.groupby("IsActiveMember")["Exited"].agg(
                Customers="size", Exited="sum", Churn_Rate="mean"
            ).reset_index()
            active["Membership"] = active["IsActiveMember"].map(
                {0: "Inactive", 1: "Active"}
            ).fillna(active["IsActiveMember"].astype(str))
            active["Churn Rate (%)"] = active["Churn_Rate"] * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=active, x="Membership", y="Churn Rate (%)", ax=ax)
            ax.set_title("Churn Rate by Active Membership")
            ax.set_xlabel("Membership")
            ax.set_ylabel("Churn rate (%)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.shape[1] >= 2:
        st.write("Numeric feature correlation")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "Age" in data.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data, x="Age", hue="Exited", bins=25, multiple="stack", ax=ax)
        ax.set_title("Age Distribution by Churn Status")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with st.spinner("Training and evaluating machine-learning models..."):
    try:
        (
            X, y, X_test, y_test, trained_models, predictions,
            probabilities, results_df
        ) = train_and_evaluate(
            data, test_size_percent / 100, int(random_state)
        )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

with tab_models:
    st.subheader("Model comparison")
    st.caption("Models are ranked by ROC-AUC, which measures how well the model ranks churned customers above non-churned customers.")
    display_results = results_df.copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(
        display_results.style.format({col: "{:.4f}" for col in metric_cols}),
        use_container_width=True
    )

    st.metric("Top model by ROC-AUC", best_model_name,
              f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    results_df.set_index("Model")[metric_cols].plot(kind="bar", ax=ax)
    ax.set_title("Comparison of Churn Prediction Models")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download model comparison CSV",
        data=make_csv_download(results_df),
        file_name="Chapter_7_Model_Comparison.csv",
        mime="text/csv"
    )

    selected_name = st.selectbox("View detailed evaluation for", list(trained_models.keys()))
    cm = confusion_matrix(y_test, predictions[selected_name], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed (0)", "Churned (1)"],
        yticklabels=["Stayed (0)", "Churned (1)"], ax=ax
    )
    ax.set_title(f"Confusion Matrix - {selected_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        y_test, predictions[selected_name], labels=[0, 1],
        target_names=["Stayed", "Churned"], zero_division=0
    )
    st.text("Classification report\n" + report)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in trained_models:
        RocCurveDisplay.from_predictions(
            y_test, probabilities[name], name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classification")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_risk:
    st.subheader("Customer churn risk scoring")
    st.write(
        f"Risk scores below are generated by **{best_model_name}**, the model with "
        "the highest ROC-AUC in this run. These scores should support human review, "
        "not make automatic customer decisions."
    )

    risk_data = X_test.copy().reset_index(drop=True)
    risk_data["Actual_Exited"] = y_test.reset_index(drop=True)
    risk_data["Predicted_Churn_Probability"] = best_model.predict_proba(X_test)[:, 1]
    risk_data["Predicted_Exited"] = best_model.predict(X_test)

    def assign_risk(probability):
        if probability < 0.30:
            return "Low Risk"
        elif probability < 0.60:
            return "Medium Risk"
        return "High Risk"

    risk_data["Churn_Risk_Category"] = (
        risk_data["Predicted_Churn_Probability"].apply(assign_risk)
    )
    risk_data = risk_data.sort_values(
        "Predicted_Churn_Probability", ascending=False
    ).reset_index(drop=True)

    summary = (
        risk_data["Churn_Risk_Category"]
        .value_counts()
        .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Number of Customers")
    )
    summary["Percentage"] = (
        summary["Number of Customers"] / max(len(risk_data), 1) * 100
    ).round(2)

    c1, c2, c3 = st.columns(3)
    risk_counts = dict(zip(summary["Risk Category"], summary["Number of Customers"]))
    c1.metric("High-risk customers", f"{risk_counts.get('High Risk', 0):,}")
    c2.metric("Medium-risk customers", f"{risk_counts.get('Medium Risk', 0):,}")
    c3.metric("Low-risk customers", f"{risk_counts.get('Low Risk', 0):,}")

    st.write("Risk category summary")
    st.dataframe(summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x="Risk Category", y="Number of Customers", ax=ax)
    ax.set_title("Customer Distribution by Predicted Churn Risk")
    ax.set_xlabel("Risk category")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Highest predicted churn risk customers (test set)")
    st.dataframe(risk_data.head(50), use_container_width=True)

    st.download_button(
        "Download customer churn risk scores",
        data=make_csv_download(risk_data),
        file_name="Chapter_7_Customer_Churn_Risk_Scores.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download risk category summary",
        data=make_csv_download(summary),
        file_name="Chapter_7_Risk_Category_Summary.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Note: Model metrics depend on the uploaded dataset, preprocessing, random split, "
    "and settings. Risk thresholds (below 0.30, 0.30–0.59, and 0.60 or above) are "
    "illustrative project thresholds and should be validated before operational use."
)
