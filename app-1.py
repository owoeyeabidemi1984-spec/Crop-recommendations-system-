from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

DATA_PATH = Path(__file__).parent / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling only matters for SVM; tree models don't need it.
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0))]
        ),
    }

    results, reports, cv_scores = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
        reports[name] = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).T
        cv_scores[name] = cross_val_score(model, X, y, cv=5).mean()

    best_name = max(results, key=results.get)
    return models, results, cv_scores, reports, best_name, X_test, y_test


# ---------- Load ----------
st.title("🌾 Crop Recommendation System")
st.caption("Enter soil and weather conditions to get the most suitable crop.")

if not DATA_PATH.exists():
    st.error(
        "`Crop_recommendation.csv` not found. Place it in the same folder as `app.py`."
    )
    st.stop()

df = load_data()
missing = [c for c in FEATURES + ["label"] if c not in df.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

with st.spinner("Training models..."):
    models, results, cv_scores, reports, best_name, X_test, y_test = train_models(df)
best_model = models[best_name]

tab_predict, tab_data, tab_models = st.tabs(
    ["🔮 Predict", "📊 Data Explorer", "🤖 Model Performance"]
)

# ---------- Predict ----------
with tab_predict:
    st.subheader("Field measurements")
    desc = df[FEATURES].describe()

    def num(label, col, step, col_ui):
        lo, hi = float(desc.loc["min", col]), float(desc.loc["max", col])
        mid = float(desc.loc["50%", col])
        return col_ui.slider(label, lo, hi, mid, step=step)

    c1, c2, c3 = st.columns(3)
    N = num("Nitrogen (N)", "N", 1.0, c1)
    P = num("Phosphorous (P)", "P", 1.0, c2)
    K = num("Potassium (K)", "K", 1.0, c3)
    temperature = num("Temperature (°C)", "temperature", 0.1, c1)
    humidity = num("Humidity (%)", "humidity", 0.1, c2)
    ph = num("Soil pH", "ph", 0.1, c3)
    rainfall = num("Rainfall (mm)", "rainfall", 1.0, c1)

    if st.button("Recommend crop", type="primary"):
        row = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES
        )
        prediction = best_model.predict(row)[0]
        st.success(f"Recommended crop: **{str(prediction).upper()}**")

        if hasattr(best_model, "predict_proba"):
            proba = pd.Series(best_model.predict_proba(row)[0], index=best_model.classes_)
            top3 = proba.sort_values(ascending=False).head(3)
            st.write("Top 3 candidates:")
            st.bar_chart(top3)
        st.caption(f"Model used: {best_name}")

# ---------- Data ----------
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]} | Missing values: {int(df.isna().sum().sum())}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Class balance")
        st.bar_chart(df["label"].value_counts())
    with col_b:
        feature = st.selectbox("Feature by crop", FEATURES, index=6)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="label", y=feature, data=df, ax=ax)
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlabel("Crop")
        fig.tight_layout()
        st.pyplot(fig)

    st.write("Feature correlation")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ---------- Models ----------
with tab_models:
    st.subheader("Model comparison")
    comp = pd.DataFrame(
        {
            "Test accuracy": {k: f"{v * 100:.2f}%" for k, v in results.items()},
            "5-fold CV accuracy": {k: f"{v * 100:.2f}%" for k, v in cv_scores.items()},
        }
    )
    st.table(comp)
    st.info(f"Best model on the test set: **{best_name}**")

    chosen = st.selectbox("Classification report", list(models), index=list(models).index(best_name))
    st.dataframe(reports[chosen].round(3), use_container_width=True)

    st.write(f"Confusion matrix: {best_name}")
    cm = confusion_matrix(y_test, best_model.predict(X_test), labels=best_model.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=best_model.classes_, yticklabels=best_model.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")
    fig.tight_layout()
    st.pyplot(fig)
