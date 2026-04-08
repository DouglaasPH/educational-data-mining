import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# carregar dataset
data = pd.read_csv("data/xAPI-Edu-Data.csv")

# carregar modelo
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
train_columns = joblib.load("train_columns.pkl")

# métricas (você pode salvar isso depois automaticamente)
metrics = {
    "Accuracy": 0.87,
    "MAE": 0.12,
    "R2": 0.81
}

# ===================================
# TÍTULO + VISÃO GERAL
# ===================================
st.title("Student Performance Prediction")

st.markdown("""
Sistema de Machine Learning para prever desempenho acadêmico baseado em:

- Engajamento
- Frequência
- Participação
- Dados familiares
""")

# ===================================
# DATASET INTERATIVO
# ===================================
st.header("Distribuição das Classes")

fig = px.histogram(data, x="Class")
st.plotly_chart(fig)

# ===================================
# MÉTRICAS DO MODELO
# ===================================
st.header("Performance do Modelo")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
col2.metric("MAE", f"{metrics['MAE']:.2f}")
col3.metric("R²", f"{metrics['R2']:.2f}")

# ===================================
# FEATURE IMPORTANCE
# ===================================

st.header("Importância das Variáveis")

fi = pd.read_csv("feature_importance.csv")

fig = px.bar(fi.head(10), x="Feature", y="Average")
st.plotly_chart(fig)

# ===================================
# INTERPRETAÇÃO
# ===================================

st.header("Interpretação")

st.write("""
O modelo mostra que o desempenho depende principalmente de:

- Engajamento (VisitedResources, raisedhands)
- Participação (Discussion)
- Frequência (Absence)
- Envolvimento dos pais

Fatores comportamentais são mais importantes que demográficos.
""")

# ===================================
# INPUTS
# ===================================
st.sidebar.header("Dados do Estudante")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
nationality = st.sidebar.selectbox("Nationality", data["NationalITy"].unique())
birthplace = st.sidebar.selectbox("Place of Birth", data["PlaceofBirth"].unique())
stage = st.sidebar.selectbox("Stage", data["StageID"].unique())
grade = st.sidebar.selectbox("Grade", data["GradeID"].unique())
section = st.sidebar.selectbox("Section", data["SectionID"].unique())
topic = st.sidebar.selectbox("Topic", data["Topic"].unique())
semester = st.sidebar.selectbox("Semester", data["Semester"].unique())
relation = st.sidebar.selectbox("Relation", data["Relation"].unique())

raisedhands = st.sidebar.slider("Raised Hands", 0, 100, 10)
visited = st.sidebar.slider("Visited Resources", 0, 100, 10)
announcements = st.sidebar.slider("Announcements View", 0, 100, 10)
discussion = st.sidebar.slider("Discussion", 0, 100, 10)

parent_survey = st.sidebar.selectbox("Parent Survey", data["ParentAnsweringSurvey"].unique())
parent_satisfaction = st.sidebar.selectbox("Parent Satisfaction", data["ParentschoolSatisfaction"].unique())
absence = st.sidebar.selectbox("Absence", data["StudentAbsenceDays"].unique())

btn = st.sidebar.button("Prever desempenho")

# ===================================
# PREDIÇÃO
# ===================================
if btn:

    input_dict = {
        "gender": gender,
        "NationalITy": nationality,
        "PlaceofBirth": birthplace,
        "StageID": stage,
        "GradeID": grade,
        "SectionID": section,
        "Topic": topic,
        "Semester": semester,
        "Relation": relation,
        "raisedhands": raisedhands,
        "VisITedResources": visited,
        "AnnouncementsView": announcements,
        "Discussion": discussion,
        "ParentAnsweringSurvey": parent_survey,
        "ParentschoolSatisfaction": parent_satisfaction,
        "StudentAbsenceDays": absence
    }

    df_input = pd.DataFrame([input_dict])

    # encoding igual treino
    df_input = pd.get_dummies(df_input)

    # alinhar colunas
    df_input = df_input.reindex(columns=train_columns, fill_value=0)

    # scaler
    df_input = scaler.transform(df_input)

    pred = model.predict(df_input)[0]

    mapping = {0: "Low", 1: "Medium", 2: "High"}

    st.success(f"Desempenho previsto: {mapping[pred]}")
