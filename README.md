# Educational Data Mining

Sistema de Machine Learning para **previsão de desempenho de estudantes** com base em dados acadêmicos, comportamentais e familiares.

**Deploy:** https://educational-data-mining.streamlit.app/

---

## Visão Geral

Este projeto utiliza técnicas de **Data Mining e Machine Learning** para classificar estudantes em três níveis de desempenho:

- 🟥 Low
- 🟨 Medium
- 🟩 High

A previsão é baseada em fatores como:

- Engajamento com a plataforma
- Participação em aula
- Frequência escolar
- Interação com materiais
- Envolvimento dos pais

---

## Tecnologias Utilizadas

- Python
- Pandas / NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn / Plotly
- Streamlit

---

## Pipeline do Projeto

O projeto segue um pipeline completo de Machine Learning:

1. **Pré-processamento**
   - Encoding de variáveis categóricas
   - Normalização dos dados

2. **Balanceamento**
   - Random UnderSampling
   - Random OverSampling
   - SMOTE

3. **Treinamento de Modelos**
   - Decision Tree
   - Random Forest
   - Logistic Regression
   - KNN
   - SVM
   - Naive Bayes

4. **Hyperparameter Tuning**
   - GridSearchCV aplicado a todos os modelos

5. **Avaliação**
   - Accuracy
   - MAE
   - R²

6. **Deploy**
   - Interface interativa com Streamlit

---

## Resultados

O modelo que apresentou um melhor desempenho foi o Random Forest após o hyperparameter tuning. Ele apresentou:

- **Accuracy:** ~0.87
- **MAE:** ~0.12
- **R²:** ~0.81

---

## Principais Insights

As variáveis mais relevantes para a previsão foram:

1. VisitedResources
2. raisedhands
3. AnnouncementsView
4. StudentAbsenceDays
5. Discussion

### Interpretação

O desempenho do aluno está fortemente relacionado a:

- Engajamento com a plataforma
- Participação ativa
- Frequência escolar
- Envolvimento dos pais

Fatores comportamentais tiveram maior impacto que fatores demográficos.

---

## Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/educational-data-mining.git
cd educational-data-mining
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
```

Ativar:

**Windows**

```bash
venv/Scripts/activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar o app

```bash
streamlit run app.py
```

## Estrutura do Projeto

```
educational-data-mining/
│
├── .gitignore
├── main.py
├── app.py
├── label_encoder.pkl
├── model.pkl
├── scaler.pkl
├── feature_importance.csv
├── relatorio_modelos.html
├── saida_experimento.txt
├── train_columns.pkl
├── data/
│   └── xAPI-Edu-Data.csv
├── requirements.txt
└── README.md
```

## Deploy

O sistema permite:

- Visualizar o dataset
- Analisar distribuição das classes
- Ver importância das variáveis
- Inserir dados de um estudante
- Obter previsão em tempo real

## Autores

Projeto acadêmico desenvolvido por Douglas Phelipe, Ivaldo Dantas e Jaldson Arthur
