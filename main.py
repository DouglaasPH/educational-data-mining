# =================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# =================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

from sklearn.preprocessing import StandardScaler

# =================================================================
# 2. CARREGAMENTO DO DATASET (LOCAL)
# =================================================================

try:
    df = pd.read_csv('data/xAPI-Edu-Data.csv')  # arquivo na mesma pasta
    print("Dataset carregado com sucesso!")
except Exception as e:
    print("Erro ao carregar o dataset:", e)
    exit()

# =================================================================
# 3. PRÉ-PROCESSAMENTO
# =================================================================

df_processed = df.copy()

# Target
target_map = {'L': 0, 'M': 1, 'H': 2}
df_processed['Class'] = df_processed['Class'].map(target_map)

# Gênero
df_processed['gender'] = df_processed['gender'].map({'F': 0, 'M': 1})

# Label Encoding
le = LabelEncoder()
categorical_cols = df_processed.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df_processed[col] = le.fit_transform(df_processed[col])

# Separação
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

# =================================================================
# 4. IMPORTÂNCIA DAS FEATURES
# =================================================================

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importance = rf.feature_importances_

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X / X.max(), y)
lr_importance = np.abs(lr.coef_).mean(axis=0)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'RF': rf_importance / rf_importance.max(),
    'LR': lr_importance / lr_importance.max()
})

feature_importance_df['Average'] = feature_importance_df[['RF', 'LR']].mean(axis=1)
feature_importance_df = feature_importance_df.sort_values(by='Average', ascending=False)

plt.figure(figsize=(10, 6))
feature_importance_df.head(10).plot(
    x='Feature',
    y=['RF', 'LR', 'Average'],
    kind='bar'
)
plt.title('Importância das Características')
plt.ylabel('Pontuação')
plt.tight_layout()
plt.show()

feature_importance_df.to_csv("feature_importance.csv", index=False)

# =================================================================
# 5. BALANCEAMENTO
# =================================================================

print(f"Distribuição Original: {Counter(y)}")

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"Após SMOTE: {Counter(y_smote)}")

# =================================================================
# 6. VISUALIZAÇÃO
# =================================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=y, ax=ax[0])
ax[0].set_title('Original')

sns.countplot(x=y_smote, ax=ax[1])
ax[1].set_title('SMOTE')

plt.tight_layout()
plt.show()

# =================================================================
# 7. MODELOS
# =================================================================

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    mean_absolute_error, r2_score
)

X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB()
}

results = {}

results = {}

print("\nTreinando modelos...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "Acurácia": acc,
        "MAE": mae,
        "R2": r2
    }

    print(f"{name}:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}\n")

# =================================================================
# 8. HPO (Random Forest)
# =================================================================

param_grids = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=10000),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    }
}

print("\n===== HYPERPARAMETER TUNING =====\n")

best_models = {}

for name, config in param_grids.items():
    print(f"\n🔍 Tunando {name}...")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name + " Tunado"] = {
        "Acurácia": acc,
        "MAE": mae,
        "R2": r2
    }

    best_models[name] = best_model

    print(f"Melhores parâmetros: {grid.best_params_}")
    print(f"Acurácia: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

# =================================================================
# 9. GRÁFICO FINAL
# =================================================================

res_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
res_df.rename(columns={'index': 'Modelo'}, inplace=True)
res_df = res_df.sort_values(by='Acurácia', ascending=False)

res_df_melt = res_df.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

plt.figure(figsize=(10, 6))
sns.barplot(data=res_df_melt, x='Valor', y='Modelo', hue='Métrica')
plt.title("Comparação de Modelos (Acurácia, MAE, R2)")
plt.show()

# =================================================================
# 10. MATRIZ DE CONFUSÃO
# =================================================================

best_model_name = res_df.iloc[0]['Modelo']
print(f"\nMelhor modelo: {best_model_name}")

best_model_key = best_model_name.replace(" Tunado", "")
best_model = best_models.get(best_model_key, None)

if best_model is not None:
    y_pred_final = best_model.predict(X_test)
else:
    # fallback caso seja modelo não tunado
    model = models[best_model_name]
    model.fit(X_train, y_train)
    y_pred_final = model.predict(X_test)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d')
plt.title("Matriz de Confusão (Melhor Modelo)")
plt.show()

# =================================================================
# 11. SALVAR MODELO E SCALER
# =================================================================

# salvar melhor modelo
joblib.dump(best_model, "model.pkl")

# salvar scaler
joblib.dump(scaler, "scaler.pkl")

# salvar colunas de treino
joblib.dump(X.columns.tolist(), "train_columns.pkl")

# salvar label encoder
joblib.dump(le, "label_encoder.pkl")

print("Modelo salvo com sucesso!")

# =================================================================
# 12. EXPORTAR RELATÓRIO
# =================================================================

df_resultados = res_df

report_dict = classification_report(y_test, y_pred_final, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_base = res_df[~res_df['Modelo'].str.contains("Tunado")]
df_tunado = res_df[res_df['Modelo'].str.contains("Tunado")]

html_content = f"""
<html>
<head>
<title>Relatório</title>
</head>

<body>

<h1>Resultados</h1>

<div class="container">

<div class="table">
<h2>Modelos Base</h2>
{df_base.to_html(index=False)}
</div>

<div class="table">
<h2>Modelos Tunados</h2>
{df_tunado.to_html(index=False)}
</div>

</div>

<h2>Classificação (Melhor Modelo)</h2>
{df_report.to_html()}

<h2>Melhor Modelo</h2>
<pre>{best_model_name}</pre>

</body>
</html>
"""

with open("relatorio_modelos.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Relatório salvo: relatorio_modelos.html")

# =================================================================
# 13. RELATÓRIO TXT
# =================================================================

with open("saida_experimento.txt", "w") as f:
    f.write("RELATORIO\n")
    f.write(str(grid.best_params_) + "\n\n")
    f.write(classification_report(y_test, y_pred_final))

print("Arquivo salvo: saida_experimento.txt")