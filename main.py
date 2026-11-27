# ==========================================================
# TP N°4 : Régression linéaire simple et multiple
# Diabetes Dataset (Scikit-learn)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------
# 1. Chargement des données
# ----------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame            # features + colonne 'target'
X = data.data              # variables explicatives (10 colonnes)
y = data.target            # variable cible (progression de la maladie)

print("=" * 60)
print("1. CHARGEMENT ET EXPLORATION")
print("=" * 60)
print(f"Dimensions : {df.shape[0]} patients, {X.shape[1]} variables explicatives")
print(f"Variables explicatives : {list(X.columns)}")
print(f"Variable cible : target\n")

print("Premières lignes du dataset :")
print(df.head(), "\n")


# ----------------------------------------------------------
# 2. Statistiques descriptives
# ----------------------------------------------------------
print("=" * 60)
print("2. STATISTIQUES DESCRIPTIVES")
print("=" * 60)
stats = df.describe().T[['mean', '50%', 'std', 'min', 'max']]
stats.columns = ['moyenne', 'médiane', 'écart-type', 'min', 'max']
print(stats.round(3), "\n")


# ----------------------------------------------------------
# 3. Valeurs manquantes et doublons
# ----------------------------------------------------------
print("=" * 60)
print("3. VALEURS MANQUANTES ET DOUBLONS")
print("=" * 60)
print("Valeurs manquantes :", df.isna().sum().sum())
print("Doublons :", df.duplicated().sum(), "\n")


# ----------------------------------------------------------
# 4. Analyse des corrélations
# ----------------------------------------------------------
print("=" * 60)
print("4. CORRÉLATIONS")
print("=" * 60)
print("Corrélation avec la cible :")
print(df.corr()['target'].sort_values(ascending=False).round(3), "\n")

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# 5. Relations variables explicatives / cible
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, col in zip(axes.flat, X.columns):
    ax.scatter(df[col], y, alpha=0.5, s=12)
    ax.set_xlabel(col)
    ax.set_ylabel('target')
    ax.set_title(f"{col} vs target")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# 6. Séparation apprentissage / test
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("=" * 60)
print("6. SÉPARATION DES DONNÉES")
print("=" * 60)
print(f"Apprentissage : {X_train.shape[0]} lignes")
print(f"Test          : {X_test.shape[0]} lignes\n")


# ----------------------------------------------------------
# 7. Régression linéaire simple (BMI → target)
# ----------------------------------------------------------
print("=" * 60)
print("7. RÉGRESSION LINÉAIRE SIMPLE")
print("=" * 60)

X_train_s = X_train[['bmi']]
X_test_s = X_test[['bmi']]

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train)

a = model_simple.coef_[0]        # pente (slope)
b = model_simple.intercept_      # constante (intercept)
print(f"Équation : Y = {a:.2f} * BMI + {b:.2f}\n")

y_pred_simple = model_simple.predict(X_test_s)

# Graphique : données et droite de régression
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test, color='steelblue', alpha=0.6, label='Données de test')
xs = np.linspace(X['bmi'].min(), X['bmi'].max(), 100).reshape(-1, 1)
plt.plot(xs, model_simple.predict(xs), color='red', linewidth=2,
         label='Droite de régression')
plt.xlabel('BMI (standardisé)')
plt.ylabel('Progression (target)')
plt.title('Régression linéaire simple')
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# 8. Régression linéaire multiple (10 variables)
# ----------------------------------------------------------
print("=" * 60)
print("8. RÉGRESSION LINÉAIRE MULTIPLE")
print("=" * 60)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

b0 = model_multi.intercept_
coefs = pd.Series(model_multi.coef_, index=X.columns)
print(f"b0 (constante) = {b0:.2f}")
print("Coefficients triés par influence absolue :")
print(coefs.sort_values(key=abs, ascending=False).round(3), "\n")

y_pred_multi = model_multi.predict(X_test)

# Influence des variables (valeurs absolues des coefficients)
plt.figure(figsize=(7, 5))
coefs.abs().sort_values().plot.barh(title="Influence des variables (|coefficient|)")
plt.xlabel("Valeur absolue du coefficient")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# 9. Évaluation et comparaison des deux modèles
# ----------------------------------------------------------
print("=" * 60)
print("9. ÉVALUATION DES PERFORMANCES")
print("=" * 60)


def evaluer(y_true, y_pred, n_features, nom):
    """Calcule les métriques de régression pour un modèle donné."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Modèle': nom,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R²': r2,
        'R² ajusté': 1 - (1 - r2) * (n - 1) / (n - n_features - 1),
    }


resultats = pd.DataFrame([
    evaluer(y_test, y_pred_simple, 1, 'Simple (BMI)'),
    evaluer(y_test, y_pred_multi, X.shape[1], 'Multiple (10 variables)'),
])
print(resultats.round(3).to_string(index=False))

# Graphique comparatif : valeurs réelles vs prédites
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, pred, titre in zip(
    axes,
    [y_pred_simple, y_pred_multi],
    ['Simple (BMI)', 'Multiple (10 variables)']
):
    ax.scatter(y_test, pred, alpha=0.6, color='steelblue')
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, 'r--', label='Prédiction parfaite')
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Valeurs prédites')
    ax.set_title(titre)
    ax.legend()
plt.tight_layout()
plt.show()

print("\nFin du TP.")