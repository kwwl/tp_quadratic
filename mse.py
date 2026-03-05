import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))


def backpropagation_quadratic(a, b, c, x, y, epsilon, n):
    erreurs = quadratic_regression(a, b, c, x) - y
    dL_da = (2 / n) * (erreurs * x**2).sum()
    dL_db = (2 / n) * (erreurs * x).sum()
    dL_dc = (2 / n) * (erreurs).sum()
    a = a - epsilon * dL_da
    b = b - epsilon * dL_db
    c = c - epsilon * dL_dc
    current_rmse = rmse(quadratic_regression(a, b, c, x), y)
    return a, b, c, current_rmse


def gradient_descent_quadratic(x, y, epsilon, n_iterations):
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    rmse_history = []
    n = len(x)
    for _ in range(n_iterations):
        a, b, c, current_rmse = backpropagation_quadratic(a, b, c, x, y, epsilon, n)
        rmse_history.append(current_rmse)
    return a, b, c, rmse_history


if __name__ == "__main__":
    # A1 Chargement et affichage
    house_prices_df = pd.read_csv(
        "/Users/kemillamouri/Desktop/Data IA HETIC 2/tp_quadratic/prix_maisons.csv"
    )
    print(house_prices_df.head())
    print(house_prices_df.dtypes)
    print(len(house_prices_df))

    # A1 Normalisation
    x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
    y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()
    house_prices_df["surface"] = (house_prices_df["surface"] - x_mean) / x_std
    house_prices_df["prix"] = (house_prices_df["prix"] - y_mean) / y_std

    # A2 Visualisation
    plt.scatter(house_prices_df["surface"], house_prices_df["prix"], alpha=0.5)
    plt.title("Visualitation des points")
    plt.xlabel("Surface")
    plt.ylabel("Prix")
    plt.show()

    # Paramètres
    epsilon = 1e-4
    n = len(house_prices_df)
    n_iterations = 50000
    x = house_prices_df["surface"].values
    y = house_prices_df["prix"].values

    # Modèle linéaire
    a = np.random.randn()
    b = np.random.randn()
    for _ in range(n_iterations):
        erreurs = erreurs = a * house_prices_df["surface"] + b - house_prices_df["prix"]
        a = a - epsilon * (2 / n) * (erreurs * house_prices_df["surface"]).sum()
        b = b - epsilon * (2 / n) * erreurs.sum()

    # E1 Entraînement quadratique
    a_quadratic, b_quadratic, c_quadratic, rmse_history = gradient_descent_quadratic(
        x, y, epsilon, n_iterations
    )

    # Figure 1  Nuage de points + courbes
    x_line = np.linspace(
        house_prices_df["surface"].min(), house_prices_df["surface"].max(), 100
    )
    y_line_lin = a * x_line + b
    y_line_quadratic = quadratic_regression(
        a_quadratic, b_quadratic, c_quadratic, x_line
    )

    plt.scatter(x, y, alpha=0.5, label="Données")
    plt.plot(x_line, y_line_lin, color="blue", label="Linéaire")
    plt.plot(x_line, y_line_quadratic, color="red", label="Quadratique")
    plt.title("Comparaison mse et modèle quadratique")
    plt.xlabel("Surface ")
    plt.ylabel("Prix")
    plt.legend()
    plt.savefig("figure1_predictions.png")
    plt.show()

    # Figure 2  RMSE en fonction des epochs
    plt.plot(rmse_history, color="red")
    plt.title("RMSE en fonction des epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.savefig("figure2_rmse.png")
    plt.show()

"""
    A1 — Pourquoi la standardisation aide la descente de gradient ?
    elle évite des gradients trop grands qui feraient rater le minimum
    — Que se passe-t-il si on ne normalise pas et que les surfaces sont en dizaines alors que les prix
    sont en centaines de milliers ?
    la descente de gradient serait inefficace car il y aura un trop
    grand ecart entre les gradients des prix et des surfaces
"""

"""
    A2 — la relation semble-t-elle parfaitement linéaire ? Que pourrait apporter un modèle
    quadratique ?
    oui la relation semble être linéaire et un modèle quadratique n’apporterait 
    pas grand chose à première vue car les données ne semblent pas suivre une légère courbe
"""
"""
    B1 — Quelles sont les trois poids du modèle ?
    a,b,c
    — En quoi ce modèle est-il plus flexible qu’un modèle affine ?
    plus de parametres donc possibilité d'avoir des relations courbes et pas seulement des droites
"""
"""
    B2 - Quelle différence entre MSE et RMSE ?
    La MSE est en unités au carré et la RMSE est en unités normales : RMSE = np.sqrt(MSE).
    - Pourquoi la RMSE est parfois plus lisible ?
    meilleure interpretation des données car elle est dans les meme unité que les données 
"""
"""
    C1 - Quelle est la dérivée de (^y-y)**2 par rapport à ^y?
    2(^y-y)
    - Quelle est la dérivée de ^y = ax2 + bx + c par rapport à a, b, c ?
    x2, x, 1
"""
"""
   D1 — À quoi sert le learning rate η ?
   c'est le pas de progression entre chaque epochs
    — Donner deux symptômes d’un learning rate trop grand.
    rmse en dent de scie ou diverge vers l'infini
    — Donner un symptôme d’un learning rate trop petit.
    rmse trop lente donc il faut beaucoup trop d'epochs pour atteindre le minimum 
"""
"""
    E1 — La RMSE doit-elle être strictement décroissante à chaque epoch ?
    Non pas strictement mais globalement décroissante
    — Quand peut-on arrêter l’entraînement (early stopping) ?
    quand la courbe de la rmse stagne 
"""
"""
    F1 - Le modèle quadratique fait-il toujours mieux ?
    non, au vu des données qui ne semblent pas suivre une légère courbe comme vu à la question A2
"""
