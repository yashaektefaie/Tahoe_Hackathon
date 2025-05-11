import anndata
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import pandas as pd

merged_anndata = anndata.read_h5ad("data/tahoe_vision_universal_embeddings.h5ad")

X = merged_anndata.obsm["X_delta"] # 60125 x 1280
Y = merged_anndata.X # 60125 x 7467
labels = merged_anndata.var.index.tolist() # 7467

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

if os.path.exists("models/linear_regression_model.pkl"):
    model = joblib.load("models/linear_regression_model.pkl")

    y_pred_test = model.predict(X_test)
    test_pearson = [pearsonr(y_test[:, i], y_pred_test[:, i])[0] for i in range(y_test.shape[1])]

    top_gene_set_indices = np.argsort(test_pearson)[-20:][::-1]

    top_gene_sets = [(test_pearson[i], labels[i]) for i in top_gene_set_indices]

    print("Top 20 gene sets with the highest correlation:")
    for correlation, gene_set in top_gene_sets:
        print(f"gene set {gene_set}: pearson correlation = {correlation:.4f}")

    plt.hist(test_pearson, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Pearson Correlation Coefficients (Test Set)")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.savefig("figures/pearson_correlation_distribution.png")

    top_20_indices_per_row = np.argsort(np.abs(y_test), axis=1)[:, -20:]

    correlations = []
    for i in range(y_test.shape[0]):
        actual_top_20 = y_test[i, top_20_indices_per_row[i]]
        predicted_top_20 = y_pred_test[i, top_20_indices_per_row[i]]
        correlation = pearsonr(actual_top_20, predicted_top_20)[0]
        correlations.append(correlation)

    average_correlation = np.mean(correlations)
    print(f"Average correlation for top 20 magnitude gene sets per row: {average_correlation:.4f}")
    
else:
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print(f"training MSE: {train_mse}")
    print(f"testing MSE: {test_mse}")

    joblib.dump(model, "models/linear_regression_model.pkl")

model = joblib.load("models/linear_regression_model.pkl")

disease_deltas = anndata.read_h5ad("data/disease_deltas.h5ad")
predicted_vision_signatures = model.predict(disease_deltas.X)

dataframe = pd.DataFrame(predicted_vision_signatures, columns = labels)

labels_combined = disease_deltas.obs.apply(
    lambda row: f"{row['cell_type']}_{row['tissue']}_{row['disease']}", axis=1
).tolist()

top_20_gene_sets = []

for index, row in dataframe.iterrows():
    top_20_indices = np.argsort(np.abs(row))[-20:][::-1]
    top_20 = [(labels[i], "down" if row.iloc[i] < 0 else "up") for i in top_20_indices]
    top_20_gene_sets.append(top_20)

with open("top_20_gene_sets.txt", "w") as f:
    for i, gene_set in enumerate(top_20_gene_sets):
        f.write(f"{labels_combined[i]}\t" + "\t".join([f"{gene}:{direction}" for gene, direction in gene_set]) + "\n")