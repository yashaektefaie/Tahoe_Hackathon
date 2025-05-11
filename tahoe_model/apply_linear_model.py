import anndata
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

merged_anndata = anndata.read_h5ad("data/tahoe_vision_universal_embeddings.h5ad")

X = merged_anndata.obsm["X_delta"] # 60125 x 1280
Y = merged_anndata.X # 60125 x 7467
labels = merged_anndata.var.index.tolist() # 7467

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

if os.path.exists("models/linear_regression_model.pkl"):
    print("model file already exists, loading the model")

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
    print("Figure saved under 'figures/pearson_correlation_distribution.png'")


    
else:
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_pearson = [pearsonr(y_train[:, i], y_pred_train[:, i])[0] for i in range(y_train.shape[1])]
    test_pearson = [pearsonr(y_test[:, i], y_pred_test[:, i])[0] for i in range(y_test.shape[1])]

    print(f"training MSE: {train_mse}")
    print(f"testing MSE: {test_mse}")
    print(f"average training pearson correlation: {np.mean(train_pearson)}")
    print(f"average testing pearson correlation: {np.mean(test_pearson)}")

    overall_correlation = pearsonr(y_pred_test.flatten(), y_test.flatten())[0]
    print(f"overall pearson correlation between y_pred_test and y_test: {overall_correlation}")

    joblib.dump(model, "models/linear_regression_model.pkl")