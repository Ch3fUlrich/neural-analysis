import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")

@app.cell
def _():
    import pprint

    import marimo as mo
    import numpy as np
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    from neural_analysis.decoding import decode
    return decode, make_classification, make_regression, mo, np, pprint, train_test_split

@app.cell
def _(decode, make_classification, mo, np, pprint, train_test_split):
    mo.md("## Decoding Classification Demo")

    # Generate mock embedding data
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classification_results = decode(
        embedding_train=X_train,
        embedding_test=X_test,
        labels_train=y_train,
        labels_test=y_test,
        test_outlier_removal=False,
        include_cv_stats=True,
        detailed_metrics=True
    )

    mo.md(f"Classification Results:\n\n```python\n{pprint.pformat(classification_results)}\n```")
    return X, X_test, X_train, classification_results, y, y_test, y_train

@app.cell
def _(decode, make_regression, mo, np, pprint, train_test_split):
    mo.md("## Decoding Regression Demo")

    # Generate mock continuous data
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, n_informative=4, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    regression_results = decode(
        embedding_train=X_train_reg,
        embedding_test=X_test_reg,
        labels_train=y_train_reg,
        labels_test=y_test_reg,
        test_outlier_removal=False,
        include_cv_stats=True
    )

    mo.md(f"Regression Results:\n\n```python\n{pprint.pformat(regression_results)}\n```")
    return (
        X_reg,
        X_test_reg,
        X_train_reg,
        regression_results,
        y_reg,
        y_test_reg,
        y_train_reg,
    )

if __name__ == "__main__":
    app.run()
