from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import shap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def run_model(outcome, data):
    """
    Trains the model with the given outcome variable. Returns
    the trained model along with the data splits.
    Args:
        outcome (str): The outcome variable to be predicted
        data (str): The path to the data file

    Returns:
        (model, data): tuple of model and each data split
    """
    X = data.drop(
        ["status_acquired", "status_closed", "status_ipo", "status_operating"],
        axis=1,
    )
    y = data[f"status_{outcome}"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def shap_explain_model(model, X_train, X_test, y_train, y_test):
    """
    Explains the model using SHAP. Returns the SHAP values

    Args:
    model (sklearn.BaseEstimator): Trained machine learning model
    X_train (pd.DataFrame): Training feature matrix
    X_test (pd.DataFrame): Testing feature matrix
    y_train (pd.Series): Training labels
    y_test (pd.Series): Testing labels
    """
    # Step 4: Generate SHAP values for explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    return shap_values


def lime_explain_model(model, X_train, X_test, y_train, y_test):
    """
    Explains the model using LIME and returns the explanation for a test instance.

    Args:
        model (sklearn.BaseEstimator): Trained machine learning model
        X_train (pd.DataFrame): Training feature matrix
        X_test (pd.DataFrame): Testing feature matrix
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=list(X_train.columns),
        class_names=["Outcome"],
        discretize_continuous=True,
    )

    # Explain an instance (first row of X_test in this example)
    exp = explainer.explain_instance(
        X_test.values[0], model.predict_proba, num_features=10
    )

    return exp


if __name__ == "__main__":
    ## NOTE ###
    ### Use this section to play around with different values of
    ### outcome, and obtain the trained model and evaluations

    df = pd.read_csv("clean_data.csv")

    # choose an outcome of interest:
    outcome = "closed"
    trained_model, X_train, X_test, y_train, y_test = run_model(outcome, df)
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for outcome '{outcome}': {round(100*accuracy, 2)}%")

    print(
        f"Confusion Matrix for outcome '{outcome}': {confusion_matrix(y_test, y_pred)}"
    )
    # Explain the model
    exp = lime_explain_model(trained_model, X_train, X_test, y_train, y_test)

    shap_values = shap_explain_model(trained_model, X_train, X_test, y_train, y_test)

    pass
