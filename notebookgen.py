# Re-run the code to generate the notebook after kernel reset

import pandas as pd
import numpy as np
import json

# Create notebook content
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧮 Financials-Based Risk Scorecard Model\n",
                "This notebook builds a credit risk scorecard using financial ratios to estimate Probability of Default (PD).\n",
                "We'll use logistic regression and validate the model with AUC, confusion matrix, and classification report."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Import libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Load the dataset\n",
                "df = pd.read_csv('Risk_Scorecard_Financials.csv')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Define features and target\n",
                "features = ['Debt_to_EBITDA', 'Interest_Coverage', 'Current_Ratio', 'Net_Profit_Margin']\n",
                "X = df[features]\n",
                "y = df['Default_Flag']"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Train-test split\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Fit logistic regression model\n",
                "model = LogisticRegression()\n",
                "model.fit(X_train, y_train)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Predict probabilities and classes\n",
                "y_pred_proba = model.predict_proba(X_test)[:, 1]\n",
                "y_pred_class = model.predict(X_test)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 7: Model evaluation\n",
                "auc = roc_auc_score(y_test, y_pred_proba)\n",
                "cm = confusion_matrix(y_test, y_pred_class)\n",
                "report = classification_report(y_test, y_pred_class)\n",
                "print(f\"AUC Score: {auc:.3f}\")\n",
                "print(\"Confusion Matrix:\\n\", cm)\n",
                "print(\"Classification Report:\\n\", report)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 8: ROC Curve\n",
                "fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
                "plt.figure(figsize=(8,6))\n",
                "plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')\n",
                "plt.plot([0,1], [0,1], linestyle='--', color='gray')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.title('ROC Curve')\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Save notebook file
notebook_path = "Risk_Scorecard_Modeling_Notebook.ipynb"
with open(notebook_path, "w") as f:
    json.dump(notebook_content, f)

notebook_path
