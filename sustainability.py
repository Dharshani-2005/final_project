import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# SECTION 0: CREATE sustainability_data.csv AUTOMATICALLY
# -------------------------------------------------------------------------

def generate_csv_dataset():
    rows = 500
    np.random.seed(42)

    data = pd.DataFrame({
        "Carbon_Emissions": np.random.uniform(10, 500, rows),
        "Water_Usage": np.random.uniform(100, 10000, rows),
        "Energy_Consumption": np.random.uniform(50, 2000, rows),
        "Waste_Generation": np.random.uniform(5, 300, rows),
        "Employee_Wellness": np.random.uniform(1, 10, rows),
        "Region": np.random.choice(['North', 'South', 'East', 'West'], rows),
        "Sustainability_Index": np.random.uniform(20, 100, rows)
    })

    data.to_csv("sustainability_data.csv", index=False)
    print("✔ sustainability_data.csv generated successfully!")


# -------------------------------------------------------------------------
# SECTION 1: MACHINE LEARNING WORKFLOW
# -------------------------------------------------------------------------

def train_and_evaluate_model():
    print("\n--- Starting ML Workflow: Random Forest Regression ---")

    data = pd.read_csv("sustainability_data.csv")

    target_col = "Sustainability_Index"
    y = data[target_col]

    # Feature Engineering
    data["Efficiency_Ratio"] = data["Energy_Consumption"] / (data["Carbon_Emissions"] + 1e-6)
    X = data.drop(target_col, axis=1)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    poly_cols = ['Carbon_Emissions', 'Water_Usage']
    remaining_num_cols = [c for c in numeric_cols if c not in poly_cols]

    poly_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('poly', poly_pipeline, poly_cols),
        ('num', num_pipeline, remaining_num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # PCA
    n_components = min(10, X_processed.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 10],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        rf, param_grid, cv=3,
        scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"\nBest Model Hyperparameters: {grid.best_params_}")

    y_pred_test = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_pca, y, cv=cv, scoring='r2', n_jobs=-1)

    print(f"Test Set RMSE: {rmse:.4f}")
    print(f"Test Set R-squared: {r2_test:.4f}")
    print(f"CV Mean R-squared: {np.mean(cv_scores):.4f}")

    # Feature Importance Plot
    fi = best_model.feature_importances_
    plt.figure(figsize=(10, 6))
    pc_labels = [f"PC{i+1}" for i in range(len(fi))]
    sns.barplot(x=fi, y=pc_labels)
    plt.title("Feature Importance of Principal Components")
    plt.xlabel("Importance Score")
    plt.ylabel("Principal Component")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved as 'feature_importance.png'")

    # Residual Plot
    residuals = y_test - y_pred_test
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted")

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")

    plt.tight_layout()
    plt.savefig("model_diagnostics.png")
    print("Model diagnostics saved as 'model_diagnostics.png'")

    # Save Model
    joblib.dump({
        'preprocessor': preprocessor,
        'pca': pca,
        'model': best_model,
        'feature_cols': list(X.columns)
    }, "sustainability_model.joblib")

    print("✔ Model saved as 'sustainability_model.joblib'")
    print("--- ML Workflow Completed ---")


# -------------------------------------------------------------------------
# SECTION 2: SIMPLE MATPLOTLIB PLOT
# -------------------------------------------------------------------------

def plot_sine_wave():
    print("\n--- Starting Matplotlib Plotting Example ---")

    x = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="y = sin(x)", linewidth=2)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

if __name__ == "__main__":
    generate_csv_dataset()
    train_and_evaluate_model()
    plot_sine_wave()
