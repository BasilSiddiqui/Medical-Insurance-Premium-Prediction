import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Set style
sns.set(style='whitegrid')

# Load dataset
df = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\medinsurance\Medicalpremium.csv")  # Adjust path as needed

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# -----------------------------
# 🔥 Heatmap for Correlation
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("🔗 Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 📊 Distributions
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Age'], kde=True, ax=axs[0], color='skyblue')
axs[0].set_title("Age Distribution")

sns.histplot(df['Height'], kde=True, ax=axs[1], color='lightgreen')
axs[1].set_title("Height Distribution")

sns.histplot(df['Weight'], kde=True, ax=axs[2], color='salmon')
axs[2].set_title("Weight Distribution")

plt.tight_layout()
plt.show()

# -----------------------------
# 🧪 Train-Test Split
# -----------------------------
X = df.drop('PremiumPrice', axis=1)
y = df['PremiumPrice']

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Age', 'Height', 'Weight']] = scaler.fit_transform(X[['Age', 'Height', 'Weight']])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 🤖 Model Comparison
# -----------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Random Forest': RandomForestRegressor(),
    'XGBRFRegressor': XGBRFRegressor()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    })

results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(results_df.sort_values(by='R2 Score', ascending=False))

# -----------------------------
# 🎯 Feature Importance
# -----------------------------
def plot_feature_importance(model, model_name):
    importances = model.feature_importances_
    sns.barplot(x=importances, y=X.columns, palette='viridis')
    plt.title(f'{model_name} - Feature Importance')
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# Plot for Random Forest
plot_feature_importance(models['Random Forest'], 'Random Forest')

# Plot for XGBRF
plot_feature_importance(models['XGBRFRegressor'], 'XGBoost Regressor')


# 🎯 Make a prediction with custom input values

# Let's say we have a 35-year-old person, 170cm tall, 72kg, no diabetes, bloodpressure, transplants, chronic diseases, allergies or cancer history, but one major surgery.
# 🎯 Sample input with all required features
sample_input = pd.DataFrame([{
    'Age': 35,
    'Diabetes': 0,
    'BloodPressureProblems': 0,
    'AnyTransplants': 0,
    'AnyChronicDiseases': 0,
    'Height': 170,
    'Weight': 72,
    'KnownAllergies': 0,
    'HistoryOfCancerInFamily': 0,
    'NumberOfMajorSurgeries': 1
}])

# ✅ Scale numeric features
sample_input[['Age', 'Height', 'Weight']] = scaler.transform(sample_input[['Age', 'Height', 'Weight']])

# 🔮 Predict
predicted_premium = models['XGBRFRegressor'].predict(sample_input)[0]
print(f"💰 Predicted Insurance Premium: {predicted_premium:.2f}")