import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# --- Step 1: Load the Full Dataset ---
print("Loading the dataset 'synthetic_dataset1.csv'...")
try:
    df = pd.read_csv('synthetic_dataset1.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'synthetic_dataset1.csv' not found.")
    print("Please make sure the script and the CSV file are in the same folder.")
    exit()

# --- Step 2: Clean and Prepare the Data ---
print("\nCleaning and preparing the data...")
# Clean column names for easier access
df.columns = df.columns.str.replace(r'[()]', '', regex=True).str.replace(' ', '_').str.lower()

# Aggregate data by date by summing up the values for each day
df['date'] = pd.to_datetime(df['date'])
df_agg = df.groupby('date').sum()
print(f"Aggregated {len(df)} rows into {len(df_agg)} unique daily records.")


# --- Step 3: Decompose Revenue into Core Components ---
print("\nPerforming Time-Series Decomposition...")
# Assuming yearly seasonality with a 365-day period
# We'll use a 90-day period to find quarterly business cycles
decomposition = seasonal_decompose(df_agg['revenue'], model='additive', period=90)

# Create and save a plot of the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=ax1, legend=False, title="Observed Daily Revenue")
ax1.set_ylabel('Revenue')
decomposition.trend.plot(ax=ax2, legend=False, title="Underlying Trend")
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3, legend=False, title="Seasonal Pattern")
ax3.set_ylabel('Seasonality')
decomposition.resid.plot(ax=ax4, legend=False, title="Residuals (Noise)")
ax4.set_ylabel('Residuals')
plt.suptitle('Revenue Time-Series Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('revenue_decomposition_advanced.png')
print("Decomposition plot saved as 'revenue_decomposition_advanced.png'.")


# --- Step 4: Build a More Realistic Marketing Mix Model ---
print("\nBuilding Advanced Marketing Mix Model...")

# Function to model the lingering effect of ads (Adstock)
def apply_adstock(series, decay_rate):
    adstocked_series = np.zeros_like(series, dtype=float)
    adstocked_series[0] = series.iloc[0]
    for i in range(1, len(series)):
        adstocked_series[i] = series.iloc[i] + decay_rate * adstocked_series[i-1]
    return adstocked_series

# Apply Adstock to each marketing channel
df_agg['google_adstock'] = apply_adstock(df_agg['marketing_spend_google'], decay_rate=0.5)
df_agg['linkedin_adstock'] = apply_adstock(df_agg['marketing_spend_linkedin'], decay_rate=0.3)
df_agg['campus_adstock'] = apply_adstock(df_agg['marketing_spend_campus'], decay_rate=0.2)
df_agg['events_adstock'] = apply_adstock(df_agg['marketing_spend_events'], decay_rate=0.1)
print("Adstock (carryover effect) applied.")

# Function to model diminishing returns (Saturation)
def apply_saturation(series, power):
    return series ** power

# Apply Saturation to the adstocked channels
df_agg['google_saturated'] = apply_saturation(df_agg['google_adstock'], power=0.7)
df_agg['linkedin_saturated'] = apply_saturation(df_agg['linkedin_adstock'], power=0.8)
df_agg['campus_saturated'] = apply_saturation(df_agg['campus_adstock'], power=0.85)
df_agg['events_saturated'] = apply_saturation(df_agg['events_adstock'], power=0.9)
print("Saturation (diminishing returns) effect applied.")


# --- Step 5: Train the Advanced Model ---
print("\nTraining the Advanced Bayesian Ridge Model...")
# Use the trend as a baseline feature
df_agg['trend'] = decomposition.trend
df_agg = df_agg.dropna()

# Define the features and the target for the model
features = ['trend', 'google_saturated', 'linkedin_saturated', 'campus_saturated', 'events_saturated']
X_advanced = df_agg[features]
y_advanced = df_agg['revenue']

# Train the model
advanced_roi_model = BayesianRidge()
advanced_roi_model.fit(X_advanced, y_advanced)
print("Advanced model trained successfully.")


# --- Step 6: Save the Final Model ---
with open('advanced_roi_model.pkl', 'wb') as f:
    pickle.dump(advanced_roi_model, f)
print("Advanced model saved as 'advanced_roi_model.pkl'.")


# --- Step 7: Display Key Insights ---
print("\n--- Advanced Model Insights ---")
print("Learned Coefficients (Feature Contributions):")
coefs = pd.Series(advanced_roi_model.coef_, index=X_advanced.columns)
print(coefs)