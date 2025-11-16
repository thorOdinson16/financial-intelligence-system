import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Date range (2 years starting from Oct 1, 2022)
start_date = datetime(2022, 10, 1)
dates = [start_date + timedelta(days=x) for x in range(730)]
date_samples = np.random.choice(dates, 2000)

# Generate synthetic data
data = {
    'Date': date_samples,
    'Revenue': np.random.uniform(5000, 20000, 2000).round(2),
    'Marketing Spend (Google)': np.random.uniform(1000, 5000, 2000).round(2),
    'Marketing Spend (LinkedIn)': np.random.uniform(800, 4000, 2000).round(2),
    'Marketing Spend (Campus)': np.random.uniform(500, 3000, 2000).round(2),
    'Marketing Spend (Events)': np.random.uniform(600, 3500, 2000).round(2),
    'Operational Costs': np.random.uniform(3000, 8000, 2000).round(2),
    'Number of Enrollments': np.random.poisson(50, 2000),
    'Retention Rates': np.random.uniform(0.7, 0.95, 2000).round(2),
    'Seasonal Trends': np.random.choice(['High', 'Medium', 'Low'], 2000),
    'Conversions (Google)': np.random.poisson(10, 2000),
    'Conversions (LinkedIn)': np.random.poisson(8, 2000),
    'Conversions (Campus)': np.random.poisson(5, 2000),
    'Conversions (Events)': np.random.poisson(6, 2000),
    'Leads Generated (Google)': np.random.poisson(20, 2000),
    'Leads Generated (LinkedIn)': np.random.poisson(15, 2000),
    'Leads Generated (Campus)': np.random.poisson(10, 2000),
    'Leads Generated (Events)': np.random.poisson(12, 2000),
    'Customer Acquisition Cost (Google)': np.random.uniform(50, 200, 2000).round(2),
    'Customer Acquisition Cost (LinkedIn)': np.random.uniform(60, 220, 2000).round(2),
    'Customer Acquisition Cost (Campus)': np.random.uniform(70, 250, 2000).round(2),
    'Customer Acquisition Cost (Events)': np.random.uniform(65, 230, 2000).round(2),
    'Payment Terms': np.random.choice(['30 days', '60 days', '90 days'], 2000),
    'Fixed Costs': np.random.uniform(2000, 5000, 2000).round(2),
    'Variable Costs': np.random.uniform(1000, 3000, 2000).round(2),
    'Liabilities': np.random.uniform(1000, 5000, 2000).round(2)
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Date
df = df.sort_values('Date')

# Save to CSV
df.to_csv('synthetic_dataset.csv', index=False)

print("Synthetic dataset with 2000 records has been generated and saved as 'synthetic_dataset.csv'.")