# 🎓 AI-Powered Financial Intelligence System

A comprehensive **Machine Learning & AI system** for educational institutions to forecast revenue, analyze marketing ROI, predict cash flow, and generate AI-powered financial reports.

---

## 🌟 Features

### 📊 **Advanced Analytics**
- **Marketing Mix Modeling (MMM)** with adstock & saturation effects
- **Time-series decomposition** (trend, seasonality, residuals)
- **Multi-model forecasting** (XGBoost, SARIMAX, Random Forest, Ensemble)

### 💰 **Financial Predictions**
- Revenue forecasting (R² = 0.99)
- Cash flow prediction (R² = 0.998)
- Liquidity risk classification (96.25% accuracy)

### 🤖 **AI-Powered Reports**
- Automated insight generation using **Llama 3.1** via Groq
- Professional PDF export with charts & recommendations
- Executive summaries and risk assessments

### 📈 **Interactive Dashboard**
- Real-time KPI monitoring
- Scenario simulation ("What-if" analysis)
- Marketing channel ROI attribution
- Beautiful visualizations with Plotly

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 DATA SOURCES                        │
│  synthetic_dataset.csv + synthetic_dataset1.csv     │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼────────┐  ┌────▼────────────────┐
│ code.ipynb     │  │ advanced_analysis.py│
│ (ML Pipeline)  │  │ (MMM Training)      │
└───────┬────────┘  └────┬────────────────┘
        │                │
        │  Generates     │  Generates
        │                │
┌───────▼────────────────▼──────────────┐
│       /models/ Directory              │
│  • xgboost_revenue_model.pkl          │
│  • rf_liquidity_model.pkl             │
│  • advanced_roi_model.pkl             │
│  • model_metadata.json                │
│  • + 10 more models & transformers    │
└───────┬───────────────────────────────┘
        │
        │  Loaded by
        │
┌───────▼──────────┐
│     app.py       │
│ (Streamlit UI)   │
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/thorOdinson16/financial-intelligence-system.git
cd financial-intelligence-system
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

5. **Run the dashboard**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
financial-intelligence-system/
├── models/                      # Trained ML models (15 files)
├── advanced_analysis.py         # Marketing Mix Model training
├── app.py                       # Streamlit dashboard
├── code.ipynb                   # ML pipeline notebook
├── dataset_generator.py         # Synthetic data generator
├── synthetic_dataset.csv        # Main dataset
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🎯 Usage Guide

### 1️⃣ **Train Models** (First Time Setup)

```bash
# Train advanced Marketing Mix Model
python advanced_analysis.py

# Train full ML pipeline (or open in Jupyter)
jupyter notebook code.ipynb
```

### 2️⃣ **Launch Dashboard**

```bash
streamlit run app.py
```

### 3️⃣ **Navigate Pages**

| Page | Purpose |
|------|---------|
| 📊 **Overview** | KPIs, trends, seasonal patterns |
| 📈 **Revenue Forecasting** | Compare 4 different models |
| 💵 **Cash Flow Analysis** | Predict liquidity needs |
| 🎯 **Marketing ROI** | True ROI per channel (adstock-adjusted) |
| 🔮 **Make Predictions** | Simulate future scenarios |
| 📄 **Generate Report** | AI-powered PDF with insights |

---

## 🤖 AI Report Generation

The system uses **Llama 3.1** (via Groq) to generate professional reports:

```python
# Example: Generate executive report
summary = generate_financial_summary(df)
report = generate_ai_report(summary)
pdf = generate_pdf_report(report, charts)
```

**Report includes:**
- Executive Summary
- Revenue Analysis
- Cash Flow Health
- Marketing Performance
- Risk Assessment
- Strategic Recommendations
- Forward-Looking Outlook

---

## 📊 Model Performance

| Model | Target | Metric | Score |
|-------|--------|--------|-------|
| XGBoost | Revenue | R² | 0.98 |
| Random Forest | Revenue | R² | 0.99 |
| SARIMAX | Revenue | R² | 0.91 |
| Stacked Ensemble | Revenue | R² | 0.91 |
| XGBoost | Cash Flow | R² | 0.998 |
| Random Forest | Liquidity Risk | Accuracy | 96.25% |
| Bayesian Ridge | ROI Attribution | MAE | 0.12 |

---

## 📈 Sample Data

The system includes synthetic datasets for demonstration:
- **synthetic_dataset.csv**: 365 days of financial transactions
- **synthetic_dataset1.csv**: Marketing spend & revenue data

**Features include:**
- Revenue, expenses, cash flow
- Marketing spend by channel
- Seasonal trends
- Payment terms
- Liquidity indicators

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
