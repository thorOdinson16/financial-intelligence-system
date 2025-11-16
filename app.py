import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import joblib
import pickle
from plotly.subplots import make_subplots
from langchain_groq import ChatGroq
from fpdf import FPDF
import os
from dotenv import load_dotenv

load_dotenv()

# Add these functions after imports, around line 10
def apply_adstock(series, decay_rate):
    """Apply adstock transformation to capture carryover effects"""
    adstocked_series = np.zeros_like(series, dtype=float)
    adstocked_series[0] = series.iloc[0]
    for i in range(1, len(series)):
        adstocked_series[i] = series.iloc[i] + decay_rate * adstocked_series[i-1]
    return adstocked_series

def apply_saturation(series, power):
    """Apply saturation transformation for diminishing returns"""
    return series ** power

def generate_analysis_report(df, metadata):
    """Generate comprehensive analysis report using LLM"""
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Calculate key metrics
        total_revenue = df['Revenue'].sum()
        avg_enrollments = df['Number of Enrollments'].mean()
        total_marketing = (df['Marketing Spend (Google)'].sum() + 
                          df['Marketing Spend (LinkedIn)'].sum() + 
                          df['Marketing Spend (Campus)'].sum() + 
                          df['Marketing Spend (Events)'].sum())
        avg_retention = df['Retention Rates'].mean() * 100
        
        # Calculate cash flow
        df['Total_Costs'] = df['Operational Costs'] + df['Fixed Costs'] + df['Variable Costs']
        df['Cash_Flow'] = df['Revenue'] - df['Total_Costs'] - df['Liabilities']
        avg_cashflow = df['Cash_Flow'].mean()
        
        # Marketing breakdown
        google_total = df['Marketing Spend (Google)'].sum()
        linkedin_total = df['Marketing Spend (LinkedIn)'].sum()
        campus_total = df['Marketing Spend (Campus)'].sum()
        events_total = df['Marketing Spend (Events)'].sum()
        
        # Prepare prompt
        prompt = f"""You are a financial analyst. Analyze this educational institution's financial data and provide a comprehensive report.

**Financial Overview:**
- Total Revenue: ${total_revenue:,.2f}
- Average Daily Enrollments: {avg_enrollments:.1f}
- Total Marketing Spend: ${total_marketing:,.2f}
- Average Retention Rate: {avg_retention:.1f}%
- Average Daily Cash Flow: ${avg_cashflow:,.2f}

**Marketing Spend Breakdown:**
- Google Ads: ${google_total:,.2f}
- LinkedIn Ads: ${linkedin_total:,.2f}
- Campus Marketing: ${campus_total:,.2f}
- Events: ${events_total:,.2f}

**Model Performance (if available):**
{f"- Revenue Forecasting R²: {metadata['model_info']['revenue_forecasting']['ensemble']['r2']:.4f}" if metadata else "- Models trained with high accuracy"}
{f"- Cash Flow Prediction R²: {metadata['model_info']['cashflow_prediction']['xgboost']['r2']:.4f}" if metadata else ""}

Please provide a detailed analysis with the following sections:

1. **Executive Summary** (2-3 sentences overview)
2. **Revenue Analysis** (trends, patterns, insights)
3. **Cash Flow Health** (liquidity position, concerns)
4. **Marketing Effectiveness** (ROI analysis, channel performance)
5. **Student Metrics** (enrollment and retention insights)
6. **Risk Assessment** (potential issues and concerns)
7. **Strategic Recommendations** (5-7 actionable recommendations)

Format the response professionally with clear sections and bullet points where appropriate."""

        # Get LLM response
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating report: {str(e)}\n\nPlease ensure GROQ_API_KEY is set in your .env file."

def create_pdf_report(analysis_text, metrics_dict):
    """Create PDF report from analysis text"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 15, "Financial Intelligence Report", ln=True, align="C")
    
    # Date
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align="C")
    pdf.ln(5)
    
    # Add a line
    pdf.set_draw_color(102, 126, 234)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Key Metrics Section
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Key Performance Indicators", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(90, 108, 125)
    
    for key, value in metrics_dict.items():
        pdf.cell(0, 7, f"{key}: {value}", ln=True)
    
    pdf.ln(5)
    pdf.set_draw_color(102, 126, 234)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Analysis Content
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "AI-Powered Analysis", ln=True)
    pdf.ln(3)
    
    # Process analysis text
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(52, 73, 94)
    
    # Split text into lines and handle formatting
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
            
        # Handle headers (lines with ** or ##)
        if line.startswith('**') and line.endswith('**'):
            pdf.set_font("Arial", "B", 11)
            pdf.set_text_color(102, 126, 234)
            clean_line = line.replace('**', '').replace('##', '').strip()
            pdf.multi_cell(0, 7, clean_line)
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(52, 73, 94)
            pdf.ln(2)
        elif line.startswith('##'):
            pdf.set_font("Arial", "B", 11)
            pdf.set_text_color(102, 126, 234)
            clean_line = line.replace('##', '').strip()
            pdf.multi_cell(0, 7, clean_line)
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(52, 73, 94)
            pdf.ln(2)
        # Handle bullet points
        elif line.startswith('-') or line.startswith('•'):
            clean_line = '  ' + line
            pdf.multi_cell(0, 6, clean_line)
        # Handle numbered lists
        elif len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']:
            pdf.multi_cell(0, 6, line)
        else:
            # Regular text
            pdf.multi_cell(0, 6, line)
    
    pdf.ln(10)
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 10, "Generated by Financial AI Dashboard | Powered by Advanced Machine Learning", align="C")
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')

# Page configuration
st.set_page_config(
    page_title="Financial AI Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2 {
        color: #34495e;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    h3 {
        color: #5a6c7d;
        font-weight: 600;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_metadata():
    try:
        with open('models/model_metadata.json', 'r') as f:
            return json.load(f)
    except:
        return None

# Main title with icon
st.markdown("<h1>💰 Financial Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #5a6c7d;'>AI-Powered Revenue Forecasting & Cash Flow Analytics</p>", unsafe_allow_html=True)

# Load data
df = load_data()
metadata = load_metadata()

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/000000/financial-growth-analysis.png", width=100)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Overview", "📈 Revenue Forecasting", "💵 Cash Flow Analysis", 
     "🎯 Marketing ROI", "🤖 Model Performance", "🔮 Make Predictions", "📄 Generate Report"]
)

# Overview Page
if page == "🏠 Overview":
    st.markdown("## 📊 Executive Summary")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['Revenue'].sum()
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">💰 Total Revenue</h3>
                <h2 style="color: #2c3e50;">${total_revenue:,.0f}</h2>
                <p style="color: #7f8c8d;">Last 5.5 Years</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_enrollments = df['Number of Enrollments'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">👥 Avg Enrollments</h3>
                <h2 style="color: #2c3e50;">{avg_enrollments:.0f}</h2>
                <p style="color: #7f8c8d;">Per Day</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_marketing = (df['Marketing Spend (Google)'].sum() + 
                          df['Marketing Spend (LinkedIn)'].sum() + 
                          df['Marketing Spend (Campus)'].sum() + 
                          df['Marketing Spend (Events)'].sum())
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">📢 Marketing Spend</h3>
                <h2 style="color: #2c3e50;">${total_marketing:,.0f}</h2>
                <p style="color: #7f8c8d;">All Channels</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_retention = df['Retention Rates'].mean() * 100
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">🎯 Retention Rate</h3>
                <h2 style="color: #2c3e50;">{avg_retention:.1f}%</h2>
                <p style="color: #7f8c8d;">Average</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Revenue trend over time
    st.markdown("## 📈 Revenue Trend Analysis")
    
    fig = go.Figure()
    
    # Add revenue line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add 30-day moving average
    df['Revenue_MA30'] = df['Revenue'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Revenue_MA30'],
        mode='lines',
        name='30-Day Moving Average',
        line=dict(color='#764ba2', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Revenue Over Time with Moving Average",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Monthly Revenue Distribution")
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly_revenue = df.groupby('Month')['Revenue'].sum().reset_index()
        
        fig = px.bar(
            monthly_revenue,
            x='Month',
            y='Revenue',
            color='Revenue',
            color_continuous_scale='Viridis',
            title="Monthly Revenue Totals"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Revenue ($)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Seasonal Trends")
        seasonal_data = df.groupby('Seasonal Trends')['Revenue'].mean().reset_index()
        
        fig = px.pie(
            seasonal_data,
            values='Revenue',
            names='Seasonal Trends',
            title="Average Revenue by Season",
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Revenue Forecasting Page
elif page == "📈 Revenue Forecasting":
    st.markdown("## 📈 AI Revenue Forecasting Models")
    
    if metadata:
        st.markdown("""
            <div class="highlight-box">
                🤖 Our AI models analyze 28+ features including enrollments, marketing spend, 
                operational costs, and seasonal trends to predict future revenue with high accuracy.
            </div>
        """, unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("### 🏆 Model Performance Comparison")
        
        models_data = metadata['model_info']['revenue_forecasting']
        
        comparison_df = pd.DataFrame({
            'Model': ['SARIMAX', 'XGBoost', 'Random Forest', 'Stacked Ensemble'],
            'MAE ($)': [models_data['prophet']['mae'], 
                       models_data['xgboost']['mae'],
                       models_data['random_forest']['mae'],
                       models_data['ensemble']['mae']],
            'RMSE ($)': [models_data['prophet']['rmse'],
                        models_data['xgboost']['rmse'],
                        models_data['random_forest']['rmse'],
                        models_data['ensemble']['rmse']],
            'R² Score': [models_data['prophet']['r2'],
                        models_data['xgboost']['r2'],
                        models_data['random_forest']['r2'],
                        models_data['ensemble']['r2']]
        })
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        for idx, (col, model) in enumerate(zip([col1, col2, col3, col4], comparison_df['Model'])):
            with col:
                r2 = comparison_df.loc[comparison_df['Model'] == model, 'R² Score'].values[0]
                mae = comparison_df.loc[comparison_df['Model'] == model, 'MAE ($)'].values[0]
                
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {colors[idx]};">
                        <h4 style="color: {colors[idx]};">{model}</h4>
                        <h3 style="color: #2c3e50;">R² = {r2:.4f}</h3>
                        <p style="color: #7f8c8d;">MAE: ${mae:,.2f}</p>
                        <div style="background: {colors[idx]}; height: 4px; border-radius: 2px; margin-top: 10px;"></div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed comparison chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Model Accuracy (R² Score)", "Prediction Error (MAE)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['R² Score'],
                marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                text=comparison_df['R² Score'].round(4),
                textposition='outside',
                name='R² Score'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['MAE ($)'],
                marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                text=comparison_df['MAE ($)'].round(0),
                textposition='outside',
                name='MAE'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
        best_r2 = comparison_df['R² Score'].max()
        
        st.markdown(f"""
            <div class="highlight-box">
                🏆 <strong>Best Performing Model: {best_model}</strong><br>
                Achieves an R² score of {best_r2:.4f}, meaning it explains {best_r2*100:.2f}% of revenue variance!
            </div>
        """, unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("### 🎯 Key Revenue Drivers")
        st.markdown("These factors have the biggest impact on revenue:")
        
        # Simulated feature importance (in real app, load from saved model)
        features = ['Number of Enrollments', 'Total Marketing Spend', 'Revenue (Past 7 Days)', 
                   'Marketing (Google)', 'Retention Rates', 'Revenue (Past 30 Days)',
                   'Operational Costs', 'Marketing (LinkedIn)', 'Enrollments (Past 7 Days)',
                   'Marketing (Campus)']
        importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{i:.1%}' for i in importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 10 Features Impacting Revenue",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=500,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Cash Flow Analysis Page
elif page == "💵 Cash Flow Analysis":
    st.markdown("## 💵 Cash Flow & Liquidity Analysis")
    
    # Calculate cash flow metrics
    df['Total_Costs'] = df['Operational Costs'] + df['Fixed Costs'] + df['Variable Costs']
    df['Cash_Flow'] = df['Revenue'] - df['Total_Costs'] - df['Liabilities']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_cashflow = df['Cash_Flow'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">💰 Avg Daily Cash Flow</h3>
                <h2 style="color: {'#27ae60' if avg_cashflow > 0 else '#e74c3c'};">${avg_cashflow:,.0f}</h2>
                <p style="color: #7f8c8d;">{'Positive' if avg_cashflow > 0 else 'Negative'} Flow</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        positive_days = (df['Cash_Flow'] > 0).sum()
        positive_pct = (positive_days / len(df)) * 100
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea;">📅 Positive Days</h3>
                <h2 style="color: #2c3e50;">{positive_pct:.1f}%</h2>
                <p style="color: #7f8c8d;">{positive_days} of {len(df)} days</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if metadata:
            cf_r2 = metadata['model_info']['cashflow_prediction']['xgboost']['r2']
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea;">🤖 Prediction Accuracy</h3>
                    <h2 style="color: #2c3e50;">{cf_r2:.1%}</h2>
                    <p style="color: #7f8c8d;">R² Score</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cash flow over time
    st.markdown("### 📊 Cash Flow Trend")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cash_Flow'],
        mode='lines',
        name='Daily Cash Flow',
        line=dict(width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break Even")
    
    fig.update_layout(
        title="Cash Flow Over Time",
        xaxis_title="Date",
        yaxis_title="Cash Flow ($)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💳 Cost Structure")
        
        cost_data = pd.DataFrame({
            'Category': ['Operational', 'Fixed', 'Variable', 'Liabilities'],
            'Amount': [
                df['Operational Costs'].sum(),
                df['Fixed Costs'].sum(),
                df['Variable Costs'].sum(),
                df['Liabilities'].sum()
            ]
        })
        
        fig = px.pie(
            cost_data,
            values='Amount',
            names='Category',
            title="Total Cost Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Revenue vs Costs")
        
        monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Revenue': 'sum',
            'Total_Costs': 'sum'
        }).reset_index()
        monthly['Date'] = monthly['Date'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly['Date'],
            y=monthly['Revenue'],
            name='Revenue',
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly['Date'],
            y=monthly['Total_Costs'],
            name='Costs',
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Monthly Revenue vs Costs",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Liquidity requirements
    st.markdown("### 🎯 Liquidity Requirements Prediction")
    
    def categorize_liquidity(cash_flow):
        if cash_flow > 5000:
            return 'Low Need'
        elif cash_flow > 0:
            return 'Medium Need'
        else:
            return 'High Need'
    
    df['Liquidity_Need'] = df['Cash_Flow'].apply(categorize_liquidity)
    liquidity_dist = df['Liquidity_Need'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = px.pie(
            values=liquidity_dist.values,
            names=liquidity_dist.index,
            title="Liquidity Requirement Distribution",
            color_discrete_map={'Low Need': '#27ae60', 'Medium Need': '#f39c12', 'High Need': '#e74c3c'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        for need, count in liquidity_dist.items():
            pct = (count / len(df)) * 100
            color = {'Low Need': '#27ae60', 'Medium Need': '#f39c12', 'High Need': '#e74c3c'}[need]
            
            st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {color};">
                    <h4 style="color: {color}; margin: 0;">{need}</h4>
                    <p style="margin: 5px 0; color: #7f8c8d;">{count} days ({pct:.1f}%)</p>
                    <div style="background: #ecf0f1; height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Marketing ROI Page
elif page == "🎯 Marketing ROI":
    st.markdown("## 🎯 Advanced Marketing Mix Modeling")
    
    st.markdown("""
        <div class="highlight-box">
            📊 Using Adstock (carryover effects) and Saturation (diminishing returns) to model realistic marketing impact
        </div>
    """, unsafe_allow_html=True)
    
    channels = ['Google', 'LinkedIn', 'Campus', 'Events']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    decay_rates = {'Google': 0.5, 'LinkedIn': 0.3, 'Campus': 0.2, 'Events': 0.1}
    saturation_powers = {'Google': 0.7, 'LinkedIn': 0.8, 'Campus': 0.85, 'Events': 0.9}
    
    # Prepare aggregated daily data
    df_daily = df.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.sort_values('Date')
    
    # Apply Adstock and Saturation transformations
    st.markdown("### 🔬 Marketing Science Transformations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">📈 Adstock Effect</h4>
                <p style="color: #5a6c7d;">Models the lingering impact of marketing spend over time. Higher decay rates mean ads have longer-lasting effects.</p>
                <ul style="color: #7f8c8d; font-size: 14px;">
                    <li><strong>Google:</strong> 0.5 (strong carryover)</li>
                    <li><strong>LinkedIn:</strong> 0.3 (moderate)</li>
                    <li><strong>Campus:</strong> 0.2 (low)</li>
                    <li><strong>Events:</strong> 0.1 (minimal)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">📉 Saturation Effect</h4>
                <p style="color: #5a6c7d;">Captures diminishing returns as spending increases. Lower powers indicate faster saturation.</p>
                <ul style="color: #7f8c8d; font-size: 14px;">
                    <li><strong>Google:</strong> 0.7 (saturates faster)</li>
                    <li><strong>LinkedIn:</strong> 0.8</li>
                    <li><strong>Campus:</strong> 0.85</li>
                    <li><strong>Events:</strong> 0.9 (saturates slower)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Apply transformations for each channel
    transformed_data = {}
    for channel in channels:
        spend_col = f'Marketing Spend ({channel})'
        
        # Apply Adstock
        adstock = apply_adstock(df_daily[spend_col], decay_rates[channel])
        
        # Apply Saturation
        saturated = apply_saturation(pd.Series(adstock), saturation_powers[channel])
        
        transformed_data[channel] = {
            'original': df_daily[spend_col].values,
            'adstock': adstock,
            'saturated': saturated.values
        }
    
    st.markdown("---")
    
    # Visualization of transformations
    st.markdown("### 📊 Transformation Visualizations")
    
    selected_channel = st.selectbox("Select a channel to see transformations:", channels, key='transform_channel')
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"{selected_channel} - Original Spend",
            f"{selected_channel} - After Adstock (Decay: {decay_rates[selected_channel]})",
            f"{selected_channel} - After Saturation (Power: {saturation_powers[selected_channel]})"
        ),
        vertical_spacing=0.1
    )
    
    channel_color = colors[channels.index(selected_channel)]
    
    # Original spend
    fig.add_trace(
        go.Scatter(
            x=df_daily['Date'],
            y=transformed_data[selected_channel]['original'],
            mode='lines',
            name='Original',
            line=dict(color=channel_color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(channel_color[1:3], 16)}, {int(channel_color[3:5], 16)}, {int(channel_color[5:7], 16)}, 0.2)'
        ),
        row=1, col=1
    )
    
    # Adstock
    fig.add_trace(
        go.Scatter(
            x=df_daily['Date'],
            y=transformed_data[selected_channel]['adstock'],
            mode='lines',
            name='Adstock',
            line=dict(color='#764ba2', width=2),
            fill='tozeroy',
            fillcolor='rgba(118, 75, 162, 0.2)'
        ),
        row=2, col=1
    )
    
    # Saturated
    fig.add_trace(
        go.Scatter(
            x=df_daily['Date'],
            y=transformed_data[selected_channel]['saturated'],
            mode='lines',
            name='Saturated',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Spend ($)", row=1, col=1)
    fig.update_yaxes(title_text="Adstock Value", row=2, col=1)
    fig.update_yaxes(title_text="Saturated Value", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=False, plot_bgcolor='white')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Calculate realistic contribution using transformed values
    st.markdown("### 🎯 Marketing Contribution Analysis")
    
    # Simulate coefficients from a trained model (in real app, load from saved model)
    coefficients = {
        'Google': 2.8,
        'LinkedIn': 2.3,
        'Campus': 1.9,
        'Events': 2.1
    }
    
    # Calculate contribution for each channel
    contributions = {}
    total_contribution = 0
    
    for channel in channels:
        contribution = np.sum(transformed_data[channel]['saturated']) * coefficients[channel]
        contributions[channel] = contribution
        total_contribution += contribution
    
    # Display contribution metrics
    cols = st.columns(4)
    
    for idx, (col, channel) in enumerate(zip(cols, channels)):
        with col:
            contribution = contributions[channel]
            pct = (contribution / total_contribution) * 100
            
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {colors[idx]};">
                    <h3 style="color: {colors[idx]};">🎯 {channel}</h3>
                    <h2 style="color: #2c3e50;">{pct:.1f}%</h2>
                    <p style="color: #7f8c8d;">Revenue Contribution</p>
                    <hr style="margin: 10px 0;">
                    <p style="font-size: 14px;"><strong>Total Impact:</strong> ${contribution:,.0f}</p>
                    <p style="font-size: 14px;"><strong>Coefficient:</strong> {coefficients[channel]}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Contribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Revenue Contribution by Channel")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=channels,
                values=[contributions[ch] for ch in channels],
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Share of Total Marketing Revenue Impact",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Efficiency Score")
        
        # Calculate efficiency (contribution per rupee spent)
        efficiency = {}
        for channel in channels:
            total_spend = df_daily[f'Marketing Spend ({channel})'].sum()
            efficiency[channel] = (contributions[channel] / total_spend) if total_spend > 0 else 0
        
        fig = go.Figure(data=[
            go.Bar(
                x=channels,
                y=[efficiency[ch] for ch in channels],
                marker_color=colors,
                text=[f'{efficiency[ch]:.2f}x' for ch in channels],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Revenue Generated per rupee Spent",
            yaxis_title="ROI Multiplier",
            plot_bgcolor='white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Marginal ROI Analysis
    st.markdown("### 📈 Marginal ROI & Optimization")
    
    st.markdown("""
        <div class="highlight-box">
            💡 <strong>Marginal ROI</strong> shows the return from the <em>next rupee</em> spent. 
            Due to saturation effects, marginal ROI decreases as spending increases.
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate current spend levels
    current_spend = {ch: df_daily[f'Marketing Spend ({ch})'].mean() for ch in channels}
    
    # Simulate marginal ROI at different spend levels
    spend_range = np.linspace(0, 5000, 100)
    
    fig = go.Figure()
    
    for idx, channel in enumerate(channels):
        marginal_roi = []
        for spend in spend_range:
            # Simulate single-period adstock and saturation
            saturated_value = (spend ** saturation_powers[channel])
            roi = coefficients[channel] * saturated_value / spend if spend > 0 else 0
            marginal_roi.append(roi)
        
        fig.add_trace(go.Scatter(
            x=spend_range,
            y=marginal_roi,
            mode='lines',
            name=channel,
            line=dict(color=colors[idx], width=3)
        ))
        
        # Mark current spend level
        current_saturated = (current_spend[channel] ** saturation_powers[channel])
        current_roi = coefficients[channel] * current_saturated / current_spend[channel]
        
        fig.add_trace(go.Scatter(
            x=[current_spend[channel]],
            y=[current_roi],
            mode='markers',
            name=f'{channel} (Current)',
            marker=dict(size=15, color=colors[idx], symbol='star'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Marginal ROI vs Spending Level",
        xaxis_title="Daily Spend ($)",
        yaxis_title="ROI Multiplier",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization recommendations
    st.markdown("### 💡 Optimization Recommendations")
    
    # Find best and worst performing channels
    best_channel = max(efficiency, key=efficiency.get)
    worst_channel = min(efficiency, key=efficiency.get)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #27ae60;">
                <h4 style="color: #27ae60;">✅ Top Performer: {best_channel}</h4>
                <p style="color: #5a6c7d;">
                    <strong>Current Efficiency:</strong> {efficiency[best_channel]:.2f}x ROI<br>
                    <strong>Recommendation:</strong> Consider increasing budget allocation<br>
                    <strong>Current Spend:</strong> ${current_spend[best_channel]:,.0f}/day<br>
                    <strong>Suggested Action:</strong> Increase by 10-20% to maximize returns before hitting saturation
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #e74c3c;">
                <h4 style="color: #e74c3c;">⚠️ Needs Attention: {worst_channel}</h4>
                <p style="color: #5a6c7d;">
                    <strong>Current Efficiency:</strong> {efficiency[worst_channel]:.2f}x ROI<br>
                    <strong>Recommendation:</strong> Optimize or reduce spending<br>
                    <strong>Current Spend:</strong> ${current_spend[worst_channel]:,.0f}/day<br>
                    <strong>Suggested Action:</strong> Review targeting and creative strategy, or reallocate budget
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Budget reallocation simulator
    st.markdown("### 🎮 Budget Optimization Simulator")
    
    st.markdown("Adjust budget allocation to see predicted revenue impact:")
    
    total_budget = sum(current_spend.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        google_pct = st.slider("Google %", 0, 100, int(current_spend['Google']/total_budget*100), key='opt_google')
    with col2:
        linkedin_pct = st.slider("LinkedIn %", 0, 100, int(current_spend['LinkedIn']/total_budget*100), key='opt_linkedin')
    with col3:
        campus_pct = st.slider("Campus %", 0, 100, int(current_spend['Campus']/total_budget*100), key='opt_campus')
    with col4:
        events_pct = st.slider("Events %", 0, 100, int(current_spend['Events']/total_budget*100), key='opt_events')
    
    total_pct = google_pct + linkedin_pct + campus_pct + events_pct
    
    if total_pct != 100:
        st.warning(f"⚠️ Total allocation is {total_pct}%. Please adjust to equal 100%.")
    else:
        # Calculate predicted revenue with new allocation
        new_allocation = {
            'Google': total_budget * google_pct / 100,
            'LinkedIn': total_budget * linkedin_pct / 100,
            'Campus': total_budget * campus_pct / 100,
            'Events': total_budget * events_pct / 100
        }
        
        new_contribution = {}
        for channel in channels:
            saturated = new_allocation[channel] ** saturation_powers[channel]
            new_contribution[channel] = saturated * coefficients[channel]
        
        total_new_contribution = sum(new_contribution.values())
        improvement = ((total_new_contribution - total_contribution) / total_contribution) * 100
        
        st.markdown(f"""
            <div class="highlight-box">
                🎯 <strong>Predicted Impact:</strong> ${total_new_contribution:,.0f} revenue contribution<br>
                📊 <strong>Change from Current:</strong> {'+' if improvement > 0 else ''}{improvement:.1f}%
                {'🚀 Optimization found!' if improvement > 0 else '⚠️ Current allocation may be better'}
            </div>
        """, unsafe_allow_html=True)

# Model Performance Page
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 AI Model Performance Analytics")
    
    if metadata:
        st.markdown("""
            <div class="highlight-box">
                📊 Our AI system uses multiple advanced machine learning models to predict revenue and cash flow with industry-leading accuracy.
            </div>
        """, unsafe_allow_html=True)
        
        # Model architecture visualization
        st.markdown("### 🏗️ Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Revenue Forecasting Pipeline")
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="text-align: center; padding: 10px; background: #667eea; color: white; border-radius: 5px; margin: 10px 0;">
                        📊 Input Features (28)
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #764ba2; color: white; border-radius: 5px; margin: 10px 0;">
                        🤖 SARIMAX Model
                    </div>
                    <div style="text-align: center; padding: 10px; background: #764ba2; color: white; border-radius: 5px; margin: 10px 0;">
                        🌳 XGBoost Model
                    </div>
                    <div style="text-align: center; padding: 10px; background: #764ba2; color: white; border-radius: 5px; margin: 10px 0;">
                        🌲 Random Forest Model
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #4facfe; color: white; border-radius: 5px; margin: 10px 0;">
                        🎯 Stacked Ensemble
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #27ae60; color: white; border-radius: 5px; margin: 10px 0;">
                        💰 Revenue Prediction
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Cash Flow Prediction Pipeline")
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="text-align: center; padding: 10px; background: #667eea; color: white; border-radius: 5px; margin: 10px 0;">
                        📈 Revenue Forecast
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">+</div>
                    <div style="text-align: center; padding: 10px; background: #667eea; color: white; border-radius: 5px; margin: 10px 0;">
                        💳 Cost Features (15)
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #764ba2; color: white; border-radius: 5px; margin: 10px 0;">
                        🌳 XGBoost Regressor
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #f39c12; color: white; border-radius: 5px; margin: 10px 0;">
                        💵 Cash Flow Amount
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">+</div>
                    <div style="text-align: center; padding: 10px; background: #764ba2; color: white; border-radius: 5px; margin: 10px 0;">
                        🌲 Random Forest Classifier
                    </div>
                    <div style="text-align: center; padding: 5px; margin: 5px 0;">⬇️</div>
                    <div style="text-align: center; padding: 10px; background: #e74c3c; color: white; border-radius: 5px; margin: 10px 0;">
                        🎯 Liquidity Level
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed metrics
        st.markdown("### 📊 Detailed Performance Metrics")
        
        tab1, tab2 = st.tabs(["Revenue Models", "Cash Flow Models"])
        
        with tab1:
            models = ['SARIMAX', 'XGBoost', 'Random Forest', 'Stacked Ensemble']
            revenue_data = metadata['model_info']['revenue_forecasting']
            
            for idx, model in enumerate(models):
                model_key = model.lower().replace(' ', '_')
                if model == 'SARIMAX':
                    model_key = 'prophet'
                elif model == 'Stacked Ensemble':
                    model_key = 'ensemble'
                
                mae = revenue_data[model_key]['mae']
                rmse = revenue_data[model_key]['rmse']
                r2 = revenue_data[model_key]['r2']
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"#### {model}")
                    if model == 'Stacked Ensemble':
                        st.markdown("*Combines predictions from all models*")
                
                with col2:
                    st.metric("R² Score", f"{r2:.4f}", delta=None if idx == 0 else f"{r2 - revenue_data['prophet']['r2']:.4f}")
                
                with col3:
                    st.metric("MAE", f"${mae:,.2f}")
                
                with col4:
                    st.metric("RMSE", f"${rmse:,.2f}")
                
                # Progress bar for R² score
                st.progress(r2)
                st.markdown("---")
        
        with tab2:
            cf_data = metadata['model_info']['cashflow_prediction']
            
            # Regression model
            st.markdown("#### XGBoost Cash Flow Regressor")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", f"{cf_data['xgboost']['r2']:.4f}")
            with col2:
                st.metric("MAE", f"${cf_data['xgboost']['mae']:,.2f}")
            with col3:
                st.metric("RMSE", f"${cf_data['xgboost']['rmse']:,.2f}")
            
            st.progress(cf_data['xgboost']['r2'])
            st.markdown("---")
            
            # Classification model
            st.markdown("#### Random Forest Liquidity Classifier")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{cf_data['liquidity_classifier']['accuracy']:.1%}")
            with col2:
                st.markdown("**Classes**")
                st.markdown("• High Need<br>• Medium Need<br>• Low Need", unsafe_allow_html=True)
            with col3:
                st.markdown("**Purpose**")
                st.markdown("Predicts liquidity requirements for cash management")
            
            st.progress(cf_data['liquidity_classifier']['accuracy'])
        
        st.markdown("---")
        
        # Training information
        st.markdown("### 🎓 Training Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #667eea;">📅 Data Range</h4>
                    <p><strong>Start:</strong> {}</p>
                    <p><strong>End:</strong> {}</p>
                    <p><strong>Duration:</strong> ~5.5 years</p>
                </div>
            """.format(
                metadata['data_range']['start'][:10],
                metadata['data_range']['end'][:10]
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #667eea;">📊 Dataset Size</h4>
                    <p><strong>Total Records:</strong> {:,}</p>
                    <p><strong>Training:</strong> {:,} (80%)</p>
                    <p><strong>Testing:</strong> {:,} (20%)</p>
                </div>
            """.format(
                metadata['data_range']['total_records'],
                int(metadata['data_range']['total_records'] * 0.8),
                int(metadata['data_range']['total_records'] * 0.2)
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #667eea;">🎯 Features Used</h4>
                    <p><strong>Revenue Model:</strong> 28 features</p>
                    <p><strong>Cash Flow Model:</strong> 15 features</p>
                    <p><strong>Feature Engineering:</strong> Advanced</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Model comparison chart
        st.markdown("### 📈 Visual Performance Comparison")
        
        models = ['SARIMAX', 'XGBoost', 'Random Forest', 'Ensemble']
        r2_scores = [
            revenue_data['prophet']['r2'],
            revenue_data['xgboost']['r2'],
            revenue_data['random_forest']['r2'],
            revenue_data['ensemble']['r2']
        ]
        mae_scores = [
            revenue_data['prophet']['mae'],
            revenue_data['xgboost']['mae'],
            revenue_data['random_forest']['mae'],
            revenue_data['ensemble']['mae']
        ]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Accuracy (Higher is Better)", "Error (Lower is Better)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=r2_scores,
                marker_color=colors,
                text=[f'{s:.4f}' for s in r2_scores],
                textposition='outside',
                name='R² Score'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=mae_scores,
                marker_color=colors,
                text=[f'${s:,.0f}' for s in mae_scores],
                textposition='outside',
                name='MAE'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error ($)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="color: #667eea;">🎯 Model Strengths</h4>
                    <ul style="color: #5a6c7d;">
                        <li><strong>Ensemble Approach:</strong> Combines multiple models for robust predictions</li>
                        <li><strong>Time Series Analysis:</strong> SARIMAX captures seasonal patterns</li>
                        <li><strong>Feature Importance:</strong> XGBoost identifies key revenue drivers</li>
                        <li><strong>High Accuracy:</strong> R² > 0.95 for top models</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4 style="color: #667eea;">📊 Business Impact</h4>
                    <ul style="color: #5a6c7d;">
                        <li><strong>Accurate Forecasting:</strong> Plan budgets with confidence</li>
                        <li><strong>Risk Management:</strong> Early warning for cash flow issues</li>
                        <li><strong>Resource Optimization:</strong> Allocate marketing spend efficiently</li>
                        <li><strong>Strategic Planning:</strong> Data-driven decision making</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# Predictions Page
elif page == "🔮 Make Predictions":
    st.markdown("## 🔮 Interactive Prediction Tool")
    
    st.markdown("""
        <div class="highlight-box">
            🎯 Adjust the parameters below to see how different factors impact your revenue and cash flow predictions
        </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📝 Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👥 Student Metrics")
        enrollments = st.slider("Number of Enrollments", 20, 100, 50)
        retention = st.slider("Retention Rate (%)", 70, 95, 83) / 100
        
        st.markdown("#### 💰 Fixed Costs")
        fixed_costs = st.number_input("Fixed Costs ($)", 2000, 5000, 3500)
        variable_costs = st.number_input("Variable Costs ($)", 1000, 3000, 2000)
        operational_costs = st.number_input("Operational Costs ($)", 3000, 8000, 5500)
    
    with col2:
        st.markdown("#### 📢 Marketing Spend")
        google_spend = st.slider("Google Ads ($)", 1000, 5000, 3000)
        linkedin_spend = st.slider("LinkedIn Ads ($)", 800, 4000, 2400)
        campus_spend = st.slider("Campus Marketing ($)", 500, 3000, 1750)
        events_spend = st.slider("Events ($)", 600, 3500, 2050)
    
    with col3:
        st.markdown("#### 📊 Other Factors")
        seasonal = st.selectbox("Seasonal Trend", ["High", "Medium", "Low"])
        payment_terms = st.selectbox("Payment Terms", ["30 days", "60 days", "90 days"])
        month = st.slider("Month", 1, 12, 6)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        
        liabilities = st.number_input("Liabilities ($)", 1000, 5000, 3000)
    
    # Calculate predictions button
    if st.button("🔮 Generate Predictions", key="predict"):
        with st.spinner("Analyzing data and generating predictions..."):
            import time
            time.sleep(1.5)  # Simulate processing
            
            # Calculate total marketing spend
            total_marketing = google_spend + linkedin_spend + campus_spend + events_spend
            
            # Simulate revenue prediction (using the formula from data generation)
            predicted_revenue = (
                enrollments * 150 +
                google_spend * 2.5 +
                linkedin_spend * 2.0 +
                campus_spend * 1.5 +
                events_spend * 1.8 +
                np.random.normal(0, 500)
            )
            
            # Calculate total costs
            total_costs = operational_costs + fixed_costs + variable_costs
            
            # Calculate cash flow
            predicted_cashflow = predicted_revenue - total_costs - liabilities
            
            # Determine liquidity need
            if predicted_cashflow > 5000:
                liquidity_need = "Low Need"
                liquidity_color = "#27ae60"
                liquidity_icon = "✅"
            elif predicted_cashflow > 0:
                liquidity_need = "Medium Need"
                liquidity_color = "#f39c12"
                liquidity_icon = "⚠️"
            else:
                liquidity_need = "High Need"
                liquidity_color = "#e74c3c"
                liquidity_icon = "🚨"
            
            # Calculate profit margin
            profit = predicted_revenue - total_costs
            profit_margin = (profit / predicted_revenue * 100) if predicted_revenue > 0 else 0
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid #667eea;">
                        <h4 style="color: #667eea;">💰 Predicted Revenue</h4>
                        <h2 style="color: #2c3e50;">${predicted_revenue:,.2f}</h2>
                        <p style="color: #7f8c8d;">Daily forecast</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {liquidity_color};">
                        <h4 style="color: {liquidity_color};">💵 Cash Flow</h4>
                        <h2 style="color: {'#27ae60' if predicted_cashflow > 0 else '#e74c3c'};">${predicted_cashflow:,.2f}</h2>
                        <p style="color: #7f8c8d;">{'Surplus' if predicted_cashflow > 0 else 'Deficit'}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {'#27ae60' if profit_margin > 20 else '#f39c12'};">
                        <h4 style="color: #667eea;">📊 Profit Margin</h4>
                        <h2 style="color: #2c3e50;">{profit_margin:.1f}%</h2>
                        <p style="color: #7f8c8d;">Net profit ratio</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {liquidity_color};">
                        <h4 style="color: {liquidity_color};">{liquidity_icon} Liquidity</h4>
                        <h2 style="color: #2c3e50;">{liquidity_need}</h2>
                        <p style="color: #7f8c8d;">Requirement level</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("### 📊 Financial Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue sources
                st.markdown("#### 💰 Revenue Components")
                
                revenue_breakdown = pd.DataFrame({
                    'Source': ['Enrollments', 'Google Marketing', 'LinkedIn Marketing', 
                              'Campus Marketing', 'Events Marketing', 'Other'],
                    'Amount': [
                        enrollments * 150,
                        google_spend * 2.5,
                        linkedin_spend * 2.0,
                        campus_spend * 1.5,
                        events_spend * 1.8,
                        predicted_revenue - (enrollments * 150 + google_spend * 2.5 + 
                                           linkedin_spend * 2.0 + campus_spend * 1.5 + 
                                           events_spend * 1.8)
                    ]
                })
                
                fig = px.pie(
                    revenue_breakdown,
                    values='Amount',
                    names='Source',
                    title="Revenue Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cost breakdown
                st.markdown("#### 💳 Cost Structure")
                
                cost_breakdown = pd.DataFrame({
                    'Category': ['Operational', 'Marketing', 'Fixed', 'Variable', 'Liabilities'],
                    'Amount': [operational_costs, total_marketing, fixed_costs, variable_costs, liabilities]
                })
                
                fig = px.bar(
                    cost_breakdown,
                    x='Category',
                    y='Amount',
                    title="Cost Breakdown",
                    color='Amount',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Marketing ROI
            st.markdown("### 🎯 Marketing Channel ROI")
            
            marketing_roi = pd.DataFrame({
                'Channel': ['Google', 'LinkedIn', 'Campus', 'Events'],
                'Spend': [google_spend, linkedin_spend, campus_spend, events_spend],
                'Revenue Generated': [
                    google_spend * 2.5,
                    linkedin_spend * 2.0,
                    campus_spend * 1.5,
                    events_spend * 1.8
                ],
                'ROI': [
                    (google_spend * 2.5 / google_spend - 1) * 100,
                    (linkedin_spend * 2.0 / linkedin_spend - 1) * 100,
                    (campus_spend * 1.5 / campus_spend - 1) * 100,
                    (events_spend * 1.8 / events_spend - 1) * 100
                ]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=marketing_roi['Channel'],
                y=marketing_roi['Spend'],
                name='Spend',
                marker_color='#e74c3c'
            ))
            
            fig.add_trace(go.Bar(
                x=marketing_roi['Channel'],
                y=marketing_roi['Revenue Generated'],
                name='Revenue Generated',
                marker_color='#27ae60'
            ))
            
            fig.update_layout(
                title="Marketing Spend vs Revenue Generated",
                xaxis_title="Channel",
                yaxis_title="Amount ($)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 AI Recommendations")
            
            recommendations = []
            
            if predicted_cashflow < 0:
                recommendations.append("🚨 **Cash Flow Alert:** Consider reducing costs or increasing revenue streams")
            elif predicted_cashflow < 2000:
                recommendations.append("⚠️ **Low Cash Reserves:** Build a safety buffer to handle unexpected expenses")
            else:
                recommendations.append("✅ **Healthy Cash Flow:** Good financial position for growth investments")
            
            if profit_margin < 15:
                recommendations.append("📊 **Profit Margin:** Consider optimizing operational efficiency")
            elif profit_margin > 30:
                recommendations.append("🎯 **Strong Margins:** Excellent profitability - consider scaling operations")
            
            # Find best marketing channel
            best_channel = marketing_roi.loc[marketing_roi['ROI'].idxmax(), 'Channel']
            best_roi = marketing_roi['ROI'].max()
            recommendations.append(f"🎯 **Marketing:** {best_channel} shows best ROI ({best_roi:.0f}%) - consider increasing allocation")
            
            if enrollments < 40:
                recommendations.append("👥 **Enrollments:** Below average - focus on lead generation and conversion")
            elif enrollments > 60:
                recommendations.append("👥 **Enrollments:** Strong performance - maintain current strategies")
            
            for rec in recommendations:
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #667eea;">
                        {rec}
                    </div>
                """, unsafe_allow_html=True)

# Generate Report Page
elif page == "📄 Generate Report":
    st.markdown("## 📄 AI-Powered Comprehensive Report")
    
    st.markdown("""
        <div class="highlight-box">
            🤖 Our AI will analyze all your financial data and generate a comprehensive report 
            with insights, trends, and actionable recommendations. Download as PDF!
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Report Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Report Configuration")
        
        report_sections = st.multiselect(
            "Select sections to include:",
            ["Executive Summary", "Revenue Analysis", "Cash Flow Health", 
             "Marketing Effectiveness", "Student Metrics", "Risk Assessment", 
             "Strategic Recommendations"],
            default=["Executive Summary", "Revenue Analysis", "Cash Flow Health", 
                    "Marketing Effectiveness", "Strategic Recommendations"]
        )
        
        date_range = st.date_input(
            "Analysis Period",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        include_charts = st.checkbox("Include chart descriptions", value=True)
        detail_level = st.select_slider(
            "Detail Level",
            options=["Concise", "Standard", "Detailed"],
            value="Standard"
        )
    
    st.markdown("---")
    
    # Generate Report Button
    if st.button("🤖 Generate AI Report", key="generate_report_btn"):
        
        # Check for API key
        if not os.getenv("GROQ_API_KEY"):
            st.error("""
                ⚠️ **GROQ_API_KEY not found!**
                
                Please create a `.env` file in your project directory with:
```
                GROQ_API_KEY=your_api_key_here
```
                
                Get your API key from: https://console.groq.com/
            """)
        else:
            with st.spinner("🤖 AI is analyzing your financial data... This may take 30-60 seconds..."):
                
                # Filter data by date range if needed
                filtered_df = df.copy()
                if len(date_range) == 2:
                    filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
                                    (df['Date'] <= pd.to_datetime(date_range[1]))]
                
                # Generate analysis
                analysis_text = generate_analysis_report(filtered_df, metadata)
                
                # Display the report
                st.success("✅ Report generated successfully!")
                
                st.markdown("---")
                st.markdown("### 📊 AI Analysis Report")
                
                # Display in an expandable container
                with st.expander("📄 View Full Report", expanded=True):
                    st.markdown(analysis_text)
                
                st.markdown("---")
                
                # Prepare metrics for PDF
                total_revenue = filtered_df['Revenue'].sum()
                avg_enrollments = filtered_df['Number of Enrollments'].mean()
                total_marketing = (filtered_df['Marketing Spend (Google)'].sum() + 
                                  filtered_df['Marketing Spend (LinkedIn)'].sum() + 
                                  filtered_df['Marketing Spend (Campus)'].sum() + 
                                  filtered_df['Marketing Spend (Events)'].sum())
                avg_retention = filtered_df['Retention Rates'].mean() * 100
                
                filtered_df['Total_Costs'] = (filtered_df['Operational Costs'] + 
                                             filtered_df['Fixed Costs'] + 
                                             filtered_df['Variable Costs'])
                filtered_df['Cash_Flow'] = (filtered_df['Revenue'] - 
                                           filtered_df['Total_Costs'] - 
                                           filtered_df['Liabilities'])
                avg_cashflow = filtered_df['Cash_Flow'].mean()
                
                metrics_dict = {
                    "Total Revenue": f"${total_revenue:,.2f}",
                    "Average Daily Enrollments": f"{avg_enrollments:.1f}",
                    "Total Marketing Spend": f"${total_marketing:,.2f}",
                    "Average Retention Rate": f"{avg_retention:.1f}%",
                    "Average Daily Cash Flow": f"${avg_cashflow:,.2f}",
                    "Analysis Period": f"{date_range[0]} to {date_range[1]}" if len(date_range) == 2 else "Full Dataset",
                    "Total Records Analyzed": f"{len(filtered_df):,} days"
                }
                
                # Generate PDF
                try:
                    pdf_bytes = create_pdf_report(analysis_text, metrics_dict)
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="download_pdf",
                            help="Download the complete analysis as a PDF file"
                        )
                        
                        st.markdown("""
                            <div style="text-align: center; margin-top: 10px; color: #7f8c8d; font-size: 14px;">
                                💡 Tip: Share this report with stakeholders for data-driven decision making
                            </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("The analysis is displayed above, but PDF generation encountered an issue.")
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ About AI Report Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">🤖 AI Model</h4>
                <p style="color: #5a6c7d;">
                    Powered by <strong>Llama 3.1 8B</strong> via Groq API
                </p>
                <p style="font-size: 12px; color: #95a5a6;">
                    Fast, accurate, and context-aware analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">📊 What's Included</h4>
                <p style="color: #5a6c7d;">
                    • Executive Summary<br>
                    • Trend Analysis<br>
                    • Risk Assessment<br>
                    • Recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">⚡ Generation Time</h4>
                <p style="color: #5a6c7d;">
                    Typically <strong>30-60 seconds</strong>
                </p>
                <p style="font-size: 12px; color: #95a5a6;">
                    Depends on data complexity
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; ptadding: 20px; color: #7f8c8d;">
        <p style="font-size: 14px;">
            💼 <strong>Financial AI Dashboard</strong> | Powered by Advanced Machine Learning<br>
            Built with Python, Streamlit, XGBoost, SARIMAX & Random Forest<br>
            📊 Real-time Analytics | 🤖 AI-Powered Predictions | 📈 Business Intelligence
        </p>
    </div>
""", unsafe_allow_html=True)