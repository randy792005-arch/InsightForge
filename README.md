# InsightForge: An Agentic AI-Based Retail Business Intelligence and Decision Support System

InsightForge is an academic-grade, production-ready Retail Business Intelligence (BI) and Decision Support System designed with a multi-agent AI architecture. The system processes raw retail sales and inventory datasets, cleans and validates records, conducts exploratory analytics, trains predictive time-series (Prophet) and machine learning (XGBoost) forecasting models, calculates replenishment and liquidation recommendations, and presents the entire workflow through an interactive Streamlit web dashboard and automated executive PDF reports.

---

## ?? Key Capabilities & Features

- **Multi-Agent AI Architecture:** Five specialized, autonomous agents coordinating in an automated pipeline.
- **Smart Data Ingestion:** Auto-detects column semantics across diverse retail dataset schemas (CSV / Excel).
- **Automated Data Cleansing & Quality Scoring:** Handles duplicates, missing values, non-positive records, and computes a comprehensive Data Quality Score (0?100%).
- **Deep Retail Analytics:** Real-time business KPIs (Revenue, Volume, AOV, Unique SKUs), daily/monthly trends, category performance, and Pareto (80/20) revenue distribution.
- **Dual Machine Learning Forecasting:**
  - **Facebook Prophet:** Additive time-series model capturing weekly and annual seasonality with confidence interval bounds.
  - **XGBoost Regressor:** Gradient-boosted trees trained with lag features (1, 2, 3, 7, 14, 21, 28) and rolling statistical windows.
  - **Authentic Evaluation Metrics:** Calculates **MAE**, **RMSE**, **MAPE**, and **R?** using hold-out backtesting. Zero fabricated metrics.
- **Actionable Inventory Decision Support:**
  - Emergency stockout alerts (< 7 days of supply) with exact target reorder quantities.
  - Fast-mover merchandising recommendations.
  - Overstock and dead stock markdown/clearance strategies (> 90 days of supply).
- **SQLite Database Integration:** Full relational persistence for raw ingestion, cleaned transactions, cached KPIs, forecast series, decision logs, and agent audit trails.
- **Executive PDF Report Generator:** Publication-ready, styled PDF export via ReportLab containing KPI tables, category performance, ML forecast evaluations, and prioritized recommendation action items.
- **100% Free & Open-Source:** Operates completely offline with zero paid API requirements.

---

## ??? Project Structure

```
InsightForge/
?
??? app.py                      # Main Streamlit application entry point
??? requirements.txt            # Complete dependency requirements
??? test_pipeline.py            # Automated end-to-end verification test suite
??? README.md                   # System documentation & usage guide
?
??? database/
?   ??? __init__.py
?   ??? db_manager.py           # SQLite connection manager and CRUD operations
?   ??? retail.db               # SQLite database file (created automatically)
?
??? agents/
?   ??? __init__.py
?   ??? base_agent.py           # Base agent class and AgentResult dataclass
?   ??? ingestion.py            # Agent 1: Data Ingestion Agent
?   ??? cleaning.py             # Agent 2: Data Cleaning Agent
?   ??? analytics.py            # Agent 3: Analytics Agent
?   ??? forecasting.py          # Agent 4: Demand Forecasting Agent
?   ??? recommendation.py       # Agent 5: Recommendation Agent
?
??? models/
?   ??? __init__.py
?   ??? prophet_model.py        # Prophet time-series forecasting wrapper
?   ??? xgboost_model.py        # XGBoost feature-engineered forecasting wrapper
?
??? dashboard/
?   ??? __init__.py
?   ??? dashboard.py            # Modular Streamlit UI layouts and page routing
?   ??? charts.py               # Interactive Plotly chart generation suite
?
??? utils/
?   ??? __init__.py
?   ??? preprocess.py           # Column detection, date parsing, quality score
?   ??? pdf_report.py           # ReportLab executive PDF generation engine
?
??? datasets/
?   ??? sample_retail_sales.csv # Realistic benchmark dataset (17,400+ transactions)
??? uploads/                    # Uploaded user datasets
??? reports/                    # Generated executive PDF reports
??? assets/                     # UI visual assets
```

---

## ?? Multi-Agent AI Workflow

```
Retail Dataset (CSV / Excel)
            ?
1. Data Ingestion Agent    -> Validates schema, detects columns, stores raw records in SQLite
            ?
2. Data Cleaning Agent     -> Deduplicates, standardizes dates, imputes missing fields, computes Quality Score
            ?
3. Analytics Agent         -> Computes KPIs, sales trends, category shares, Pareto 80/20 curve
            ?
4. Forecasting Agent       -> Trains Prophet/XGBoost, computes MAE/RMSE/MAPE, projects future demand
            ?
5. Recommendation Agent    -> Evaluates stock vs. demand, generates reorder orders & clearance actions
            ?
Interactive Dashboard & Executive PDF Report
```

---

## ?? Installation and Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git (optional)

### 2. Install Dependencies
Navigate to the project directory and install the required packages:

```bash
cd InsightForge
pip install -r requirements.txt
```

### 3. Run Automated Integration Tests
To verify all agents, models, database operations, and PDF generation:

```bash
python test_pipeline.py
```

### 4. Launch the Web Application
Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## ?? Dashboard Navigation Guide

1. **?? Executive BI Dashboard:** High-level executive KPI cards, 5-agent visual stepper, sales revenue trends with 7-day moving averages, category donut chart, and critical stockout alerts.
2. **?? Data Ingestion:** Upload custom CSV/Excel retail files or load the built-in 17,400-record benchmark dataset with 1 click.
3. **?? Data Cleaning:** Review the Data Quality Score (0?100%), completeness/uniqueness breakdown, and inspect cleaned records.
4. **?? Sales Analytics:** Interactive Plotly charts for daily/monthly sales, day-of-week distributions, top 10 best sellers, bottom 10 slow movers, and Pareto (80/20) analysis.
5. **?? Demand Forecasting:** Select between Prophet and XGBoost, adjust forecast horizon (7?90 days), and review authentic out-of-sample backtest validation metrics (MAE, RMSE, MAPE, R?).
6. **?? Inventory & Recommendations:** Interactive Inventory Health Matrix (Stock vs. Sales Velocity) and prioritized decision support table with reorder quantities and financial impact estimates.
7. **?? Executive PDF Report:** One-click generation and instant download of publication-ready PDF intelligence briefs.

---

## ?? Testing with Custom Datasets

InsightForge automatically recognizes common column naming variations:

| Concept | Supported Column Names / Aliases |
| :--- | :--- |
| **Date** | `Date`, `Order_Date`, `InvoiceDate`, `Transaction_Date`, `Sale_Date`, `Day` |
| **Product** | `Product`, `Product_Name`, `Item`, `Item_Name`, `Description`, `Product_Title` |
| **Category** | `Category`, `Product_Category`, `Department`, `Dept`, `Segment`, `Group` |
| **Quantity** | `Quantity`, `Qty`, `Units`, `Units_Sold`, `Volume`, `Count` |
| **Price** | `Unit_Price`, `Price`, `Selling_Price`, `Rate`, `Retail_Price` |
| **Sales** | `Total_Sales`, `Sales`, `Revenue`, `Total_Amount`, `Amount`, `Net_Sales` |
| **Stock** | `Current_Stock`, `Stock`, `Inventory`, `Stock_Level`, `Quantity_On_Hand` |

---

## ?? Academic Integrity Note

All predictive metrics (MAE, RMSE, MAPE, R?), time-series forecasts, and replenishment recommendations are computed deterministically from the underlying dataset. No results are hard-coded or fabricated.
