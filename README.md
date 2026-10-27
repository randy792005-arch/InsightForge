# InsightForge: Autonomous Multi-Agent Retail BI & Decision Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Database-SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Models-Prophet_%26_XGBoost](https://img.shields.io/badge/Models-Prophet%20%7C%20XGBoost-orange.svg)]()
[![Currency-INR_%E2%82%B9](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-green.svg)]()
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

> **InsightForge** transforms raw retail sales datasets into autonomous business intelligence, predictive demand forecasts, and prescriptive inventory decisions — calibrated specifically for the **Indian retail market** with all metrics, prices, and revenue in **Indian Rupees (₹)**.

---

## 1. What is InsightForge? (In Simple Words)

Running a retail store or e-commerce business comes with everyday headaches:
- **Stockouts:** You run out of high-demand items, losing revenue and sending loyal customers to competitors.
- **Overstock:** Slow-moving items gather dust on warehouse shelves, tying up cash that could be used to grow your business.
- **Traditional Dashboards:** Most dashboards only show what happened in the past. They do **not** tell you *what will sell tomorrow* or *how many units to reorder today*.

**InsightForge solves this.** Think of it as an **autonomous team of 5 AI retail analysts working 24/7 for your business**:
1. You upload any sales CSV or Excel file.
2. The AI agents inspect, clean, and analyze your sales history.
3. Machine learning models predict customer demand over the next 7 to 90 days.
4. The system calculates exact purchase orders to replenish low stock and promotional markdowns to liquidate dead stock.
5. You can explore everything through an interactive web dashboard or download an executive PDF report for management review.

---

## 2. Meet the 5 Autonomous AI Agents

InsightForge operates as an autonomous multi-agent pipeline where each specialized agent performs a dedicated role:

```mermaid
flowchart TD
    A["📁 Raw Retail Sales File (CSV / Excel)"] --> B["📥 1. Ingestion Agent<br/><i>Schema detection & SQLite storage</i>"]
    B --> C["🧹 2. Cleaning Agent<br/><i>Deduplication, date parsing & quality score</i>"]
    C --> D["📊 3. Analytics Agent<br/><i>Core KPIs, Pareto 80/20 & trends in ₹ (INR)</i>"]
    D --> E["📈 4. Forecasting Agent<br/><i>Prophet & XGBoost ML demand projections</i>"]
    E --> F["💡 5. Recommendation Agent<br/><i>Stockout alerts, reorder POs & liquidation</i>"]
    F --> G["🖥️ Interactive Streamlit Dashboard<br/><i>(7 Modular Management Views)</i>"]
    F --> H["📑 Executive PDF Intelligence Brief<br/><i>(One-click ReportLab export)</i>"]

    style A fill:#F8FAFC,stroke:#94A3B8,stroke-width:2px,color:#0F172A
    style B fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A
    style C fill:#F0FDF4,stroke:#10B981,stroke-width:2px,color:#065F46
    style D fill:#EEF2FF,stroke:#6366F1,stroke-width:2px,color:#3730A3
    style E fill:#FFFBEB,stroke:#F59E0B,stroke-width:2px,color:#92400E
    style F fill:#FEF2F2,stroke:#EF4444,stroke-width:2px,color:#991B1B
    style G fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#FFFFFF
    style H fill:#1E293B,stroke:#F43F5E,stroke-width:2px,color:#FFFFFF
```

| Agent | Name | Real-World Role | What It Does |
| :---: | :--- | :--- | :--- |
| **01** | **Data Ingestion Agent** | *The Data Receiver* | Automatically detects column headers across messy CSV formats (dates, products, sales, quantities) and securely archives raw records in SQLite. |
| **02** | **Data Cleaning Agent** | *The Quality Inspector* | Strips currency symbols (₹, $, commas), handles negative returns, standardizes timestamps, fixes missing values, and scores data health (0% to 100%). |
| **03** | **Analytics Agent** | *The Business Analyst* | Computes total revenue, orders, units, average order value (AOV), top/bottom products, and identifies which 20% of products generate 80% of revenue (Pareto Principle). |
| **04** | **Forecasting Agent** | *The Future Predictor* | Uses **Facebook Prophet** and **XGBoost** to forecast daily revenue for the next 7 to 90 days with authentic backtesting metrics (MAE, RMSE, MAPE, R²). |
| **05** | **Recommendation Agent** | *The Smart Decision Advisor* | Compares current inventory velocity against supplier lead times to generate prioritized action items: emergency replenishment POs, safety stock buffers, and clearance markdowns. |

---

## 3. Key Features

- **100% Indian Retail Context (₹)**: All metric cards, charts, hover tooltips, data tables, and PDF reports display amounts in Indian Rupees (₹). Zero foreign dollar clutter.
- **Universal CSV / Excel Support**: Upload any retail transaction sheet. InsightForge intelligently identifies semantic concepts like *InvoiceDate*, *Selling Price*, *MRP*, *Item Name*, and *Stock*.
- **Dual Forecasting Algorithms**:
  - **Facebook Prophet**: Robust against seasonal swings, holiday spikes, and weekly cycles.
  - **XGBoost Regressor**: High-performance gradient-boosted trees with dynamic rolling lag windows.
- **Authentic ML Backtesting**: Measures accuracy using real hold-out test sets. Calculates genuine Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). Zero fabricated metrics.
- **Prescriptive Decision Matrix**:
  - **Critical Stockout Alerts**: Highlights products with less than 7 days of supply remaining with estimated lost revenue protected.
  - **Overstock Liquidation**: Flags slow-moving items (> 90 days cover) with suggested clearance markdowns (20% to 30%) to free tied-up working capital.
- **One-Click Executive PDF Export**: Built-in ReportLab engine with registered TrueType Unicode fonts that compiles KPIs, category tables, ML accuracy cards, and decision recommendations into an executive briefing document.
- **100% Free, Private & Offline**: Powered by local SQLite. No monthly cloud fees, no API keys, and your sales data never leaves your machine.

---

## 4. Quick Start in 3 Minutes

### Step 1: Clone the Repository
```bash
git clone https://github.com/randy792005-arch/InsightForge.git
cd InsightForge
```

### Step 2: Install Dependencies
Ensure you have Python 3.10 or newer installed:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Verification Suite (Optional but Recommended)
Verify that all 5 agents, the SQLite database, ML models, and PDF engine are working:
```bash
python test_pipeline.py
```
*Expected output: `STATUS: ALL TESTS PASSED (100% OPERATIONAL)`*

### Step 4: Launch the Web Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 5. Dashboard Page Guide

InsightForge features 7 dedicated pages accessible from the sidebar:

1. **Executive BI Dashboard**: High-level store summary with 5 executive KPI cards (Total Revenue in ₹, Units Sold, Orders, AOV, Active SKUs), 5-agent visual stepper, 7-day moving average revenue trend, category share donut, and quick file upload.
2. **Data Ingestion (Agent 1)**: Upload any custom CSV or Excel file or load the built-in 17,400+ transaction benchmark dataset with 1 click.
3. **Data Cleaning (Agent 2)**: View the Data Quality Score (0% to 100%), inspection logs, missing value resolution, and preview the normalized dataset.
4. **Sales Analytics (Agent 3)**: Interactive Plotly visualizations for Daily Revenue, Monthly Trends, Day-of-Week patterns, Top 10 Best Sellers, Bottom 10 Slow Movers, and Pareto (80/20) revenue concentration.
5. **Demand Forecasting (Agent 4)**: Choose between Facebook Prophet and XGBoost, adjust forecast horizon (7 to 90 days), view confidence interval ribbons, and inspect genuine hold-out validation metrics (MAE & RMSE in ₹).
6. **Inventory & Recommendations (Agent 5)**: Interactive Stock vs. Velocity matrix and actionable decision table with emergency replenishment quantities and clearance actions.
7. **Executive PDF Report**: Generate and instantly download a styled PDF brief summarizing all business KPIs, category tables, forecasting accuracy, and action items.

---

## 6. Supported CSV Format & Smart Detection

InsightForge automatically detects and maps column headers even if they use different names:

| Standard Concept | Automatically Recognized Column Names |
| :--- | :--- |
| **Date** | `Date`, `Order_Date`, `InvoiceDate`, `Transaction_Date`, `Sale_Date`, `Day`, `Timestamp` |
| **Product Name** | `Product`, `Product_Name`, `Item`, `Item_Name`, `Description`, `Product_Title`, `SKU_Name` |
| **Category** | `Category`, `Product_Category`, `Department`, `Dept`, `Segment`, `Group`, `Product_Type` |
| **Quantity** | `Quantity`, `Qty`, `Units`, `Units_Sold`, `Volume`, `Count`, `Pieces`, `Sales_Volume` |
| **Unit Price** | `Unit_Price`, `Price`, `Selling_Price`, `Rate`, `MRP`, `Cost_Per_Unit`, `Product_Price` |
| **Total Sales** | `Total_Sales`, `Sales`, `Revenue`, `Total_Amount`, `Amount`, `Net_Sales`, `Order_Value` |
| **Current Stock** | `Current_Stock`, `Stock`, `Inventory`, `Stock_Level`, `Quantity_On_Hand`, `Available_Stock` |

> **Tip:** If your CSV has missing columns (such as missing total sales or missing category), the Cleaning Agent automatically calculates `Total Sales = Quantity * Price` and infers categories using product keyword intelligence.

---

## 7. Project File Layout

```text
InsightForge/
├── app.py                      # Main Streamlit web application entry point
├── requirements.txt            # Python dependency requirements
├── test_pipeline.py            # Automated end-to-end integration test suite
├── README.md                   # System documentation & usage guide
│
├── agents/                     # Multi-Agent AI System
│   ├── base_agent.py           # Abstract base class & AgentResult dataclass
│   ├── ingestion.py            # Agent 1: Schema mapping & raw persistence
│   ├── cleaning.py             # Agent 2: Sanitization & data quality scoring
│   ├── analytics.py            # Agent 3: Business KPIs & Pareto analysis
│   ├── forecasting.py          # Agent 4: Prophet & XGBoost demand models
│   └── recommendation.py       # Agent 5: Replenishment & liquidation decisions
│
├── models/                     # Machine Learning Forecasting
│   ├── prophet_model.py        # Facebook Prophet time-series wrapper
│   └── xgboost_model.py        # XGBoost gradient-boosted regression wrapper
│
├── dashboard/                  # Streamlit User Interface
│   ├── dashboard.py            # Page layouts, sidebar routing & session management
│   ├── charts.py               # Interactive Plotly chart builders with INR (₹) formatting
│   └── styles.py               # SaaS CSS design system & SVG vector icons
│
├── utils/                      # Utilities & Reporting
│   ├── preprocess.py           # Semantic column detection & numeric sanitization
│   └── pdf_report.py           # ReportLab executive PDF generator with Unicode fonts
│
├── database/                   # SQLite Relational Persistence
│   ├── db_manager.py           # Schema setup, connection management & CRUD queries
│   └── retail.db               # Local SQLite database (created automatically)
│
├── datasets/                   # Benchmark Datasets
│   └── sample_retail_sales.csv # 17,400+ transaction multi-year retail dataset
└── reports/                    # Output directory for generated PDF reports
```

---

## 8. Academic Integrity & Transparency Guarantee

- **Zero Mock Metrics**: All model evaluation statistics (MAE, RMSE, MAPE, R²) are computed deterministically from out-of-sample hold-out validation.
- **Deterministic Logic**: Inventory recommendations are calculated directly from empirical daily sales velocity and supplier lead-time parameters.
- **Data Privacy**: All data processing and machine learning training happens entirely locally on your machine with zero external cloud dependencies.

---

## 9. License

This project is licensed under the **MIT License** — free to use, modify, and distribute for educational, commercial, and personal projects.
