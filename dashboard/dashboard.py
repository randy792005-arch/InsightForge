"""
InsightForge Interactive Streamlit UI Dashboard
Provides an enterprise SaaS web interface demonstrating the 5-agent AI workflow,
interactive exploratory analytics, machine learning demand forecasting, inventory decision support,
and executive PDF report generation on any uploaded retail dataset.
"""

import os
import io
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from database.db_manager import DatabaseManager
from agents.ingestion import DataIngestionAgent
from agents.cleaning import DataCleaningAgent
from agents.analytics import AnalyticsAgent
from agents.forecasting import ForecastingAgent
from agents.recommendation import RecommendationAgent
from dashboard.charts import (
    create_revenue_trend_chart,
    create_monthly_sales_chart,
    create_dow_sales_chart,
    create_category_pie_chart,
    create_top_products_bar,
    create_pareto_chart,
    create_forecast_chart,
    create_model_evaluation_chart,
    create_inventory_matrix_chart
)
from dashboard.styles import (
    inject_custom_css,
    render_dashboard_header,
    render_agent_stepper_ui,
    render_kpi_cards_grid,
    render_alert_banner
)
from utils.pdf_report import PDFReportGenerator


def init_session_state():
    """Initialize Streamlit session state stores for multi-agent pipeline state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {
            "ingestion": {"status": "PENDING", "msg": "Waiting for dataset"},
            "cleaning": {"status": "PENDING", "msg": "Waiting for ingestion"},
            "analytics": {"status": "PENDING", "msg": "Waiting for cleaning"},
            "forecasting": {"status": "PENDING", "msg": "Waiting for analytics"},
            "recommendation": {"status": "PENDING", "msg": "Waiting for forecasting"}
        }
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    if "column_mapping" not in st.session_state:
        st.session_state.column_mapping = {}
    if "quality_report" not in st.session_state:
        st.session_state.quality_report = None
    if "analytics_data" not in st.session_state:
        st.session_state.analytics_data = None
    if "forecasting_data" not in st.session_state:
        st.session_state.forecasting_data = None
    if "recommendation_data" not in st.session_state:
        st.session_state.recommendation_data = None
    if "dataset_filename" not in st.session_state:
        st.session_state.dataset_filename = None
    if "pdf_report_path" not in st.session_state:
        st.session_state.pdf_report_path = None


def run_full_pipeline(file_source, filename: str, model_type: str = "Prophet", forecast_horizon: int = 30):
    """Coordinates the execution of all 5 specialized agents sequentially on the target dataset."""
    db = st.session_state.db
    # Generate fresh session ID for this execution run
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.session_id = session_id
    st.session_state.pdf_report_path = None

    # Reset seek pointer if file stream
    if hasattr(file_source, "seek"):
        try:
            file_source.seek(0)
        except Exception:
            pass

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Reset statuses
    st.session_state.agent_status = {
        "ingestion": {"status": "PENDING", "msg": "Validating schema & parsing raw records"},
        "cleaning": {"status": "PENDING", "msg": "Waiting for ingestion"},
        "analytics": {"status": "PENDING", "msg": "Waiting for cleaning"},
        "forecasting": {"status": "PENDING", "msg": "Waiting for analytics"},
        "recommendation": {"status": "PENDING", "msg": "Waiting for forecasting"}
    }

    # Step 1: Ingestion Agent
    status_text.markdown(f"**[Agent 1/5] Data Ingestion Agent** validating schema and parsing `{filename}`...")
    ingestion_agent = DataIngestionAgent(db_manager=db)
    ingestion_res = ingestion_agent.run(file_source=file_source, filename=filename, session_id=session_id)
    progress_bar.progress(20)

    if not ingestion_res.is_success:
        st.session_state.agent_status["ingestion"] = {"status": "ERROR", "msg": ingestion_res.message}
        status_text.error(f"Data Ingestion Agent Failed: {ingestion_res.message}")
        return False

    st.session_state.agent_status["ingestion"] = {"status": "SUCCESS", "msg": ingestion_res.message}
    st.session_state.df_raw = ingestion_res.data["df_raw"]
    st.session_state.column_mapping = ingestion_res.data["column_mapping"]
    st.session_state.dataset_filename = filename

    # Step 2: Cleaning Agent
    status_text.markdown("**[Agent 2/5] Data Cleaning Agent** deduplicating, standardizing and scoring data quality...")
    cleaning_agent = DataCleaningAgent(db_manager=db)
    cleaning_res = cleaning_agent.run(
        df_raw=st.session_state.df_raw,
        column_mapping=st.session_state.column_mapping,
        session_id=session_id
    )
    progress_bar.progress(40)

    if not cleaning_res.is_success:
        st.session_state.agent_status["cleaning"] = {"status": "ERROR", "msg": cleaning_res.message}
        status_text.error(f"Data Cleaning Agent Failed: {cleaning_res.message}")
        return False

    st.session_state.agent_status["cleaning"] = {"status": "SUCCESS", "msg": cleaning_res.message}
    st.session_state.df_clean = cleaning_res.data["df_clean"]
    st.session_state.quality_report = cleaning_res.data["quality_report"]

    # Step 3: Analytics Agent
    status_text.markdown("**[Agent 3/5] Analytics Agent** computing business KPIs, sales trends and category distributions...")
    analytics_agent = AnalyticsAgent(db_manager=db)
    analytics_res = analytics_agent.run(df_clean=st.session_state.df_clean, session_id=session_id)
    progress_bar.progress(60)

    if not analytics_res.is_success:
        st.session_state.agent_status["analytics"] = {"status": "ERROR", "msg": analytics_res.message}
        status_text.error(f"Analytics Agent Failed: {analytics_res.message}")
        return False

    st.session_state.agent_status["analytics"] = {"status": "SUCCESS", "msg": analytics_res.message}
    st.session_state.analytics_data = analytics_res.data

    # Step 4: Forecasting Agent
    status_text.markdown(f"**[Agent 4/5] Forecasting Agent** training {model_type} demand forecast model...")
    forecasting_agent = ForecastingAgent(db_manager=db)
    forecasting_res = forecasting_agent.run(
        df_clean=st.session_state.df_clean,
        model_type=model_type,
        forecast_horizon=forecast_horizon,
        session_id=session_id
    )
    progress_bar.progress(80)

    if not forecasting_res.is_success:
        st.session_state.agent_status["forecasting"] = {"status": "ERROR", "msg": forecasting_res.message}
        status_text.error(f"Forecasting Agent Failed: {forecasting_res.message}")
        return False

    st.session_state.agent_status["forecasting"] = {"status": "SUCCESS", "msg": forecasting_res.message}
    st.session_state.forecasting_data = forecasting_res.data

    # Step 5: Recommendation Agent
    status_text.markdown("**[Agent 5/5] Recommendation Agent** evaluating inventory velocity & formulating prescriptive actions...")
    rec_agent = RecommendationAgent(db_manager=db)
    rec_res = rec_agent.run(
        analytics_data=st.session_state.analytics_data,
        forecasting_data=st.session_state.forecasting_data,
        session_id=session_id
    )
    progress_bar.progress(100)

    if not rec_res.is_success:
        st.session_state.agent_status["recommendation"] = {"status": "ERROR", "msg": rec_res.message}
        status_text.error(f"Recommendation Agent Failed: {rec_res.message}")
        return False

    st.session_state.agent_status["recommendation"] = {"status": "SUCCESS", "msg": rec_res.message}
    st.session_state.recommendation_data = rec_res.data

    status_text.success(f"Multi-Agent AI Pipeline Completed Successfully for '{filename}'! All 5 agents executed.")
    time.sleep(0.3)
    return True


# ---------------- PAGE 1: EXECUTIVE BI DASHBOARD ---------------- #
def page_executive_dashboard():
    render_dashboard_header(
        title="InsightForge Executive BI Dashboard",
        subtitle="Autonomous Multi-Agent Retail Business Intelligence and Decision Support System",
        badge_text="InsightForge Enterprise"
    )

    analytics_data = st.session_state.analytics_data
    if analytics_data is None:
        st.info("Welcome to InsightForge. To begin, upload your own retail dataset or load the benchmark dataset.")
        
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            st.markdown("### Upload Your Dataset")
            st.caption("Supports CSV, XLSX with standard retail columns (Date, Sales/Qty, Products).")
            exec_file = st.file_uploader("Choose file (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"], key="exec_upload")
            if exec_file is not None:
                if st.button(f"Process '{exec_file.name}'", type="primary", key="btn_exec_proc"):
                    with st.spinner("Executing 5-Agent AI Pipeline on your file..."):
                        run_full_pipeline(exec_file, filename=exec_file.name)
                        st.rerun()

        with c_up2:
            st.markdown("### Or Load Benchmark Retail Dataset")
            st.caption("17,486 multi-category enterprise retail sales transactions (2024-2025).")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load 17,400+ Sample Retail Dataset", type="secondary", key="btn_exec_sample"):
                sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "sample_retail_sales.csv")
                if os.path.exists(sample_path):
                    with st.spinner("Executing 5-Agent AI Pipeline on sample dataset..."):
                        run_full_pipeline(sample_path, filename="sample_retail_sales.csv")
                        st.rerun()
                else:
                    st.error("Sample dataset file not found.")
        return

    # 1. 5-Agent Status Stepper
    render_agent_stepper_ui(st.session_state.agent_status)

    # Active Dataset Status Bar
    kpis = analytics_data["kpis"]
    fname = st.session_state.get("dataset_filename", "Custom Dataset")
    st.markdown(
        f"""<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: #475569;">
        <div><b>Active Dataset:</b> <code style="color: #2563EB;">{fname}</code> | <b>{kpis.get('total_transactions', 0):,}</b> Transactions | <b>{kpis.get('unique_products', 0)}</b> Products | <b>{kpis.get('unique_categories', 0)}</b> Categories | Scope: <b>{kpis.get('start_date', '')}</b> to <b>{kpis.get('end_date', '')}</b></div>
        <div><span style="color: #059669; font-weight: 600;">All 5 Agents Active</span></div>
        </div>""",
        unsafe_allow_html=True
    )

    # 2. Top-Level Enterprise KPI Cards
    render_kpi_cards_grid(kpis)

    # 3. Urgent Stockout Alerts Banner
    rec_data = st.session_state.recommendation_data
    if rec_data and rec_data.get("critical_count", 0) > 0:
        crit_count = rec_data["critical_count"]
        render_alert_banner(critical_count=crit_count)

    # 4. Primary Visualizations Grid
    col_left, col_right = st.columns([1.55, 1.0])

    with col_left:
        st.plotly_chart(create_revenue_trend_chart(analytics_data["daily_sales"]), use_container_width=True)

    with col_right:
        st.plotly_chart(create_category_pie_chart(analytics_data["category_stats"]), use_container_width=True)

    col_b1, col_b2 = st.columns([1.1, 1.0])
    with col_b1:
        st.plotly_chart(create_top_products_bar(analytics_data["top_10_products"], top_n=8), use_container_width=True)

    with col_b2:
        fc_data = st.session_state.forecasting_data
        if fc_data and "forecast_df" in fc_data:
            st.plotly_chart(create_forecast_chart(fc_data["forecast_df"], model_name=fc_data["model_name"]), use_container_width=True)
        else:
            st.plotly_chart(create_monthly_sales_chart(analytics_data["monthly_sales"]), use_container_width=True)


# ---------------- PAGE 2: DATA INGESTION ---------------- #
def page_data_ingestion():
    render_dashboard_header(
        title="Data Ingestion Agent (Agent 1)",
        subtitle="Automated file parsing, semantic column detection, and SQLite raw persistence",
        badge_text="Agent 1: Ingestion"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Retail Dataset (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            help="Supports standard retail columns: Date, Product, Category, Quantity, Price, Sales, Stock"
        )

    with col2:
        st.markdown("**Quick Benchmark Test:**")
        if st.button("Load 17,400+ Sample Retail Dataset", type="secondary", key="btn_ingest_sample"):
            sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "sample_retail_sales.csv")
            if os.path.exists(sample_path):
                with st.spinner("Ingesting sample dataset..."):
                    run_full_pipeline(sample_path, filename="sample_retail_sales.csv")
                    st.success("Sample dataset loaded and processed through multi-agent pipeline!")
                    st.rerun()

    # If file uploaded
    if uploaded_file is not None:
        st.write(f"**Selected File:** `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")
        if st.button(f"Execute 5-Agent Pipeline on '{uploaded_file.name}'", type="primary", key="btn_ingest_proc"):
            with st.spinner(f"Running Multi-Agent Pipeline on '{uploaded_file.name}'..."):
                success = run_full_pipeline(uploaded_file, filename=uploaded_file.name)
                if success:
                    st.success(f"Pipeline finished! Click 'Executive Dashboard' in navigation to view full analysis.")
                    st.rerun()

    st.markdown("---")

    # Ingestion Diagnostics Preview
    if st.session_state.df_raw is not None:
        st.subheader(f"Raw Ingestion Diagnostics: {st.session_state.get('dataset_filename', '')}")
        df_raw = st.session_state.df_raw
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", f"{len(df_raw):,}")
        c2.metric("Raw Columns", f"{len(df_raw.columns)}")
        c3.metric("Mapped Concepts", f"{len(st.session_state.column_mapping)}")
        c4.metric("Memory Footprint", f"{round(df_raw.memory_usage(deep=True).sum()/1024, 1)} KB")

        st.markdown("##### Detected Semantic Column Mapping:")
        mapping_df = pd.DataFrame([
            {"Standard Retail Concept": k.replace("_", " ").title(), "Matched Raw Column": v}
            for k, v in st.session_state.column_mapping.items()
        ])
        st.dataframe(mapping_df, use_container_width=True)

        st.markdown("##### Raw Dataset Preview (First 15 Records):")
        st.dataframe(df_raw.head(15), use_container_width=True)


# ---------------- PAGE 3: DATA CLEANING ---------------- #
def page_data_cleaning():
    render_dashboard_header(
        title="Data Cleaning Agent (Agent 2)",
        subtitle="Deduplication, missing value imputation, type normalization, and comprehensive data health scoring",
        badge_text="Agent 2: Cleaning"
    )

    if st.session_state.df_clean is None:
        st.info("Please ingest a dataset first to review data cleaning results.")
        return

    quality = st.session_state.quality_report

    # Health Score Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Quality Score", f"{quality['overall_score']}%", delta=f"Grade {quality['grade']}")
    c2.metric("Clean Records Retained", f"{quality['clean_rows']:,}", delta=f"{quality['raw_rows'] - quality['clean_rows']:,} dropped")
    c3.metric("Duplicates Removed", f"{quality['duplicates_removed']:,}")
    c4.metric("Imputed Data Cells", f"{quality['missing_cells_imputed']:,}")

    st.markdown("---")

    # Quality Pillars Grid
    st.subheader("Data Health Pillars Assessment")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.progress(quality["completeness_score"] / 30.0, text=f"Completeness: {quality['completeness_score']}/30")
    q_col2.progress(quality["uniqueness_score"] / 25.0, text=f"Uniqueness: {quality['uniqueness_score']}/25")
    q_col3.progress(quality["validity_score"] / 25.0, text=f"Validity: {quality['validity_score']}/25")
    q_col4.progress(quality["date_score"] / 20.0, text=f"Continuity: {quality['date_score']}/20")

    st.markdown("---")

    # Cleaned Dataset Sample
    st.subheader("Normalized Cleaned Dataset Preview")
    st.dataframe(st.session_state.df_clean.head(25), use_container_width=True)


# ---------------- PAGE 4: SALES ANALYTICS ---------------- #
def page_sales_analytics():
    render_dashboard_header(
        title="Analytics Agent (Agent 3)",
        subtitle="Exploratory data analysis, multi-dimensional sales breakdown, category revenue shares, and Pareto distribution",
        badge_text="Agent 3: Analytics"
    )

    analytics_data = st.session_state.analytics_data
    if analytics_data is None:
        st.info("Please ingest and clean a dataset to view analytics.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Sales Trends", 
        "Category Performance", 
        "Product Rankings", 
        "Pareto (80/20) Analysis"
    ])

    with tab1:
        st.plotly_chart(create_revenue_trend_chart(analytics_data["daily_sales"]), use_container_width=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(create_monthly_sales_chart(analytics_data["monthly_sales"]), use_container_width=True)
        with col_m2:
            st.plotly_chart(create_dow_sales_chart(analytics_data["dow_sales"]), use_container_width=True)

    with tab2:
        col_c1, col_c2 = st.columns([1, 1.2])
        with col_c1:
            st.plotly_chart(create_category_pie_chart(analytics_data["category_stats"]), use_container_width=True)
        with col_c2:
            st.markdown("##### Category Performance Breakdown:")
            cat_table = analytics_data["category_stats"].copy()
            cat_table["total_sales"] = cat_table["total_sales"].apply(lambda v: f"₹{v:,.2f}")
            cat_table["quantity_sold"] = cat_table["quantity_sold"].apply(lambda v: f"{v:,.0f}")
            cat_table["revenue_share_pct"] = cat_table["revenue_share_pct"].apply(lambda v: f"{v:.1f}%")
            cat_table["avg_order_value"] = cat_table["avg_order_value"].apply(lambda v: f"₹{v:,.2f}")
            st.dataframe(cat_table, use_container_width=True)

    with tab3:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(create_top_products_bar(analytics_data["top_10_products"], top_n=10), use_container_width=True)
        with col_p2:
            st.markdown("##### Bottom 10 Slow-Moving Products:")
            bot_df = analytics_data["bottom_10_products"].copy()
            bot_df["total_sales"] = bot_df["total_sales"].apply(lambda v: f"₹{v:,.2f}")
            bot_df["quantity_sold"] = bot_df["quantity_sold"].apply(lambda v: f"{v:,.0f}")
            st.dataframe(bot_df[["product_name", "category", "total_sales", "quantity_sold", "current_stock"]], use_container_width=True)

    with tab4:
        st.plotly_chart(create_pareto_chart(analytics_data["pareto_df"]), use_container_width=True)
        kpis = analytics_data["kpis"]
        p_count = kpis.get("pareto_80_product_count", 0)
        p_pct = kpis.get("pareto_80_product_pct", 0.0)
        st.info(f"Pareto Insight: Just {p_count} products ({p_pct}% of catalog) generate 80% of total business revenue.")


# ---------------- PAGE 5: DEMAND FORECASTING ---------------- #
def page_demand_forecasting():
    render_dashboard_header(
        title="Forecasting Agent (Agent 4)",
        subtitle="Machine learning & time-series sales demand forecasting with hold-out validation metrics (MAE, RMSE, MAPE, R2)",
        badge_text="Agent 4: Forecasting"
    )

    df_clean = st.session_state.df_clean
    if df_clean is None:
        st.info("Please ingest and clean a dataset to run forecasting.")
        return

    # Interactive Model Controls Card
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1.2, 1.2, 1])

    with col_cfg1:
        model_choice = st.selectbox(
            "Select Forecasting Model:",
            ["Prophet (Additive Trend + Seasonality)", "XGBoost (Lag Features + Gradient Boosting)", "Both (Comparative Ensemble)"],
            index=0
        )

    with col_cfg2:
        horizon_choice = st.slider("Forecast Horizon (Days):", min_value=7, max_value=90, value=30, step=7)

    with col_cfg3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_fc_btn = st.button("Retrain Forecast", type="primary")

    if run_fc_btn:
        model_name_param = "Prophet" if "Prophet" in model_choice else "XGBoost" if "XGBoost" in model_choice else "Both"
        with st.spinner(f"Training {model_name_param} forecasting model..."):
            agent = ForecastingAgent(db_manager=st.session_state.db)
            res = agent.run(
                df_clean=df_clean,
                model_type=model_name_param,
                forecast_horizon=horizon_choice,
                session_id=st.session_state.session_id
            )
            if res.is_success:
                st.session_state.forecasting_data = res.data
                rec_agent = RecommendationAgent(db_manager=st.session_state.db)
                rec_res = rec_agent.run(
                    analytics_data=st.session_state.analytics_data,
                    forecasting_data=res.data,
                    session_id=st.session_state.session_id
                )
                st.session_state.recommendation_data = rec_res.data
                st.success("Forecasting model trained successfully!")
                st.rerun()

    fc_data = st.session_state.forecasting_data
    if fc_data is None:
        st.info("Forecasting has not been executed yet. Click 'Retrain Forecast' above.")
        return

    metrics = fc_data["metrics"]

    # Authentic Evaluation Metrics Grid
    st.markdown("#### Authentic Model Evaluation Metrics (Hold-out Backtesting)")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Model Used", fc_data["model_name"])
    m_col2.metric("Mean Absolute Error (MAE)", f"₹{metrics['mae']:,.2f}")
    m_col3.metric("Root Mean Squared Error (RMSE)", f"₹{metrics['rmse']:,.2f}")
    m_col4.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['mape']:.2f}%")
    m_col5.metric("R2 Score", f"{metrics['r2']:.3f}")

    st.markdown("---")

    # Interactive Forecast Visualization
    st.plotly_chart(create_forecast_chart(fc_data["forecast_df"], model_name=fc_data["model_name"]), use_container_width=True)

    # Backtest & Future Projection Grid
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        if "backtest_df" in fc_data and fc_data["backtest_df"] is not None:
            st.plotly_chart(create_model_evaluation_chart(fc_data["backtest_df"], model_name=fc_data["model_name"]), use_container_width=True)

    with col_f2:
        st.markdown(f"##### Next {metrics.get('forecast_horizon', 30)}-Day Projected Revenue")
        proj_rev = fc_data.get("projected_total_revenue", 0.0)
        proj_daily = fc_data.get("projected_avg_daily", 0.0)

        st.metric("Total Projected Sales", f"₹{proj_rev:,.2f}", delta=f"~₹{proj_daily:,.2f}/day")

        future_table = fc_data["forecast_df"][fc_data["forecast_df"]["is_historical"] == 0].copy()
        future_table["date"] = pd.to_datetime(future_table["date"]).dt.strftime("%Y-%m-%d")
        disp_future = future_table.copy()
        disp_future["predicted_sales"] = disp_future["predicted_sales"].apply(lambda v: f"₹{v:,.2f}")
        disp_future["lower_bound"] = disp_future["lower_bound"].apply(lambda v: f"₹{v:,.2f}")
        disp_future["upper_bound"] = disp_future["upper_bound"].apply(lambda v: f"₹{v:,.2f}")
        st.dataframe(disp_future[["date", "predicted_sales", "lower_bound", "upper_bound"]].head(15), use_container_width=True)


# ---------------- PAGE 6: INVENTORY & RECOMMENDATIONS ---------------- #
def page_recommendations():
    render_dashboard_header(
        title="Recommendation Agent (Agent 5)",
        subtitle="Prescriptive decision support: replenishment purchase orders, safety stock alerts, and liquidation actions",
        badge_text="Agent 5: Decision Support"
    )

    analytics_data = st.session_state.analytics_data
    rec_data = st.session_state.recommendation_data

    if analytics_data is None or rec_data is None:
        st.info("Please process a dataset to generate recommendations.")
        return

    # Recommendation Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Action Recommendations", rec_data.get("total_recommendations", 0))
    c2.metric("Critical Urgency", rec_data.get("critical_count", 0), delta="Immediate Action", delta_color="inverse")
    c3.metric("High Priority", rec_data.get("high_count", 0))
    c4.metric("Medium / Liquidation", rec_data.get("medium_count", 0))

    st.markdown("---")

    # Inventory Health Matrix Chart
    if "product_inventory" in analytics_data:
        st.plotly_chart(create_inventory_matrix_chart(analytics_data["product_inventory"]), use_container_width=True)

    # Actionable Recommendations Table with Filter
    rec_df = rec_data.get("recommendations_df", pd.DataFrame())
    if not rec_df.empty:
        st.subheader("Actionable Decision Support Table")

        priority_filter = st.selectbox("Filter by Priority Level:", ["All Priorities", "CRITICAL", "HIGH", "MEDIUM"], index=0)
        
        filtered_df = rec_df.copy()
        if priority_filter != "All Priorities":
            filtered_df = filtered_df[filtered_df["priority"] == priority_filter]

        st.dataframe(
            filtered_df[[
                "priority", "product_name", "category", "issue_type",
                "trigger_metric", "action_recommended", "expected_impact", "reorder_qty"
            ]],
            use_container_width=True
        )


# ---------------- PAGE 7: EXECUTIVE PDF REPORT ---------------- #
def page_pdf_report():
    render_dashboard_header(
        title="Executive PDF Report Generator",
        subtitle="Export publication-ready retail BI executive briefs with KPIs, category shares, ML forecast metrics, and decision tables",
        badge_text="Executive PDF Export"
    )

    analytics_data = st.session_state.analytics_data
    forecasting_data = st.session_state.forecasting_data
    recommendation_data = st.session_state.recommendation_data

    if analytics_data is None or forecasting_data is None or recommendation_data is None:
        st.info("Please run the full multi-agent pipeline before generating a report.")
        return

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        if st.button("Generate Executive PDF Report", type="primary"):
            with st.spinner("Compiling intelligence & building ReportLab PDF..."):
                pdf_gen = PDFReportGenerator()
                pdf_path = pdf_gen.generate_report(
                    dataset_info={"filename": st.session_state.dataset_filename},
                    kpis=analytics_data["kpis"],
                    category_stats=analytics_data["category_stats"],
                    top_products=analytics_data["top_10_products"],
                    forecast_metrics=forecasting_data["metrics"],
                    forecast_summary=forecasting_data,
                    recommendations_df=recommendation_data["recommendations_df"],
                    quality_report=st.session_state.quality_report
                )
                st.session_state.pdf_report_path = pdf_path
                st.success("PDF Report generated successfully!")

    if st.session_state.pdf_report_path and os.path.exists(st.session_state.pdf_report_path):
        with open(st.session_state.pdf_report_path, "rb") as f:
            pdf_bytes = f.read()
        
        filename = os.path.basename(st.session_state.pdf_report_path)
        st.download_button(
            label=f"Download {filename} ({round(len(pdf_bytes)/1024, 1)} KB)",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            type="secondary"
        )

    st.markdown("---")
    st.subheader("Report Content Preview")
    st.markdown(f"""
    - **Dataset Scope:** `{st.session_state.dataset_filename}` ({analytics_data['kpis'].get('total_transactions', 0):,} Transactions)
    - **Total Revenue Analyzed:** `₹{analytics_data['kpis'].get('total_revenue', 0):,.2f}`
    - **Forecasting Model:** `{forecasting_data.get('model_name', 'Prophet')}` (MAE: `₹{forecasting_data['metrics'].get('mae', 0):,.2f}`, MAPE: `{forecasting_data['metrics'].get('mape', 0):.2f}%`)
    - **Inventory Action Items:** `{recommendation_data.get('total_recommendations', 0)}` Prioritized Recommendations
    """)


def main_dashboard_app():
    """Main navigation controller for Streamlit dashboard."""
    init_session_state()
    inject_custom_css()

    # Redesigned Enterprise Sidebar with clean SVG logo rendered via st.sidebar.html
    st.sidebar.html("""<div class="sidebar-brand-box">
<div style="background: rgba(37, 99, 235, 0.2); border: 1px solid rgba(96, 165, 250, 0.4); border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;">
<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMiIgaGVpZ2h0PSIyMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM2MEE1RkEiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik02IDJMMyA2djE0YTIgMiAwIDAgMCAyIDJoMTRhMiAyIDAgMCAwIDItMlY2bC0zLTR6Ii8+PGxpbmUgeDE9IjMiIHkxPSI2IiB4Mj0iMjEiIHkyPSI2Ii8+PHBhdGggZD0iTTE2IDEwYTQgNCAwIDAgMS04IDAiLz48L3N2Zz4=" width="20" height="20" alt="Brand Logo" />
</div>
<div>
<div class="sidebar-brand-title">InsightForge</div>
<div class="sidebar-brand-sub">Enterprise Retail BI</div>
</div>
</div>""")

    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "Executive Dashboard",
            "Data Ingestion",
            "Data Cleaning",
            "Sales Analytics",
            "Demand Forecasting",
            "Inventory & Recommendations",
            "Executive PDF Report"
        ]
    )

    # Sidebar Dataset Status & Quick Uploader
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Active Retail Dataset")
    if st.session_state.df_clean is not None:
        fname = st.session_state.get("dataset_filename", "Custom Dataset")
        n_rows = len(st.session_state.df_clean)
        st.sidebar.caption(f"**{fname}** ({n_rows:,} clean records)")
    else:
        st.sidebar.caption("No dataset loaded yet.")

    sidebar_file = st.sidebar.file_uploader(
        "Quick Upload Dataset",
        type=["csv", "xlsx", "xls"],
        key="sidebar_quick_upload",
        help="Upload CSV or Excel to execute multi-agent analysis"
    )
    if sidebar_file is not None:
        if st.sidebar.button(f"Process '{sidebar_file.name}'", type="primary", key="btn_side_proc"):
            with st.spinner(f"Running pipeline on '{sidebar_file.name}'..."):
                run_full_pipeline(sidebar_file, filename=sidebar_file.name)
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Agentic AI Pipeline Status")
    statuses = st.session_state.agent_status
    status_htmls = []
    for name, key in [
        ("Ingestion Agent", "ingestion"),
        ("Cleaning Agent", "cleaning"),
        ("Analytics Agent", "analytics"),
        ("Forecasting Agent", "forecasting"),
        ("Recommendation", "recommendation")
    ]:
        s = statuses[key]["status"]
        dot_class = "dot-success" if s in ["SUCCESS", "WARNING"] else "dot-error" if s == "ERROR" else "dot-pending"
        status_htmls.append(f'<div class="sidebar-status-item"><span><span class="{dot_class}"></span><b>{name}</b></span><span><code>{s}</code></span></div>')

    st.sidebar.html("".join(status_htmls))

    st.sidebar.markdown("---")
    st.sidebar.caption("InsightForge")

    # Page Routing
    if page == "Executive Dashboard":
        page_executive_dashboard()
    elif page == "Data Ingestion":
        page_data_ingestion()
    elif page == "Data Cleaning":
        page_data_cleaning()
    elif page == "Sales Analytics":
        page_sales_analytics()
    elif page == "Demand Forecasting":
        page_demand_forecasting()
    elif page == "Inventory & Recommendations":
        page_recommendations()
    elif page == "Executive PDF Report":
        page_pdf_report()
