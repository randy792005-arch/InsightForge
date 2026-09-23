"""
InsightForge Multi-Agent Pipeline Verification Suite
Runs automated end-to-end testing across all 5 specialized AI agents,
SQLite database persistence, Prophet/XGBoost models, and PDF report generation.
"""

import os
import sys

# Add InsightForge root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from agents.ingestion import DataIngestionAgent
from agents.cleaning import DataCleaningAgent
from agents.analytics import AnalyticsAgent
from agents.forecasting import ForecastingAgent
from agents.recommendation import RecommendationAgent
from utils.pdf_report import PDFReportGenerator


def run_test_suite():
    print("=" * 75)
    print("INSIGHTFORGE: MULTI-AGENT BI PIPELINE INTEGRATION TEST")
    print("=" * 75)

    # 1. Database Initialization
    db = DatabaseManager()
    assert os.path.exists(db.db_path), "Database creation failed"
    print(f"[?] Step 1: DatabaseManager operational at {db.db_path}")

    # 2. Data Ingestion Agent
    sample_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "sample_retail_sales.csv")
    ingest_agent = DataIngestionAgent(db_manager=db)
    ingest_res = ingest_agent.run(file_source=sample_csv, filename="sample_retail_sales.csv", session_id="test_session")
    assert ingest_res.is_success, f"Ingestion failed: {ingest_res.message}"
    print(f"[?] Step 2: Data Ingestion Agent -> {len(ingest_res.data['df_raw']):,} rows ingested in {ingest_res.execution_time_sec}s")

    # 3. Data Cleaning Agent
    clean_agent = DataCleaningAgent(db_manager=db)
    clean_res = clean_agent.run(
        df_raw=ingest_res.data["df_raw"],
        column_mapping=ingest_res.data["column_mapping"],
        session_id="test_session"
    )
    assert clean_res.is_success, f"Cleaning failed: {clean_res.message}"
    q = clean_res.data["quality_report"]
    print(f"[?] Step 3: Data Cleaning Agent -> {len(clean_res.data['df_clean']):,} clean rows | Quality Score: {q['overall_score']}% (Grade: {q['grade']})")

    # 4. Analytics Agent
    analytics_agent = AnalyticsAgent(db_manager=db)
    analytics_res = analytics_agent.run(df_clean=clean_res.data["df_clean"], session_id="test_session")
    assert analytics_res.is_success, f"Analytics failed: {analytics_res.message}"
    kpis = analytics_res.data["kpis"]
    print(f"[?] Step 4: Analytics Agent -> Total Revenue: ${kpis['total_revenue']:,.2f} | Units: {kpis['total_units_sold']:,.0f} | SKUs: {kpis['unique_products']}")

    # 5. Forecasting Agent (Prophet)
    fc_agent = ForecastingAgent(db_manager=db)
    fc_prophet_res = fc_agent.run(df_clean=clean_res.data["df_clean"], model_type="Prophet", forecast_horizon=30, session_id="test_session")
    assert fc_prophet_res.is_success, f"Prophet forecasting failed: {fc_prophet_res.message}"
    pm = fc_prophet_res.data["metrics"]
    print(f"[?] Step 5: Forecasting Agent (Prophet) -> MAE=${pm['mae']:.2f}, RMSE=${pm['rmse']:.2f}, MAPE={pm['mape']:.2f}%, R2={pm['r2']:.3f}")

    # 6. Forecasting Agent (XGBoost)
    fc_xgb_res = fc_agent.run(df_clean=clean_res.data["df_clean"], model_type="XGBoost", forecast_horizon=30, session_id="test_session")
    assert fc_xgb_res.is_success, f"XGBoost forecasting failed: {fc_xgb_res.message}"
    xm = fc_xgb_res.data["metrics"]
    print(f"[?] Step 6: Forecasting Agent (XGBoost) -> MAE=${xm['mae']:.2f}, RMSE=${xm['rmse']:.2f}, MAPE={xm['mape']:.2f}%, R2={xm['r2']:.3f}")

    # 7. Recommendation Agent
    rec_agent = RecommendationAgent(db_manager=db)
    rec_res = rec_agent.run(
        analytics_data=analytics_res.data,
        forecasting_data=fc_prophet_res.data,
        session_id="test_session"
    )
    assert rec_res.is_success, f"Recommendation failed: {rec_res.message}"
    print(f"[?] Step 7: Recommendation Agent -> {rec_res.data['total_recommendations']} Recommendations (Critical: {rec_res.data['critical_count']}, High: {rec_res.data['high_count']})")

    # 8. PDF Report Generator
    pdf_gen = PDFReportGenerator()
    pdf_path = pdf_gen.generate_report(
        dataset_info={"filename": "sample_retail_sales.csv"},
        kpis=analytics_res.data["kpis"],
        category_stats=analytics_res.data["category_stats"],
        top_products=analytics_res.data["top_10_products"],
        forecast_metrics=fc_prophet_res.data["metrics"],
        forecast_summary=fc_prophet_res.data,
        recommendations_df=rec_res.data["recommendations_df"],
        quality_report=clean_res.data["quality_report"]
    )
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000, "PDF export failed"
    print(f"[?] Step 8: PDF Report Generator -> {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

    print("=" * 75)
    print("STATUS: ALL TESTS PASSED (100% OPERATIONAL)")
    print("=" * 75)


if __name__ == "__main__":
    run_test_suite()
