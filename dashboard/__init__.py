"""
InsightForge Dashboard Package
"""
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

__all__ = [
    "create_revenue_trend_chart",
    "create_monthly_sales_chart",
    "create_dow_sales_chart",
    "create_category_pie_chart",
    "create_top_products_bar",
    "create_pareto_chart",
    "create_forecast_chart",
    "create_model_evaluation_chart",
    "create_inventory_matrix_chart",
    "inject_custom_css",
    "render_dashboard_header",
    "render_agent_stepper_ui",
    "render_kpi_cards_grid",
    "render_alert_banner"
]
