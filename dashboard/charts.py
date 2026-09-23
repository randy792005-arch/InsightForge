"""
InsightForge Interactive Chart Engine
Generates modern, publication-grade Plotly interactive visualizations for retail sales,
revenue time series, category breakdown, Pareto distributions, ML demand forecasts, and inventory health.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# Enterprise Color Palette
COLOR_PRIMARY = "#2563EB"       # Royal Blue
COLOR_SECONDARY = "#0D9488"     # Teal
COLOR_ACCENT = "#F59E0B"        # Amber
COLOR_DANGER = "#EF4444"        # Rose / Red
COLOR_SUCCESS = "#10B981"       # Emerald Green
COLOR_DARK = "#0F172A"          # Dark Slate
COLOR_GRID = "#F1F5F9"          # Light Slate Grid
PALETTE = ["#2563EB", "#0D9488", "#6366F1", "#F59E0B", "#EC4899", "#06B6D4", "#10B981", "#64748B"]

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def apply_enterprise_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "", height: int = 380) -> go.Figure:
    """Applies cohesive enterprise SaaS layout typography, spacing, and styling."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT_FAMILY, size=15, color="#0F172A"),
            x=0.01,
            y=0.96
        ),
        font=dict(family=FONT_FAMILY, size=12, color="#475569"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=25, t=55, b=35),
        height=height,
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#FFFFFF"
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(family=FONT_FAMILY, size=12, color="#64748B")) if xaxis_title else None,
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(family=FONT_FAMILY, size=12, color="#64748B")) if yaxis_title else None,
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0"
        )
    )
    return fig


def create_revenue_trend_chart(df_daily: pd.DataFrame) -> go.Figure:
    """Generates dual-axis daily revenue bar & 7-day rolling average chart."""
    fig = go.Figure()

    # Daily actual sales bars
    fig.add_trace(go.Bar(
        x=df_daily["date"],
        y=df_daily["total_sales"],
        name="Daily Revenue ($)",
        marker_color="rgba(37, 99, 235, 0.28)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Daily Sales: $%{y:,.2f}<extra></extra>"
    ))

    # 7-day moving average curve
    if "sales_7d_ma" in df_daily.columns:
        fig.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily["sales_7d_ma"],
            name="7-Day Moving Avg",
            mode="lines",
            line=dict(color="#1D4ED8", width=2.8),
            hovertemplate="<b>7D Avg:</b> $%{y:,.2f}<extra></extra>"
        ))

    apply_enterprise_layout(fig, title="Daily Revenue Trend & 7-Day Moving Average", yaxis_title="Revenue ($)", height=370)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_monthly_sales_chart(df_monthly: pd.DataFrame) -> go.Figure:
    """Generates monthly revenue bars with clean currency formatting."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_monthly["month_label"],
        y=df_monthly["total_sales"],
        name="Monthly Revenue",
        marker=dict(
            color=df_monthly["total_sales"],
            colorscale=[[0, "#93C5FD"], [1, "#1E40AF"]],
            showscale=False
        ),
        text=df_monthly["total_sales"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
    ))

    apply_enterprise_layout(fig, title="Monthly Revenue Performance", yaxis_title="Revenue ($)", height=340)
    return fig


def create_dow_sales_chart(dow_df: pd.DataFrame) -> go.Figure:
    """Generates Day of Week sales distribution bar chart."""
    fig = px.bar(
        dow_df,
        x="day_name",
        y="total_sales",
        text="share_pct",
        color="total_sales",
        color_continuous_scale=[[0, "#BAE6FD"], [1, "#0284C7"]]
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.2f}<br>Share: %{text:.1f}%<extra></extra>"
    )
    apply_enterprise_layout(fig, title="Sales Distribution by Day of Week", yaxis_title="Revenue ($)", height=340)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def create_category_pie_chart(df_category: pd.DataFrame) -> go.Figure:
    """Generates interactive category revenue contribution donut chart."""
    fig = px.pie(
        df_category,
        names="category",
        values="total_sales",
        hole=0.55,
        color_discrete_sequence=PALETTE
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    apply_enterprise_layout(fig, title="Revenue Share by Product Category", height=370)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=11))
    )
    return fig


def create_top_products_bar(df_products: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Generates horizontal bar chart for top-selling products."""
    top_df = df_products.head(top_n).sort_values(by="total_sales", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["total_sales"],
        y=top_df["product_name"],
        orientation="h",
        marker=dict(
            color=top_df["total_sales"],
            colorscale=[[0, "#99F6E4"], [1, "#0D9488"]],
            showscale=False
        ),
        text=top_df["total_sales"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Sales: $%{x:,.2f}<br>Units: %{customdata[0]:,}<extra></extra>",
        customdata=top_df[["quantity_sold"]]
    ))

    apply_enterprise_layout(fig, title=f"Top {top_n} Best-Selling Products by Revenue", xaxis_title="Revenue ($)", height=370)
    fig.update_layout(margin=dict(l=150, r=40, t=55, b=35))
    return fig


def create_pareto_chart(df_pareto: pd.DataFrame) -> go.Figure:
    """Generates 80/20 Pareto cumulative revenue curve."""
    top_pareto = df_pareto.head(20).copy()

    fig = go.Figure()

    # Product revenue bars
    fig.add_trace(go.Bar(
        x=top_pareto["product_name"],
        y=top_pareto["total_sales"],
        name="Product Sales ($)",
        marker_color="#3B82F6",
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.2f}<extra></extra>"
    ))

    # Cumulative % line
    fig.add_trace(go.Scatter(
        x=top_pareto["product_name"],
        y=top_pareto["cumulative_share_pct"],
        name="Cumulative Share %",
        mode="lines+markers",
        line=dict(color="#EF4444", width=2.8),
        marker=dict(size=6, color="#EF4444"),
        yaxis="y2",
        hovertemplate="<b>Cumulative:</b> %{y:.1f}%<extra></extra>"
    ))

    # 80% Reference line
    fig.add_hline(y=80, line_dash="dash", line_color="#DC2626", annotation_text="80% Threshold", yref="y2")

    apply_enterprise_layout(fig, title="Pareto (80/20) Product Revenue Distribution", height=400)
    fig.update_layout(
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(title="Product Sales ($)", side="left"),
        yaxis2=dict(title="Cumulative Share (%)", side="right", overlaying="y", range=[0, 105], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=90)
    )
    return fig


def create_forecast_chart(forecast_df: pd.DataFrame, model_name: str = "Prophet") -> go.Figure:
    """Generates interactive historical actuals + future forecast with uncertainty envelope."""
    fig = go.Figure()

    hist_df = forecast_df[forecast_df["is_historical"] == 1].copy()
    future_df = forecast_df[forecast_df["is_historical"] == 0].copy()

    # 1. Historical Actual Sales
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["actual_sales"],
            name="Historical Sales",
            mode="lines",
            line=dict(color="#2563EB", width=2),
            hovertemplate="<b>Historical:</b> %{x|%b %d, %Y}<br>Sales: $%{y:,.2f}<extra></extra>"
        ))

    # 2. Confidence Interval Ribbon
    if not future_df.empty and "lower_bound" in future_df.columns and "upper_bound" in future_df.columns:
        fig.add_trace(go.Scatter(
            x=list(future_df["date"]) + list(future_df["date"])[::-1],
            y=list(future_df["upper_bound"]) + list(future_df["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Confidence Interval",
            hoverinfo="skip"
        ))

    # 3. Predicted Future Sales
    if not future_df.empty:
        fig.add_trace(go.Scatter(
            x=future_df["date"],
            y=future_df["predicted_sales"],
            name=f"Forecast ({model_name})",
            mode="lines+markers",
            line=dict(color="#D97706", width=2.8),
            marker=dict(size=4),
            hovertemplate="<b>Forecast:</b> %{x|%b %d, %Y}<br>Predicted: $%{y:,.2f}<extra></extra>"
        ))

    apply_enterprise_layout(fig, title=f"Demand Forecast ({model_name}) with 90% Confidence Band", yaxis_title="Sales ($)", height=400)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_model_evaluation_chart(backtest_df: pd.DataFrame, model_name: str = "Prophet") -> go.Figure:
    """Generates hold-out backtest comparison of Actual vs Predicted sales."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=backtest_df["date"],
        y=backtest_df["actual"],
        name="Actual Sales",
        mode="lines+markers",
        line=dict(color="#0F172A", width=2),
        marker=dict(size=5),
        hovertemplate="<b>Date:</b> %{x|%b %d}<br>Actual: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=backtest_df["date"],
        y=backtest_df["predicted"],
        name="Model Prediction",
        mode="lines+markers",
        line=dict(color="#059669", width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="<b>Predicted:</b> $%{y:,.2f}<extra></extra>"
    ))

    apply_enterprise_layout(fig, title=f"Hold-Out Backtesting Validation ({model_name})", yaxis_title="Sales ($)", height=340)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_inventory_matrix_chart(df_inventory: pd.DataFrame) -> go.Figure:
    """Generates Stock vs Velocity matrix scatter plot."""
    status_colors = {
        "Critical Stock": "#DC2626",
        "Low Stock": "#F59E0B",
        "Optimal": "#10B981",
        "Overstocked": "#6366F1",
        "Out of Stock": "#991B1B"
    }

    fig = px.scatter(
        df_inventory,
        x="daily_velocity",
        y="current_stock",
        size="quantity_sold",
        color="stock_status",
        color_discrete_map=status_colors,
        hover_name="product_name",
        hover_data={"category": True, "days_of_inventory": True, "total_sales": ":$,.2f"}
    )

    apply_enterprise_layout(fig, title="Inventory Health Matrix: Current Stock vs. Sales Velocity", xaxis_title="Daily Sales Velocity (Units/Day)", yaxis_title="Current Stock (Units)", height=400)
    return fig
