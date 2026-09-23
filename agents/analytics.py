"""
InsightForge Analytics Agent (Agent 3)
Responsible for:
- Computing core retail KPIs (Total Revenue, Quantity Sold, AOV, Transactions)
- Analyzing sales trends across time horizons (Daily, Weekly, Monthly, Seasonality)
- Evaluating product and category performance (Top/Bottom, Revenue Share, Pareto 80/20)
- Assessing inventory status and sales velocity
- Persisting analytical summaries and KPIs into SQLite
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent


class AnalyticsAgent(BaseAgent):
    """
    Autonomous agent for retail exploratory data analysis, business KPIs, and category diagnostics.
    """

    def __init__(self, db_manager=None):
        super().__init__(name="Analytics Agent", db_manager=db_manager)

    def _execute(
        self, 
        df_clean: pd.DataFrame, 
        session_id: str = "default_session",
        **kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Performs in-depth retail business intelligence and exploratory sales analysis.
        """
        self.log(f"Computing retail analytics across {len(df_clean):,} clean records...")

        df = df_clean.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 1. Primary High-Level KPIs
        total_revenue = float(df["total_sales"].sum())
        total_units_sold = float(df["quantity"].sum())
        total_transactions = int(len(df))
        unique_products = int(df["product_name"].nunique())
        unique_categories = int(df["category"].nunique())
        
        avg_order_value = round(total_revenue / max(1, total_transactions), 2)
        avg_unit_price = round(total_revenue / max(1.0, total_units_sold), 2)

        min_date = df["date"].min()
        max_date = df["date"].max()
        total_days = max(1, (max_date - min_date).days + 1)
        
        avg_daily_revenue = round(total_revenue / total_days, 2)
        avg_daily_units = round(total_units_sold / total_days, 2)

        # Inventory valuation
        total_current_stock = float(df.groupby("product_name")["current_stock"].first().sum())
        avg_price_per_product = df.groupby("product_name")["unit_price"].mean()
        stock_per_product = df.groupby("product_name")["current_stock"].first()
        estimated_stock_valuation = float((stock_per_product * avg_price_per_product).sum())

        kpis = {
            "total_revenue": round(total_revenue, 2),
            "total_units_sold": round(total_units_sold, 0),
            "total_transactions": total_transactions,
            "avg_order_value": avg_order_value,
            "avg_unit_price": avg_unit_price,
            "unique_products": unique_products,
            "unique_categories": unique_categories,
            "total_days": total_days,
            "avg_daily_revenue": avg_daily_revenue,
            "avg_daily_units": avg_daily_units,
            "total_current_stock": round(total_current_stock, 0),
            "estimated_stock_valuation": round(estimated_stock_valuation, 2),
            "start_date": min_date.strftime("%Y-%m-%d"),
            "end_date": max_date.strftime("%Y-%m-%d")
        }
        self.log(f"Computed core KPIs: Revenue=${total_revenue:,.2f}, Units={total_units_sold:,.0f}, AOV=${avg_order_value:,.2f}")

        # 2. Time-Series Aggregations
        # Daily
        daily_sales = (
            df.groupby(pd.Grouper(key="date", freq="D"))
            .agg(total_sales=("total_sales", "sum"), quantity=("quantity", "sum"), transactions=("total_sales", "count"))
            .reset_index()
        )
        # Ensure smooth continuous daily timeline
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        daily_sales = (
            daily_sales.set_index("date")
            .reindex(all_dates, fill_value=0.0)
            .rename_axis("date")
            .reset_index()
        )
        daily_sales["sales_7d_ma"] = daily_sales["total_sales"].rolling(7, min_periods=1).mean().round(2)

        # Monthly
        monthly_sales = (
            df.groupby(pd.Grouper(key="date", freq="ME"))
            .agg(total_sales=("total_sales", "sum"), quantity=("quantity", "sum"), transactions=("total_sales", "count"))
            .reset_index()
        )
        monthly_sales["month_label"] = monthly_sales["date"].dt.strftime("%b %Y")
        monthly_sales["mom_growth_pct"] = (monthly_sales["total_sales"].pct_change() * 100).fillna(0.0).round(1)

        # Day of Week Distribution
        df["day_name"] = df["date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_sales = (
            df.groupby("day_name")
            .agg(total_sales=("total_sales", "sum"), quantity=("quantity", "sum"), transactions=("total_sales", "count"))
            .reindex(day_order)
            .reset_index()
        )
        dow_sales["share_pct"] = (dow_sales["total_sales"] / max(1.0, total_revenue) * 100).round(1)

        # 3. Product Performance & Rankings
        product_stats = (
            df.groupby(["product_name", "category"])
            .agg(
                total_sales=("total_sales", "sum"),
                quantity_sold=("quantity", "sum"),
                transactions=("total_sales", "count"),
                avg_price=("unit_price", "mean"),
                current_stock=("current_stock", "first")
            )
            .reset_index()
        )
        product_stats["revenue_share_pct"] = (product_stats["total_sales"] / max(1.0, total_revenue) * 100).round(2)
        product_stats = product_stats.sort_values(by="total_sales", ascending=False).reset_index(drop=True)

        top_10_products = product_stats.head(10).copy()
        bottom_10_products = product_stats.tail(10).sort_values(by="total_sales", ascending=True).copy()

        # 4. Category Performance Breakdown
        category_stats = (
            df.groupby("category")
            .agg(
                total_sales=("total_sales", "sum"),
                quantity_sold=("quantity", "sum"),
                transactions=("total_sales", "count"),
                product_count=("product_name", "nunique"),
                avg_order_value=("total_sales", "mean")
            )
            .reset_index()
        )
        category_stats["revenue_share_pct"] = (category_stats["total_sales"] / max(1.0, total_revenue) * 100).round(2)
        category_stats = category_stats.sort_values(by="total_sales", ascending=False).reset_index(drop=True)

        # 5. Pareto (80/20) Product Analysis
        pareto_df = product_stats.copy()
        pareto_df["cumulative_sales"] = pareto_df["total_sales"].cumsum()
        pareto_df["cumulative_share_pct"] = (pareto_df["cumulative_sales"] / max(1.0, total_revenue) * 100).round(1)
        pareto_df["rank"] = pareto_df.index + 1
        pareto_80_count = int((pareto_df["cumulative_share_pct"] <= 80.0).sum() + 1)
        pareto_80_pct = round((pareto_80_count / max(1, unique_products)) * 100, 1)

        kpis["pareto_80_product_count"] = pareto_80_count
        kpis["pareto_80_product_pct"] = pareto_80_pct

        # 6. Inventory Status and Velocity Analysis
        product_inventory = product_stats.copy()
        product_inventory["daily_velocity"] = (product_inventory["quantity_sold"] / total_days).round(2)
        # Days of inventory remaining = current_stock / daily_velocity
        product_inventory["days_of_inventory"] = np.where(
            product_inventory["daily_velocity"] > 0,
            (product_inventory["current_stock"] / product_inventory["daily_velocity"]).round(1),
            999.0
        )

        def assign_stock_status(row):
            if row["current_stock"] <= 0:
                return "Out of Stock"
            elif row["days_of_inventory"] <= 7:
                return "Critical Stock"
            elif row["days_of_inventory"] <= 14:
                return "Low Stock"
            elif row["days_of_inventory"] > 90 and row["daily_velocity"] < 0.5:
                return "Overstocked"
            else:
                return "Optimal"

        product_inventory["stock_status"] = product_inventory.apply(assign_stock_status, axis=1)

        # 7. Persist KPIs into SQLite
        self.db.save_analytics_kpis(kpis, dataset_id=session_id)
        self.log("Persisted business KPIs into SQLite table 'analytics_kpis'.")

        status_msg = (
            f"Analyzed {total_transactions:,} orders across {unique_products} products and {unique_categories} categories. "
            f"Total Revenue: ${total_revenue:,.2f} over {total_days} days."
        )

        return {
            "kpis": kpis,
            "daily_sales": daily_sales,
            "monthly_sales": monthly_sales,
            "dow_sales": dow_sales,
            "product_stats": product_stats,
            "top_10_products": top_10_products,
            "bottom_10_products": bottom_10_products,
            "category_stats": category_stats,
            "pareto_df": pareto_df,
            "product_inventory": product_inventory
        }, "SUCCESS", status_msg
