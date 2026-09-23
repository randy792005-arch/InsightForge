"""
InsightForge Recommendation Agent (Agent 5)
Responsible for:
- Synthesizing Analytics and Forecasting agent outputs
- Formulating data-driven inventory replenishment decisions
- Detecting stockout risks and calculating safety stock reorders
- Identifying high-demand opportunities and dead-stock clearance actions
- Categorizing recommendations by urgency (CRITICAL, HIGH, MEDIUM, LOW)
- Persisting structured decision records into SQLite
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    """
    Autonomous Decision Support Agent formulating prescriptive inventory, merchandising, and pricing actions.
    """

    def __init__(self, db_manager=None):
        super().__init__(name="Recommendation Agent", db_manager=db_manager)

    def _execute(
        self, 
        analytics_data: Dict[str, Any], 
        forecasting_data: Dict[str, Any], 
        session_id: str = "default_session",
        lead_time_days: int = 7,
        target_cover_days: int = 30,
        **kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Synthesizes analytics and forecast intelligence into prioritized retail action recommendations.
        """
        self.log("Synthesizing analytics and forecast models to generate decision recommendations...")

        product_inventory = analytics_data.get("product_inventory", pd.DataFrame())
        kpis = analytics_data.get("kpis", {})
        top_10 = analytics_data.get("top_10_products", pd.DataFrame())
        bottom_10 = analytics_data.get("bottom_10_products", pd.DataFrame())
        category_stats = analytics_data.get("category_stats", pd.DataFrame())
        
        projected_avg_daily = forecasting_data.get("projected_avg_daily", 0.0)
        overall_avg_daily = kpis.get("avg_daily_revenue", 1.0)
        demand_growth_ratio = (projected_avg_daily / max(1.0, overall_avg_daily)) if overall_avg_daily > 0 else 1.0

        recommendations = []

        if not product_inventory.empty:
            for _, item in product_inventory.iterrows():
                product_name = str(item["product_name"])
                category = str(item["category"])
                current_stock = float(item["current_stock"])
                daily_velocity = float(item["daily_velocity"])
                days_of_inventory = float(item["days_of_inventory"])
                unit_price = float(item.get("avg_price", 20.0))

                # Expected daily demand factoring forecast trend
                forecasted_daily_demand = max(0.2, round(daily_velocity * demand_growth_ratio, 2))

                # Reorder calculation: Target Stock = (Daily Demand * Target Cover Days) + Safety Buffer
                target_stock = round(forecasted_daily_demand * target_cover_days)
                reorder_qty = max(0.0, target_stock - current_stock)

                # Rule 1: Immediate Stockout / Critical Replenishment (< 7 Days of Stock)
                if current_stock <= 0:
                    recommendations.append({
                        "product_name": product_name,
                        "category": category,
                        "priority": "CRITICAL",
                        "issue_type": "Stockout Alert",
                        "trigger_metric": f"Stock is 0 units (0.0 Days Cover)",
                        "action_recommended": f"Issue immediate emergency purchase order for {reorder_qty:.0f} units to prevent lost sales revenue.",
                        "expected_impact": f"Recovers estimated ${reorder_qty * unit_price:,.2f} in potential lost sales over next {target_cover_days} days.",
                        "current_stock": current_stock,
                        "daily_velocity": daily_velocity,
                        "reorder_qty": reorder_qty
                    })
                elif days_of_inventory <= lead_time_days:
                    recommendations.append({
                        "product_name": product_name,
                        "category": category,
                        "priority": "CRITICAL",
                        "issue_type": "Urgent Replenishment",
                        "trigger_metric": f"Only {days_of_inventory:.1f} days of inventory remaining (Stock: {current_stock:.0f})",
                        "action_recommended": f"Expedite reorder of {reorder_qty:.0f} units. Stock will deplete before standard supplier lead time ({lead_time_days} days).",
                        "expected_impact": f"Prevents imminent stockout and protects ${reorder_qty * unit_price:,.2f} in revenue.",
                        "current_stock": current_stock,
                        "daily_velocity": daily_velocity,
                        "reorder_qty": reorder_qty
                    })
                elif days_of_inventory <= 14:
                    recommendations.append({
                        "product_name": product_name,
                        "category": category,
                        "priority": "HIGH",
                        "issue_type": "Upcoming Reorder",
                        "trigger_metric": f"{days_of_inventory:.1f} days of inventory (Stock: {current_stock:.0f})",
                        "action_recommended": f"Place standard purchase order for {reorder_qty:.0f} units within the next 3 business days.",
                        "expected_impact": f"Maintains optimal buffer stock and avoids rush freight delivery surcharges.",
                        "current_stock": current_stock,
                        "daily_velocity": daily_velocity,
                        "reorder_qty": reorder_qty
                    })

                # Rule 2: Overstock & Dead Stock (> 90 Days Cover with Low Velocity)
                elif days_of_inventory > 90 and daily_velocity < 0.8:
                    excess_units = max(0.0, current_stock - (daily_velocity * 45))
                    recommendations.append({
                        "product_name": product_name,
                        "category": category,
                        "priority": "MEDIUM",
                        "issue_type": "Overstock Liquidation",
                        "trigger_metric": f"{days_of_inventory:.1f} days of supply (Stock: {current_stock:.0f}, Velocity: {daily_velocity:.2f} units/day)",
                        "action_recommended": f"Apply a 20% - 30% promotional markdown or create a product bundle to liquidate ~{excess_units:.0f} excess units.",
                        "expected_impact": f"Frees up approx. ${excess_units * unit_price:,.2f} in tied-up working capital and reduces warehousing holding costs.",
                        "current_stock": current_stock,
                        "daily_velocity": daily_velocity,
                        "reorder_qty": 0.0
                    })

        # Rule 3: Top Revenue Fast Movers Strategy
        if not top_10.empty:
            for _, top_item in top_10.head(3).iterrows():
                p_name = str(top_item["product_name"])
                p_cat = str(top_item["category"])
                p_rev = float(top_item["total_sales"])
                p_share = float(top_item["revenue_share_pct"])

                # Add strategic recommendation if not already critically flagged
                if not any(r["product_name"] == p_name and r["priority"] == "CRITICAL" for r in recommendations):
                    recommendations.append({
                        "product_name": p_name,
                        "category": p_cat,
                        "priority": "HIGH",
                        "issue_type": "Fast-Mover Merchandising",
                        "trigger_metric": f"Top #{_ + 1} Best Seller ({p_share:.1f}% of total store revenue, ${p_rev:,.2f})",
                        "action_recommended": "Allocate prime digital & physical merchandising display. Negotiate tier-1 bulk volume vendor discounts.",
                        "expected_impact": "Maximizes conversion rate and improves gross margin on core revenue driving SKU.",
                        "current_stock": float(top_item.get("current_stock", 0)),
                        "daily_velocity": round(float(top_item["quantity_sold"]) / max(1, kpis.get("total_days", 30)), 2),
                        "reorder_qty": 0.0
                    })

        # Rule 4: Category Capital Allocation
        if not category_stats.empty:
            top_cat = category_stats.iloc[0]
            recommendations.append({
                "product_name": f"All [{top_cat['category']}] SKUs",
                "category": str(top_cat["category"]),
                "priority": "MEDIUM",
                "issue_type": "Category Expansion",
                "trigger_metric": f"Dominant category contributing {top_cat['revenue_share_pct']:.1f}% (${top_cat['total_sales']:,.2f}) of revenue",
                "action_recommended": f"Prioritize 40% of seasonal inventory budget to expand product depth within {top_cat['category']}.",
                "expected_impact": "Captures expanding category demand and accelerates year-over-year revenue expansion.",
                "current_stock": 0.0,
                "daily_velocity": 0.0,
                "reorder_qty": 0.0
            })

        rec_df = pd.DataFrame(recommendations)
        
        # Sort by Priority (CRITICAL -> HIGH -> MEDIUM -> LOW)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        if not rec_df.empty:
            rec_df["priority_rank"] = rec_df["priority"].map(priority_order)
            rec_df = rec_df.sort_values(by=["priority_rank", "current_stock"]).drop(columns=["priority_rank"]).reset_index(drop=True)

        self.log(f"Generated {len(rec_df)} prioritized retail decision recommendations.")

        # Persist to SQLite
        if not rec_df.empty:
            self.db.save_recommendations(rec_df, dataset_id=session_id)
            self.log("Persisted decision recommendations into SQLite table 'recommendations'.")

        critical_count = int((rec_df["priority"] == "CRITICAL").sum()) if not rec_df.empty else 0
        high_count = int((rec_df["priority"] == "HIGH").sum()) if not rec_df.empty else 0
        medium_count = int((rec_df["priority"] == "MEDIUM").sum()) if not rec_df.empty else 0

        status_msg = (
            f"Generated {len(rec_df)} retail recommendations: "
            f"{critical_count} Critical, {high_count} High, {medium_count} Medium Priority."
        )

        return {
            "recommendations_df": rec_df,
            "total_recommendations": len(rec_df),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "stockout_alerts": rec_df[rec_df["issue_type"].str.contains("Stockout|Replenishment", na=False)] if not rec_df.empty else pd.DataFrame(),
            "overstock_alerts": rec_df[rec_df["issue_type"].str.contains("Overstock", na=False)] if not rec_df.empty else pd.DataFrame()
        }, "SUCCESS", status_msg
