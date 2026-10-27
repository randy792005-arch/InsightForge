"""
InsightForge Forecasting Agent (Agent 4)
Responsible for:
- Orchestrating time-series (Prophet) and machine learning (XGBoost) sales forecasting
- Rigorously evaluating models with authentic hold-out metrics (MAE, RMSE, MAPE, R2)
- Projecting future demand over configurable horizons (7, 14, 30, 60, 90 days)
- Persisting forecast time series and evaluation metrics into SQLite
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from agents.base_agent import BaseAgent
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster


class ForecastingAgent(BaseAgent):
    """
    Autonomous agent for demand forecasting, predictive modeling, and model evaluation.
    """

    def __init__(self, db_manager=None):
        super().__init__(name="Forecasting Agent", db_manager=db_manager)

    def _execute(
        self, 
        df_clean: pd.DataFrame, 
        model_type: str = "Prophet", 
        forecast_horizon: int = 30,
        session_id: str = "default_session",
        **kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Trains the selected model and produces future sales projections and metrics.
        """
        self.log(f"Initiating demand forecasting with model: {model_type} (Horizon: {forecast_horizon} days)...")

        model_type_clean = model_type.strip().lower()
        
        forecast_df = None
        metrics = {}
        backtest_df = None
        secondary_forecast = None
        secondary_metrics = None

        if "prophet" in model_type_clean:
            self.log("Training Facebook Prophet time-series model on historical sales...")
            forecaster = ProphetForecaster(forecast_horizon=forecast_horizon)
            forecast_df, metrics, backtest_df = forecaster.train_and_forecast(df_clean, forecast_horizon=forecast_horizon)
            model_display_name = "Prophet"
            
        elif "xgboost" in model_type_clean:
            self.log("Training XGBoost Regressor with calendar and lag features...")
            forecaster = XGBoostForecaster(forecast_horizon=forecast_horizon)
            forecast_df, metrics, backtest_df = forecaster.train_and_forecast(df_clean, forecast_horizon=forecast_horizon)
            model_display_name = "XGBoost"
            
        elif "both" in model_type_clean or "compare" in model_type_clean:
            self.log("Running comparative training for both Prophet and XGBoost...")
            prophet_forecaster = ProphetForecaster(forecast_horizon=forecast_horizon)
            p_df, p_metrics, p_backtest = prophet_forecaster.train_and_forecast(df_clean, forecast_horizon=forecast_horizon)
            
            xgb_forecaster = XGBoostForecaster(forecast_horizon=forecast_horizon)
            x_df, x_metrics, x_backtest = xgb_forecaster.train_and_forecast(df_clean, forecast_horizon=forecast_horizon)

            # Choose primary model based on lower MAE on hold-out test
            if p_metrics["mae"] <= x_metrics["mae"]:
                forecast_df, metrics, backtest_df = p_df, p_metrics, p_backtest
                secondary_forecast, secondary_metrics = x_df, x_metrics
                model_display_name = "Prophet (Best Performing)"
            else:
                forecast_df, metrics, backtest_df = x_df, x_metrics, x_backtest
                secondary_forecast, secondary_metrics = p_df, p_metrics
                model_display_name = "XGBoost (Best Performing)"
        else:
            # Default to Prophet
            self.log(f"Unrecognized model '{model_type}', defaulting to Prophet.")
            forecaster = ProphetForecaster(forecast_horizon=forecast_horizon)
            forecast_df, metrics, backtest_df = forecaster.train_and_forecast(df_clean, forecast_horizon=forecast_horizon)
            model_display_name = "Prophet"

        self.log(f"Model evaluation complete: MAE=₹{metrics['mae']:.2f}, RMSE=₹{metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%, R2={metrics['r2']:.3f}")

        # Future projected sales summary
        future_only = forecast_df[forecast_df["is_historical"] == 0]
        projected_total_revenue = float(future_only["predicted_sales"].sum())
        projected_avg_daily = float(future_only["predicted_sales"].mean()) if not future_only.empty else 0.0

        # Persist to SQLite
        self.db.save_forecast_results(
            forecast_df=forecast_df,
            metrics_dict=metrics,
            model_type=model_display_name,
            dataset_id=session_id,
            target_name="Overall Revenue"
        )
        self.log("Persisted forecast series and evaluation metrics to SQLite table 'forecast_records'.")

        status_msg = (
            f"Forecast completed using {model_display_name}. Projected {forecast_horizon}-day sales: "
            f"₹{projected_total_revenue:,.2f} (MAE: ₹{metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%)."
        )

        return {
            "forecast_df": forecast_df,
            "metrics": metrics,
            "backtest_df": backtest_df,
            "model_name": model_display_name,
            "projected_total_revenue": round(projected_total_revenue, 2),
            "projected_avg_daily": round(projected_avg_daily, 2),
            "secondary_forecast": secondary_forecast,
            "secondary_metrics": secondary_metrics
        }, "SUCCESS", status_msg
