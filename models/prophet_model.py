"""
InsightForge Prophet Forecasting Model
Provides robust time-series sales and demand forecasting using Facebook Prophet,
including historical train-test backtesting and authentic evaluation metrics (MAE, RMSE, MAPE, R2).
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import logging

# Suppress cmdstanpy / prophet verbose output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


class ProphetForecaster:
    """
    Time-series forecasting wrapper for Prophet.
    Handles data aggregation, train-test split evaluation, and multi-period future demand projection.
    """

    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.metrics = {}
        self.freq = "D"

    def prepare_series(self, df: pd.DataFrame, date_col: str = "date", target_col: str = "total_sales") -> pd.DataFrame:
        """
        Aggregates clean retail dataframe to series formatted for Prophet ('ds', 'y')
        with automatic frequency detection (daily, weekly, monthly).
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Detect frequency
        sorted_dates = pd.Series(df_copy[date_col].dt.normalize().unique()).sort_values()
        diffs = sorted_dates.diff().dt.days.dropna()
        median_gap = float(diffs.median()) if not diffs.empty else 1.0

        if median_gap >= 25:
            self.freq = "MS"
        elif 6 <= median_gap <= 8:
            self.freq = "W"
        else:
            self.freq = "D"

        daily_df = (
            df_copy.groupby(pd.Grouper(key=date_col, freq=self.freq))[target_col]
            .sum()
            .reset_index()
        )
        daily_df.columns = ["ds", "y"]
        daily_df["ds"] = pd.to_datetime(daily_df["ds"])
        daily_df = daily_df.sort_values("ds").reset_index(drop=True)

        if self.freq == "D" and len(daily_df) > 1:
            all_dates = pd.date_range(start=daily_df["ds"].min(), end=daily_df["ds"].max(), freq="D")
            daily_df = daily_df.set_index("ds").reindex(all_dates, fill_value=0.0).rename_axis("ds").reset_index()

        return daily_df

    def train_and_forecast(
        self, 
        df: pd.DataFrame, 
        forecast_horizon: Optional[int] = None, 
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Fits Prophet on historical data, conducts out-of-sample backtesting, and generates future predictions.
        """
        from prophet import Prophet

        horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon
        daily_df = self.prepare_series(df)
        total_points = len(daily_df)

        if total_points < 7:
            raise ValueError(
                f"Insufficient historical data for forecasting: dataset contains only {total_points} data points. "
                "A minimum of 7 data points is required to train reliable trend models."
            )

        # 1. Backtesting Split (Hold-out evaluation)
        test_size = max(2, min(30, int(total_points * test_ratio)))
        if test_size >= total_points:
            test_size = max(1, total_points - 2)

        train_df = daily_df.iloc[:-test_size].copy()
        test_df = daily_df.iloc[-test_size:].copy()

        # Fit model on training partition for evaluation
        eval_model = Prophet(
            yearly_seasonality=True if total_points >= 365 else "auto",
            weekly_seasonality=True if (total_points >= 14 and self.freq == "D") else False,
            daily_seasonality=False,
            interval_width=0.90,
            changepoint_prior_scale=0.05
        )
        eval_model.fit(train_df)

        # Predict test window
        periods_test = test_size
        future_test = eval_model.make_future_dataframe(periods=periods_test, freq=self.freq, include_history=False)
        test_pred = eval_model.predict(future_test)

        # Align test actuals and test predictions
        y_true = test_df["y"].values
        y_pred = test_pred["yhat"].clip(lower=0).values[:len(y_true)]

        # Authentic Error Metric Calculations
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        
        non_zero_mask = y_true > 0
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        else:
            mape = 0.0
        
        try:
            r2 = float(r2_score(y_true, y_pred))
            if r2 < -1.0:
                r2 = -1.0
        except Exception:
            r2 = 0.0

        metrics = {
            "model_name": "Prophet (Additive / Trend-Seasonality)",
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 3),
            "train_days": len(train_df),
            "test_days": len(y_true),
            "forecast_horizon": horizon
        }

        # Backtest comparison dataframe
        backtest_df = pd.DataFrame({
            "date": test_df["ds"].values[:len(y_pred)],
            "actual": y_true,
            "predicted": y_pred
        })

        # 2. Refit on FULL historical dataset for best future forecasting
        full_model = Prophet(
            yearly_seasonality=True if total_points >= 365 else "auto",
            weekly_seasonality=True if (total_points >= 14 and self.freq == "D") else False,
            daily_seasonality=False,
            interval_width=0.90,
            changepoint_prior_scale=0.05
        )
        full_model.fit(daily_df)

        # Compute future periods
        if self.freq == "MS":
            future_periods = max(1, horizon // 30)
        elif self.freq == "W":
            future_periods = max(1, horizon // 7)
        else:
            future_periods = horizon

        future_df = full_model.make_future_dataframe(periods=future_periods, freq=self.freq)
        forecast_output = full_model.predict(future_df)

        # Clip predictions to non-negative sales
        forecast_output["yhat"] = forecast_output["yhat"].clip(lower=0)
        forecast_output["yhat_lower"] = forecast_output["yhat_lower"].clip(lower=0)
        forecast_output["yhat_upper"] = forecast_output["yhat_upper"].clip(lower=0)

        daily_dict = dict(zip(daily_df["ds"].dt.strftime("%Y-%m-%d"), daily_df["y"]))

        combined_records = []
        for _, row in forecast_output.iterrows():
            d_str = row["ds"].strftime("%Y-%m-%d")
            is_hist = 1 if d_str in daily_dict else 0
            actual_val = daily_dict.get(d_str, None)

            combined_records.append({
                "date": d_str,
                "is_historical": is_hist,
                "actual_sales": actual_val,
                "predicted_sales": round(float(row["yhat"]), 2),
                "lower_bound": round(float(row["yhat_lower"]), 2),
                "upper_bound": round(float(row["yhat_upper"]), 2)
            })

        combined_forecast_df = pd.DataFrame(combined_records)
        return combined_forecast_df, metrics, backtest_df
