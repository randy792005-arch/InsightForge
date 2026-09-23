"""
InsightForge XGBoost Forecasting Model
Provides high-performance machine learning regression for retail sales forecasting,
engineering calendar temporal features, multi-step lags, and rolling statistics.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")


class XGBoostForecaster:
    """
    Feature-engineered gradient boosting forecaster for demand and sales.
    Includes lag creation, calendar features, recursive multi-step forecasting, and authentic backtesting.
    """

    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_names = []
        self.feature_importances = {}
        self.metrics = {}

    def prepare_series(self, df: pd.DataFrame, date_col: str = "date", target_col: str = "total_sales") -> pd.DataFrame:
        """Aggregates clean retail dataframe to continuous daily sales."""
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        daily_df = (
            df_copy.groupby(pd.Grouper(key=date_col, freq="D"))[target_col]
            .sum()
            .reset_index()
        )
        daily_df.columns = ["date", "sales"]
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date").reset_index(drop=True)
        # Ensure daily continuity
        all_dates = pd.date_range(start=daily_df["date"].min(), end=daily_df["date"].max(), freq="D")
        daily_df = daily_df.set_index("date").reindex(all_dates, fill_value=0.0).rename_axis("date").reset_index()
        return daily_df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rich temporal, lag, and rolling window features for XGBoost.
        """
        df_feat = df.copy()
        df_feat["date"] = pd.to_datetime(df_feat["date"])

        # Calendar features
        df_feat["dayofweek"] = df_feat["date"].dt.dayofweek
        df_feat["dayofmonth"] = df_feat["date"].dt.day
        df_feat["month"] = df_feat["date"].dt.month
        df_feat["quarter"] = df_feat["date"].dt.quarter
        df_feat["dayofyear"] = df_feat["date"].dt.dayofyear
        df_feat["is_weekend"] = df_feat["dayofweek"].isin([5, 6]).astype(int)
        df_feat["weekofyear"] = df_feat["date"].dt.isocalendar().week.astype(int)

        # Lag features
        lags = [1, 2, 3, 7, 14, 21, 28]
        for lag in lags:
            df_feat[f"lag_{lag}"] = df_feat["sales"].shift(lag)

        # Rolling window features (shifted to prevent lookahead leakage)
        df_feat["rolling_mean_7"] = df_feat["sales"].shift(1).rolling(window=7, min_periods=1).mean()
        df_feat["rolling_std_7"] = df_feat["sales"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
        df_feat["rolling_mean_14"] = df_feat["sales"].shift(1).rolling(window=14, min_periods=1).mean()
        df_feat["rolling_mean_30"] = df_feat["sales"].shift(1).rolling(window=30, min_periods=1).mean()

        return df_feat

    def train_and_forecast(
        self, 
        df: pd.DataFrame, 
        forecast_horizon: Optional[int] = None, 
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Trains XGBoost Regressor with iterative multi-step future forecasting and backtesting metrics.
        """
        horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon
        daily_df = self.prepare_series(df)
        total_points = len(daily_df)

        if total_points < 14:
            raise ValueError(
                f"Insufficient historical data for forecasting: dataset contains only {total_points} days. "
                "A minimum of 14 continuous days is required to train reliable time-series models."
            )

        # Feature dataframe
        feat_df = self.create_features(daily_df)
        # Drop rows with NaN from earliest lags (first 28 days or min available)
        valid_feat_df = feat_df.dropna().reset_index(drop=True)

        if len(valid_feat_df) < 7:
            # Fallback for short series: drop only lag_1 and calendar NaNs
            valid_feat_df = feat_df.dropna(subset=["lag_1", "rolling_mean_7"]).reset_index(drop=True)
            valid_feat_df = valid_feat_df.fillna(0.0)

        feature_cols = [c for c in valid_feat_df.columns if c not in ["date", "sales"]]
        self.feature_names = feature_cols

        # 1. Backtesting Split
        test_size = max(7, min(30, int(len(valid_feat_df) * test_ratio)))
        train_df = valid_feat_df.iloc[:-test_size].copy()
        test_df = valid_feat_df.iloc[-test_size:].copy()

        X_train = train_df[feature_cols]
        y_train = train_df["sales"]
        X_test = test_df[feature_cols]
        y_test = test_df["sales"]

        # Train evaluation model
        eval_reg = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=0
        )
        eval_reg.fit(X_train, y_train)
        y_pred = eval_reg.predict(X_test).clip(min=0)

        # Authentic Error Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))

        non_zero_mask = y_test > 0
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100)
        else:
            mape = 0.0

        try:
            r2 = float(r2_score(y_test, y_pred))
            if r2 < -1.0:
                r2 = -1.0
        except Exception:
            r2 = 0.0

        metrics = {
            "model_name": "XGBoost Regressor (Gradient Boosted Trees)",
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 3),
            "train_days": len(train_df),
            "test_days": test_size,
            "forecast_horizon": horizon
        }

        backtest_df = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_test.values,
            "predicted": y_pred
        })

        # 2. Refit on all available data
        full_reg = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=0
        )
        full_reg.fit(valid_feat_df[feature_cols], valid_feat_df["sales"])
        self.model = full_reg

        # Extract feature importances
        importances = dict(zip(feature_cols, [float(x) for x in full_reg.feature_importances_]))
        self.feature_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        # 3. Recursive multi-step future forecasting
        last_date = daily_df["date"].max()
        extended_sales = list(daily_df["sales"].values)
        extended_dates = list(daily_df["date"].values)
        future_predictions = []

        # Residual std for confidence intervals
        residual_std = float(np.std(y_test.values - y_pred)) if len(y_test) > 1 else (mae * 1.25)

        for step in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=step)
            # Create features for this future step
            sales_series = pd.Series(extended_sales)
            
            feat_row = {
                "dayofweek": next_date.dayofweek,
                "dayofmonth": next_date.day,
                "month": next_date.month,
                "quarter": next_date.quarter,
                "dayofyear": next_date.dayofyear,
                "is_weekend": 1 if next_date.dayofweek in [5, 6] else 0,
                "weekofyear": int(next_date.isocalendar().week)
            }

            # Lags
            lags = [1, 2, 3, 7, 14, 21, 28]
            for lag in lags:
                feat_row[f"lag_{lag}"] = extended_sales[-lag] if len(extended_sales) >= lag else extended_sales[0]

            # Rolling
            feat_row["rolling_mean_7"] = float(sales_series.iloc[-7:].mean()) if len(sales_series) >= 7 else float(sales_series.mean())
            feat_row["rolling_std_7"] = float(sales_series.iloc[-7:].std()) if len(sales_series) >= 7 else 0.0
            feat_row["rolling_mean_14"] = float(sales_series.iloc[-14:].mean()) if len(sales_series) >= 14 else float(sales_series.mean())
            feat_row["rolling_mean_30"] = float(sales_series.iloc[-30:].mean()) if len(sales_series) >= 30 else float(sales_series.mean())

            # Convert to DataFrame
            X_future = pd.DataFrame([feat_row])[feature_cols]
            pred_val = float(np.clip(full_reg.predict(X_future)[0], 0, None))

            # Bounds with increasing uncertainty horizon
            uncertainty_growth = 1.0 + (step / horizon) * 0.5
            lower_bound = max(0.0, pred_val - (1.645 * residual_std * uncertainty_growth))
            upper_bound = pred_val + (1.645 * residual_std * uncertainty_growth)

            future_predictions.append({
                "date": next_date,
                "predicted_sales": round(pred_val, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "trend": round(pred_val, 2),
                "is_historical": 0,
                "actual_sales": np.nan
            })

            # Append to history for next recursive lag steps
            extended_sales.append(pred_val)
            extended_dates.append(next_date)

        # Historical part for unified view
        hist_records = []
        for idx, row in daily_df.iterrows():
            hist_records.append({
                "date": row["date"],
                "predicted_sales": round(float(row["sales"]), 2),
                "lower_bound": round(float(row["sales"]), 2),
                "upper_bound": round(float(row["sales"]), 2),
                "trend": round(float(row["sales"]), 2),
                "is_historical": 1,
                "actual_sales": round(float(row["sales"]), 2)
            })

        combined_forecast_df = pd.DataFrame(hist_records + future_predictions)
        self.metrics = metrics
        return combined_forecast_df, metrics, backtest_df
