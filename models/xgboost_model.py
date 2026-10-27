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
    Includes adaptive lag creation, calendar features, recursive multi-step forecasting, and authentic backtesting.
    """

    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_names = []
        self.feature_importances = {}
        self.metrics = {}
        self.freq = "D"

    def prepare_series(self, df: pd.DataFrame, date_col: str = "date", target_col: str = "total_sales") -> pd.DataFrame:
        """Aggregates clean retail dataframe to series with automatic frequency detection."""
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
        daily_df.columns = ["date", "sales"]
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        if self.freq == "D" and len(daily_df) > 1:
            all_dates = pd.date_range(start=daily_df["date"].min(), end=daily_df["date"].max(), freq="D")
            daily_df = daily_df.set_index("date").reindex(all_dates, fill_value=0.0).rename_axis("date").reset_index()

        return daily_df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates adaptive temporal, lag, and rolling window features for XGBoost.
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

        n_rows = len(df_feat)
        if n_rows < 15:
            lags = [1, 2]
            df_feat["rolling_mean_3"] = df_feat["sales"].shift(1).rolling(window=2, min_periods=1).mean()
        elif n_rows < 40:
            lags = [1, 2, 3, 7]
            df_feat["rolling_mean_7"] = df_feat["sales"].shift(1).rolling(window=3, min_periods=1).mean()
            df_feat["rolling_std_7"] = df_feat["sales"].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        else:
            lags = [1, 2, 3, 7, 14, 21, 28]
            df_feat["rolling_mean_7"] = df_feat["sales"].shift(1).rolling(window=7, min_periods=1).mean()
            df_feat["rolling_std_7"] = df_feat["sales"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
            df_feat["rolling_mean_14"] = df_feat["sales"].shift(1).rolling(window=14, min_periods=1).mean()
            df_feat["rolling_mean_30"] = df_feat["sales"].shift(1).rolling(window=30, min_periods=1).mean()

        for lag in lags:
            df_feat[f"lag_{lag}"] = df_feat["sales"].shift(lag)

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

        if total_points < 7:
            raise ValueError(
                f"Insufficient historical data for forecasting: dataset contains only {total_points} data points. "
                "A minimum of 7 data points is required to train reliable trend models."
            )

        # Feature dataframe
        feat_df = self.create_features(daily_df)
        valid_feat_df = feat_df.dropna().reset_index(drop=True)

        if len(valid_feat_df) < 5:
            valid_feat_df = feat_df.fillna(0.0).reset_index(drop=True)

        feature_cols = [c for c in valid_feat_df.columns if c not in ["date", "sales"]]
        self.feature_names = feature_cols

        # 1. Backtesting Split
        test_size = max(2, min(30, int(len(valid_feat_df) * test_ratio)))
        if test_size >= len(valid_feat_df):
            test_size = max(1, len(valid_feat_df) - 2)

        train_df = valid_feat_df.iloc[:-test_size].copy()
        test_df = valid_feat_df.iloc[-test_size:].copy()

        X_train = train_df[feature_cols]
        y_train = train_df["sales"]
        X_test = test_df[feature_cols]
        y_test = test_df["sales"]

        # Train evaluation model
        eval_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
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
            "test_days": len(y_test),
            "forecast_horizon": horizon
        }

        backtest_df = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_test.values,
            "predicted": y_pred
        })

        # 2. Train on FULL series
        full_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=0
        )
        full_reg.fit(valid_feat_df[feature_cols], valid_feat_df["sales"])
        self.model = full_reg

        # Feature Importances
        importances = full_reg.feature_importances_
        self.feature_importances = dict(zip(feature_cols, [round(float(v), 4) for v in importances]))

        # 3. Recursive Multi-Step Future Forecasting
        if self.freq == "MS":
            future_steps = max(1, horizon // 30)
            date_freq = "MS"
        elif self.freq == "W":
            future_steps = max(1, horizon // 7)
            date_freq = "W"
        else:
            future_steps = horizon
            date_freq = "D"

        last_date = daily_df["date"].max()
        future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq=date_freq)[1:]

        future_history = daily_df.copy()
        future_preds = []

        for step_date in future_dates:
            temp_row = pd.DataFrame([{"date": step_date, "sales": np.nan}])
            future_history = pd.concat([future_history, temp_row], ignore_index=True)
            feat_history = self.create_features(future_history)
            
            latest_features = feat_history.iloc[-1:][feature_cols].fillna(0.0)
            pred_step = float(full_reg.predict(latest_features)[0])
            pred_step = max(0.0, pred_step)
            
            future_history.loc[future_history.index[-1], "sales"] = pred_step
            future_preds.append((step_date, pred_step))

        # Residual volatility estimate for bounds
        residuals = np.abs(y_test.values - y_pred)
        margin = float(np.percentile(residuals, 75)) if len(residuals) > 0 else (daily_df["sales"].std() * 0.5)
        margin = max(margin, 10.0)

        # Build combined output dataframe
        historical_records = []
        for _, row in daily_df.iterrows():
            historical_records.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "is_historical": 1,
                "actual_sales": round(float(row["sales"]), 2),
                "predicted_sales": round(float(row["sales"]), 2),
                "lower_bound": round(float(row["sales"]), 2),
                "upper_bound": round(float(row["sales"]), 2)
            })

        future_records = []
        for f_date, f_pred in future_preds:
            future_records.append({
                "date": f_date.strftime("%Y-%m-%d"),
                "is_historical": 0,
                "actual_sales": None,
                "predicted_sales": round(f_pred, 2),
                "lower_bound": round(max(0.0, f_pred - margin), 2),
                "upper_bound": round(f_pred + margin, 2)
            })

        combined_forecast_df = pd.DataFrame(historical_records + future_records)
        return combined_forecast_df, metrics, backtest_df
