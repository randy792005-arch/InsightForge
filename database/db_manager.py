"""
InsightForge SQLite Database Manager
Provides persistent relational storage for raw data, clean data,
analytics metrics, forecast series, decision recommendations, and pipeline audit logs.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd


class DatabaseManager:
    """
    Manages SQLite database connections, schema setup, data ingestion,
    analytical caching, and multi-agent pipeline logging.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(curr_dir, "retail.db")
        else:
            self.db_path = db_path
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a thread-safe sqlite3 connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize all database tables with appropriate constraints and indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Raw Retail Ingestion Records
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_retail_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                filename TEXT,
                row_index INTEGER,
                raw_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Cleaned Retail Transactions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS clean_retail_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                date TEXT NOT NULL,
                product_id TEXT,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit_price REAL NOT NULL,
                total_sales REAL NOT NULL,
                current_stock REAL DEFAULT 0,
                region TEXT DEFAULT 'Global',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Index for fast aggregation and filtering
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_clean_dataset_date 
            ON clean_retail_data (dataset_id, date)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_clean_dataset_category 
            ON clean_retail_data (dataset_id, category)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_clean_dataset_product 
            ON clean_retail_data (dataset_id, product_name)
            """)

            # 3. Analytics Summary & KPIs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_kpis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                kpi_key TEXT NOT NULL,
                kpi_value REAL NOT NULL,
                kpi_label TEXT NOT NULL,
                kpi_group TEXT DEFAULT 'general',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(dataset_id, kpi_key)
            )
            """)

            # 4. Forecast Output Records
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                target_name TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                predicted_sales REAL NOT NULL,
                lower_bound REAL,
                upper_bound REAL,
                is_historical INTEGER DEFAULT 0,
                actual_sales REAL,
                mae REAL,
                rmse REAL,
                mape REAL,
                r2 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 5. Inventory Recommendations & Decision Support
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                trigger_metric TEXT NOT NULL,
                action_recommended TEXT NOT NULL,
                expected_impact TEXT,
                current_stock REAL,
                daily_velocity REAL,
                reorder_qty REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 6. Multi-Agent Pipeline Execution Logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                execution_time_sec REAL NOT NULL,
                details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()

    def save_raw_data(self, df: pd.DataFrame, upload_id: str, filename: str = "uploaded_file", max_rows: int = 50000) -> int:
        """Save raw uploaded records as JSON rows for audit and replay."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM raw_retail_data WHERE upload_id = ?", (upload_id,))
            
            # Fast vectorized conversion avoiding row.to_json() index collision
            df_slice = df.head(max_rows)
            records = df_slice.to_dict(orient="records")
            rows = [
                (upload_id, filename, int(idx), json.dumps(rec, default=str))
                for idx, rec in enumerate(records)
            ]
            
            cursor.executemany("""
            INSERT INTO raw_retail_data (upload_id, filename, row_index, raw_json)
            VALUES (?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)

    def save_clean_data(self, df: pd.DataFrame, dataset_id: str) -> int:
        """Save standardized cleaned data rows into SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM clean_retail_data WHERE dataset_id = ?", (dataset_id,))
            
            # Prepare columns
            df_to_save = df.copy()
            df_to_save["dataset_id"] = dataset_id
            
            required_cols = ["date", "product_name", "category", "quantity", "unit_price", "total_sales"]
            for col in required_cols:
                if col not in df_to_save.columns:
                    raise ValueError(f"Missing required clean column: {col}")
            
            if "product_id" not in df_to_save.columns:
                df_to_save["product_id"] = df_to_save["product_name"].astype(str)
            if "current_stock" not in df_to_save.columns:
                df_to_save["current_stock"] = 0.0
            if "region" not in df_to_save.columns:
                df_to_save["region"] = "Global"

            # Convert date to string YYYY-MM-DD
            if pd.api.types.is_datetime64_any_dtype(df_to_save["date"]):
                df_to_save["date"] = df_to_save["date"].dt.strftime("%Y-%m-%d")
            else:
                df_to_save["date"] = df_to_save["date"].astype(str)

            cols = ["dataset_id", "date", "product_id", "product_name", "category", 
                    "quantity", "unit_price", "total_sales", "current_stock", "region"]
            records = df_to_save[cols].to_records(index=False).tolist()

            cursor.executemany("""
            INSERT INTO clean_retail_data 
            (dataset_id, date, product_id, product_name, category, quantity, unit_price, total_sales, current_stock, region)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            return len(records)

    def load_clean_data(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        """Load clean data from SQLite as a Pandas DataFrame."""
        with self.get_connection() as conn:
            if dataset_id:
                query = "SELECT * FROM clean_retail_data WHERE dataset_id = ? ORDER BY date ASC"
                df = pd.read_sql_query(query, conn, params=(dataset_id,))
            else:
                query = "SELECT * FROM clean_retail_data ORDER BY date ASC"
                df = pd.read_sql_query(query, conn)
            
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    def save_analytics_kpis(self, kpis_dict: Dict[str, Any], dataset_id: str):
        """Upsert analytics KPI key-values into database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for key, val in kpis_dict.items():
                if isinstance(val, (int, float)):
                    label = key.replace("_", " ").title()
                    cursor.execute("""
                    INSERT INTO analytics_kpis (dataset_id, kpi_key, kpi_value, kpi_label, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(dataset_id, kpi_key) DO UPDATE SET
                    kpi_value=excluded.kpi_value,
                    kpi_label=excluded.kpi_label,
                    updated_at=CURRENT_TIMESTAMP
                    """, (dataset_id, key, float(val), label))
            conn.commit()

    def load_analytics_kpis(self, dataset_id: Optional[str] = None) -> Dict[str, float]:
        """Load latest cached KPIs as dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if dataset_id:
                cursor.execute("SELECT kpi_key, kpi_value FROM analytics_kpis WHERE dataset_id = ?", (dataset_id,))
            else:
                cursor.execute("SELECT kpi_key, kpi_value FROM analytics_kpis ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return {row["kpi_key"]: row["kpi_value"] for row in rows}

    def save_forecast_results(
        self, 
        forecast_df: pd.DataFrame, 
        metrics_dict: Dict[str, float], 
        model_type: str, 
        dataset_id: str, 
        target_name: str = "Overall"
    ):
        """Store forecast series predictions and evaluation metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            DELETE FROM forecast_records 
            WHERE dataset_id = ? AND model_type = ? AND target_name = ?
            """, (dataset_id, model_type, target_name))

            mae = metrics_dict.get("mae", 0.0)
            rmse = metrics_dict.get("rmse", 0.0)
            mape = metrics_dict.get("mape", 0.0)
            r2 = metrics_dict.get("r2", 0.0)

            rows = []
            for _, row in forecast_df.iterrows():
                date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                pred_sales = float(row.get("predicted_sales", row.get("yhat", 0.0)))
                lower_b = float(row.get("lower_bound", row.get("yhat_lower", pred_sales * 0.9)))
                upper_b = float(row.get("upper_bound", row.get("yhat_upper", pred_sales * 1.1)))
                is_hist = int(row.get("is_historical", 0))
                act_sales = float(row["actual_sales"]) if ("actual_sales" in row and pd.notnull(row["actual_sales"])) else None

                rows.append((
                    dataset_id, model_type, target_name, date_str,
                    pred_sales, lower_b, upper_b, is_hist, act_sales,
                    mae, rmse, mape, r2
                ))

            cursor.executemany("""
            INSERT INTO forecast_records 
            (dataset_id, model_type, target_name, forecast_date, predicted_sales, lower_bound, upper_bound, is_historical, actual_sales, mae, rmse, mape, r2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def load_forecast_results(self, dataset_id: Optional[str] = None, model_type: Optional[str] = None) -> pd.DataFrame:
        """Load stored forecast records."""
        with self.get_connection() as conn:
            params = []
            query = "SELECT * FROM forecast_records"
            conditions = []
            if dataset_id:
                conditions.append("dataset_id = ?")
                params.append(dataset_id)
            if model_type:
                conditions.append("model_type = ?")
                params.append(model_type)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY forecast_date ASC"

            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty and "forecast_date" in df.columns:
                df["forecast_date"] = pd.to_datetime(df["forecast_date"])
            return df

    def save_recommendations(self, rec_df: pd.DataFrame, dataset_id: str):
        """Save inventory and business recommendations into SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recommendations WHERE dataset_id = ?", (dataset_id,))

            rows = []
            for _, row in rec_df.iterrows():
                rows.append((
                    dataset_id,
                    str(row.get("product_name", "Unknown")),
                    str(row.get("category", "General")),
                    str(row.get("priority", "MEDIUM")),
                    str(row.get("issue_type", "General")),
                    str(row.get("trigger_metric", "N/A")),
                    str(row.get("action_recommended", "")),
                    str(row.get("expected_impact", "")),
                    float(row.get("current_stock", 0.0)),
                    float(row.get("daily_velocity", 0.0)),
                    float(row.get("reorder_qty", 0.0))
                ))

            cursor.executemany("""
            INSERT INTO recommendations 
            (dataset_id, product_name, category, priority, issue_type, trigger_metric, action_recommended, expected_impact, current_stock, daily_velocity, reorder_qty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def load_recommendations(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        """Load stored recommendations."""
        with self.get_connection() as conn:
            if dataset_id:
                query = "SELECT * FROM recommendations WHERE dataset_id = ? ORDER BY CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END, current_stock ASC"
                return pd.read_sql_query(query, conn, params=(dataset_id,))
            else:
                query = "SELECT * FROM recommendations ORDER BY CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END, current_stock ASC"
                return pd.read_sql_query(query, conn)

    def log_agent_step(
        self, 
        session_id: str, 
        agent_name: str, 
        status: str, 
        message: str, 
        execution_time_sec: float, 
        details: Optional[Dict[str, Any]] = None
    ):
        """Record agent workflow step for audit, UI status indicator, and academic verification."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            details_json = json.dumps(details, default=str) if details else "{}"
            cursor.execute("""
            INSERT INTO pipeline_logs (session_id, agent_name, status, message, execution_time_sec, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, agent_name, status, message, execution_time_sec, details_json))
            conn.commit()

    def get_pipeline_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch pipeline execution logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("""
                SELECT session_id, agent_name, status, message, execution_time_sec, details_json, created_at
                FROM pipeline_logs WHERE session_id = ? ORDER BY id ASC
                """, (session_id,))
            else:
                cursor.execute("""
                SELECT session_id, agent_name, status, message, execution_time_sec, details_json, created_at
                FROM pipeline_logs ORDER BY id DESC LIMIT 50
                """)
            rows = cursor.fetchall()
            logs = []
            for row in rows:
                logs.append({
                    "session_id": row["session_id"],
                    "agent_name": row["agent_name"],
                    "status": row["status"],
                    "message": row["message"],
                    "execution_time_sec": row["execution_time_sec"],
                    "details": json.loads(row["details_json"]) if row["details_json"] else {},
                    "created_at": row["created_at"]
                })
            return logs

    def clear_dataset_data(self, dataset_id: str):
        """Clear all records associated with a specific dataset id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM clean_retail_data WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM analytics_kpis WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM forecast_records WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM recommendations WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM raw_retail_data WHERE upload_id = ?", (dataset_id,))
            conn.commit()
