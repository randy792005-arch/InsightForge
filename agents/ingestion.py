"""
InsightForge Data Ingestion Agent (Agent 1)
Responsible for:
- Accepting CSV / Excel files
- Validating file structure and format
- Detecting column semantics automatically
- Persisting raw data into SQLite
- Generating ingestion diagnostic reports
"""

import os
import io
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
from agents.base_agent import BaseAgent
from utils.preprocess import detect_columns


class DataIngestionAgent(BaseAgent):
    """
    Autonomous agent responsible for raw retail dataset ingestion, schema detection, and raw persistence.
    """

    def __init__(self, db_manager=None):
        super().__init__(name="Data Ingestion Agent", db_manager=db_manager)

    def _execute(
        self, 
        file_source: Union[str, io.BytesIO, pd.DataFrame], 
        filename: str = "dataset.csv",
        session_id: str = "default_session",
        **kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Ingests the dataset, detects columns, and stores raw records in SQLite.
        """
        self.log(f"Reading dataset source: {filename}...")

        # 1. Load into Pandas DataFrame
        if isinstance(file_source, pd.DataFrame):
            df_raw = file_source.copy()
        elif isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise FileNotFoundError(f"File not found: {file_source}")
            if file_source.endswith(".csv"):
                # Try common encodings
                try:
                    df_raw = pd.read_csv(file_source, encoding="utf-8")
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(file_source, encoding="latin-1")
            elif file_source.endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(file_source)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel (.xlsx/.xls) file.")
        elif isinstance(file_source, io.BytesIO) or hasattr(file_source, "read"):
            if filename.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(file_source, encoding="utf-8")
                except UnicodeDecodeError:
                    file_source.seek(0)
                    df_raw = pd.read_csv(file_source, encoding="latin-1")
            elif filename.endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(file_source)
            else:
                # Default attempt as CSV
                df_raw = pd.read_csv(file_source)
        else:
            raise ValueError(f"Invalid file source type: {type(file_source)}")

        self.log(f"File parsed successfully: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns detected.")

        if df_raw.empty:
            raise ValueError("The uploaded dataset is empty (0 rows). Please upload a dataset with valid records.")

        # 2. Semantic Column Detection & Schema Validation
        matched_mapping, missing_mandatory, missing_optional = detect_columns(df_raw)
        self.log(f"Detected column mapping: {matched_mapping}")

        if missing_mandatory:
            err_msg = f"Dataset is missing required columns: {', '.join(missing_mandatory)}. Please ensure your dataset includes dates, products/categories, and sales/quantities."
            self.log(f"Validation failure: {err_msg}")
            raise ValueError(err_msg)

        if missing_optional:
            self.log(f"Optional columns not present (will be intelligently defaulted): {', '.join(missing_optional)}")

        # 3. Save Raw Data into SQLite
        rows_saved = self.db.save_raw_data(df_raw, upload_id=session_id, filename=filename)
        self.log(f"Saved {rows_saved} raw records to SQLite (upload_id={session_id}).")

        # 4. Generate Ingestion Metadata
        summary = {
            "total_rows": int(len(df_raw)),
            "total_columns": int(len(df_raw.columns)),
            "column_names": list(df_raw.columns),
            "column_mapping": matched_mapping,
            "missing_optional": missing_optional,
            "memory_usage_kb": round(df_raw.memory_usage(deep=True).sum() / 1024, 2),
            "preview_data": df_raw.head(10).to_dict(orient="records")
        }

        status = "WARNING" if missing_optional else "SUCCESS"
        status_msg = (
            f"Successfully ingested {len(df_raw):,} records and {len(df_raw.columns)} columns. "
            f"Mapped {len(matched_mapping)} retail concepts."
        )
        if missing_optional:
            status_msg += f" Note: {len(missing_optional)} optional field(s) defaulted ({', '.join(missing_optional)})."

        return {
            "df_raw": df_raw,
            "column_mapping": matched_mapping,
            "metadata": summary,
            "filename": filename
        }, status, status_msg
