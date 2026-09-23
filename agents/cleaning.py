"""
InsightForge Data Cleaning Agent (Agent 2)
Responsible for:
- Handling missing values and cross-field imputation
- Removing duplicate records
- Correcting data types and standardizing date formats
- Validating non-negative numeric boundaries
- Calculating comprehensive Data Quality Score
- Persisting clean dataset into SQLite
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from agents.base_agent import BaseAgent
from utils.preprocess import standardize_dataframe, compute_data_quality_score


class DataCleaningAgent(BaseAgent):
    """
    Autonomous agent for retail data cleansing, type normalization, and quality validation.
    """

    def __init__(self, db_manager=None):
        super().__init__(name="Data Cleaning Agent", db_manager=db_manager)

    def _execute(
        self, 
        df_raw: pd.DataFrame, 
        column_mapping: Dict[str, str], 
        session_id: str = "default_session",
        **kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Cleans and standardizes the ingested raw dataframe.
        """
        self.log(f"Starting data cleaning pipeline for {len(df_raw):,} raw records...")

        initial_row_count = len(df_raw)

        # 1. Standardize column names and types
        self.log("Standardizing columns and data types to internal schema...")
        df_clean = standardize_dataframe(df_raw, column_mapping)

        # 2. Handle invalid dates
        invalid_dates_count = df_clean["date"].isna().sum()
        if invalid_dates_count > 0:
            self.log(f"Removing {invalid_dates_count} records with unparseable date values.")
            df_clean = df_clean.dropna(subset=["date"]).reset_index(drop=True)

        # 3. Deduplication
        duplicates_count = df_clean.duplicated().sum()
        if duplicates_count > 0:
            self.log(f"Detected and eliminated {duplicates_count} duplicate record(s).")
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        # 4. Filter invalid non-positive numbers
        invalid_numeric_mask = (
            (df_clean["quantity"] <= 0) | 
            (df_clean["unit_price"] <= 0) | 
            (df_clean["total_sales"] <= 0)
        )
        invalid_numeric_count = invalid_numeric_mask.sum()
        if invalid_numeric_count > 0:
            self.log(f"Filtered out {invalid_numeric_count} records with invalid non-positive sales/quantity.")
            df_clean = df_clean[~invalid_numeric_mask].reset_index(drop=True)

        # 5. Sort chronologically
        df_clean = df_clean.sort_values(by="date").reset_index(drop=True)

        if df_clean.empty:
            raise ValueError("All records were discarded during cleaning due to invalid dates or non-positive values.")

        final_row_count = len(df_clean)
        self.log(f"Cleaned dataset: {final_row_count:,} valid records retained from {initial_row_count:,} raw rows.")

        # 6. Compute Data Quality Score
        quality_report = compute_data_quality_score(df_raw, df_clean)
        self.log(f"Calculated Data Quality Score: {quality_report['overall_score']}% (Grade: {quality_report['grade']}).")

        # 7. Persist Clean Data to SQLite
        saved_count = self.db.save_clean_data(df_clean, dataset_id=session_id)
        self.log(f"Persisted {saved_count:,} cleaned records to SQLite table 'clean_retail_data'.")

        cleaning_summary = {
            "initial_rows": initial_row_count,
            "cleaned_rows": final_row_count,
            "rows_removed": initial_row_count - final_row_count,
            "duplicates_removed": int(duplicates_count),
            "invalid_dates_removed": int(invalid_dates_count),
            "invalid_numerics_removed": int(invalid_numeric_count),
            "quality_score": quality_report["overall_score"],
            "quality_grade": quality_report["grade"],
            "quality_details": quality_report
        }

        status = "SUCCESS" if quality_report["overall_score"] >= 80 else "WARNING"
        status_msg = (
            f"Successfully cleaned dataset: {final_row_count:,} records prepared. "
            f"Data Quality Score: {quality_report['overall_score']}% (Grade: {quality_report['grade']})."
        )

        return {
            "df_clean": df_clean,
            "quality_report": quality_report,
            "cleaning_summary": cleaning_summary
        }, status, status_msg
