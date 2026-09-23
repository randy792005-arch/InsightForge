"""
InsightForge Preprocessing and Column Normalization Utilities
Handles automatic semantic column detection, robust date parsing,
missing data calculation, and data quality scoring for retail datasets.
"""

import re
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd


# Comprehensive semantic dictionary for retail columns
COLUMN_KEYWORDS = {
    "date": [
        "date", "order_date", "invoice_date", "transaction_date", "sale_date", 
        "sales_date", "datetime", "timestamp", "order date", "invoice date", "transaction date",
        "purchasedate", "order_day", "day", "invoicedate", "trans_date"
    ],
    "product_name": [
        "product_name", "product", "item", "item_name", "item_description", "description",
        "sku_name", "product_title", "product name", "item name", "product description",
        "item description", "product_desc", "merchandise", "goods"
    ],
    "product_id": [
        "product_id", "item_id", "sku", "sku_id", "product_code", "item_code", "code",
        "product id", "item id", "sku id", "productcode", "itemcode", "barcode"
    ],
    "category": [
        "category", "product_category", "department", "dept", "product_type", "item_category",
        "segment", "product category", "item category", "dept_name", "group", "class",
        "merchandise_group", "family"
    ],
    "quantity": [
        "quantity", "qty", "units", "units_sold", "quantity_sold", "volume", "count",
        "items_sold", "quantity sold", "units sold", "order_qty", "sales_volume", "pieces"
    ],
    "unit_price": [
        "unit_price", "price", "selling_price", "rate", "retail_price", "unit_cost",
        "product_price", "unit price", "selling price", "retail price", "price_per_unit", "mrp"
    ],
    "total_sales": [
        "total_sales", "sales", "revenue", "total_amount", "total_price", "amount",
        "sales_amount", "turnover", "gross_sales", "total sales", "total amount",
        "sales amount", "net_sales", "order_value", "total_revenue", "line_total"
    ],
    "current_stock": [
        "current_stock", "stock", "inventory", "stock_level", "stock_quantity", "qty_in_stock",
        "quantity_on_hand", "inventory_level", "units_in_stock", "current stock", "stock level",
        "on_hand", "available_stock", "stock_count", "qoh"
    ],
    "region": [
        "region", "store", "store_id", "location", "city", "country", "branch", "state",
        "store_location", "territory", "outlet", "zone", "warehouse"
    ]
}

MANDATORY_CONCEPTS = ["date", "product_name"]  # Must have date + product
NUMERIC_CONCEPTS = ["quantity", "unit_price", "total_sales", "current_stock"]


def normalize_str(s: str) -> str:
    """Clean and normalize column header string."""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def detect_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Automatically map raw DataFrame columns to standard retail schema concepts.
    
    Returns:
        matched_mapping: dict mapping standard concept -> raw column name
        missing_mandatory: list of required concepts that could not be detected
        missing_optional: list of optional concepts not detected
    """
    raw_cols = list(df.columns)
    normalized_raw_cols = {col: normalize_str(col) for col in raw_cols}
    
    matched_mapping = {}
    used_cols = set()

    # Exact normalized matching first
    for concept, keywords in COLUMN_KEYWORDS.items():
        for raw_col, norm_col in normalized_raw_cols.items():
            if raw_col in used_cols:
                continue
            if norm_col in [normalize_str(k) for k in keywords]:
                matched_mapping[concept] = raw_col
                used_cols.add(raw_col)
                break

    # Substring / partial matching for unassigned concepts
    for concept, keywords in COLUMN_KEYWORDS.items():
        if concept in matched_mapping:
            continue
        for raw_col, norm_col in normalized_raw_cols.items():
            if raw_col in used_cols:
                continue
            for kw in keywords:
                norm_kw = normalize_str(kw)
                if norm_kw in norm_col or norm_col in norm_kw:
                    matched_mapping[concept] = raw_col
                    used_cols.add(raw_col)
                    break
            if concept in matched_mapping:
                break

    # Validate mandatory concepts
    missing_mandatory = []
    
    # Check Date
    if "date" not in matched_mapping:
        missing_mandatory.append("Date (e.g. Order Date / Transaction Date)")
    
    # Check Product / Category
    if "product_name" not in matched_mapping and "category" not in matched_mapping:
        missing_mandatory.append("Product Name or Category")

    # Check Sales / Quantity
    if "total_sales" not in matched_mapping and "quantity" not in matched_mapping:
        missing_mandatory.append("Sales Amount or Quantity Sold")

    # Optional columns
    missing_optional = []
    for concept in ["category", "product_id", "unit_price", "current_stock", "region"]:
        if concept not in matched_mapping:
            missing_optional.append(concept)

    return matched_mapping, missing_mandatory, missing_optional


def parse_dates_robust(series: pd.Series) -> pd.Series:
    """
    Parses datetime strings safely across multiple date formats.
    """
    # First attempt: pandas default parser with dayfirst handling
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notnull().sum() / max(1, len(series)) > 0.8:
            return parsed
    except Exception:
        pass

    # Try common formats explicitly
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S",
        "%Y%m%d", "%b %d, %Y", "%d %b %Y"
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notnull().sum() / max(1, len(series)) > 0.8:
                return parsed
        except Exception:
            continue

    return pd.to_datetime(series, errors="coerce")


def fill_missing_sales_quantities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cross-computes missing sales, quantity, and unit_price fields
    based on the arithmetic relation: total_sales = quantity * unit_price
    """
    logs = []
    df = df.copy()

    # 1. total_sales missing, quantity & unit_price present
    if "total_sales" in df.columns and "quantity" in df.columns and "unit_price" in df.columns:
        mask = df["total_sales"].isna() | (df["total_sales"] == 0)
        valid_inputs = mask & df["quantity"].notnull() & df["unit_price"].notnull() & (df["quantity"] > 0)
        recalculated_count = valid_inputs.sum()
        if recalculated_count > 0:
            df.loc[valid_inputs, "total_sales"] = (
                df.loc[valid_inputs, "quantity"] * df.loc[valid_inputs, "unit_price"]
            )
            logs.append(f"Calculated missing Total Sales for {recalculated_count} records from Quantity * Unit Price.")

    # 2. unit_price missing, total_sales & quantity present
    if "unit_price" in df.columns and "total_sales" in df.columns and "quantity" in df.columns:
        mask = df["unit_price"].isna() | (df["unit_price"] == 0)
        valid_inputs = mask & df["total_sales"].notnull() & df["quantity"].notnull() & (df["quantity"] > 0)
        recalculated_count = valid_inputs.sum()
        if recalculated_count > 0:
            df.loc[valid_inputs, "unit_price"] = (
                df.loc[valid_inputs, "total_sales"] / df.loc[valid_inputs, "quantity"]
            )
            logs.append(f"Derived missing Unit Price for {recalculated_count} records from Total Sales / Quantity.")

    # 3. quantity missing, total_sales & unit_price present
    if "quantity" in df.columns and "total_sales" in df.columns and "unit_price" in df.columns:
        mask = df["quantity"].isna() | (df["quantity"] == 0)
        valid_inputs = mask & df["total_sales"].notnull() & df["unit_price"].notnull() & (df["unit_price"] > 0)
        recalculated_count = valid_inputs.sum()
        if recalculated_count > 0:
            df.loc[valid_inputs, "quantity"] = (
                df.loc[valid_inputs, "total_sales"] / df.loc[valid_inputs, "unit_price"]
            ).round()
            logs.append(f"Derived missing Quantity for {recalculated_count} records from Total Sales / Unit Price.")

    return df, logs


def standardize_dataframe(df: pd.DataFrame, col_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Maps detected columns to the standardized internal schema, applies data types,
    and fills missing columns with sensible retail defaults.
    """
    df_clean = pd.DataFrame()

    # Invert mapping: raw_col -> standard_concept
    inv_map = {v: k for k, v in col_mapping.items()}
    df_renamed = df.rename(columns=inv_map)

    # Date
    if "date" in df_renamed.columns:
        df_clean["date"] = parse_dates_robust(df_renamed["date"])
    else:
        # Fallback to current date range
        df_clean["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="D")

    # Product Name
    if "product_name" in df_renamed.columns:
        df_clean["product_name"] = df_renamed["product_name"].astype(str).str.strip().str.title()
    elif "category" in df_renamed.columns:
        df_clean["product_name"] = df_renamed["category"].astype(str).str.strip().str.title() + " Item"
    else:
        df_clean["product_name"] = "Retail Product " + df.index.astype(str)

    # Product ID
    if "product_id" in df_renamed.columns:
        df_clean["product_id"] = df_renamed["product_id"].astype(str).str.strip()
    else:
        # Generate SKU from product name initials and hash
        df_clean["product_id"] = df_clean["product_name"].apply(
            lambda name: "SKU-" + "".join([w[0] for w in name.split()[:3]]).upper() + "-" + str(abs(hash(name)) % 1000).zfill(3)
        )

    # Category
    if "category" in df_renamed.columns:
        df_clean["category"] = df_renamed["category"].astype(str).str.strip().str.title()
    else:
        df_clean["category"] = "General Retail"

    # Numeric Columns: Quantity, Unit Price, Total Sales
    if "quantity" in df_renamed.columns:
        df_clean["quantity"] = pd.to_numeric(df_renamed["quantity"], errors="coerce").fillna(1.0)
    else:
        df_clean["quantity"] = 1.0

    if "unit_price" in df_renamed.columns:
        df_clean["unit_price"] = pd.to_numeric(df_renamed["unit_price"], errors="coerce")
    else:
        df_clean["unit_price"] = np.nan

    if "total_sales" in df_renamed.columns:
        df_clean["total_sales"] = pd.to_numeric(df_renamed["total_sales"], errors="coerce")
    else:
        df_clean["total_sales"] = np.nan

    # Cross compute missing sales / quantities / prices
    df_clean, _ = fill_missing_sales_quantities(df_clean)

    # Default fallbacks if still NaN
    if df_clean["total_sales"].isna().all() and df_clean["unit_price"].isna().all():
        df_clean["unit_price"] = 25.0
        df_clean["total_sales"] = df_clean["quantity"] * 25.0
    elif df_clean["total_sales"].isna().all():
        df_clean["total_sales"] = df_clean["quantity"] * df_clean["unit_price"].fillna(20.0)
    elif df_clean["unit_price"].isna().all():
        df_clean["unit_price"] = df_clean["total_sales"] / df_clean["quantity"].replace(0, 1)

    df_clean["unit_price"] = df_clean["unit_price"].fillna(10.0)
    df_clean["total_sales"] = df_clean["total_sales"].fillna(df_clean["quantity"] * df_clean["unit_price"])

    # Stock / Inventory
    if "current_stock" in df_renamed.columns:
        df_clean["current_stock"] = pd.to_numeric(df_renamed["current_stock"], errors="coerce").fillna(0.0)
    else:
        # Estimate stock per product based on average daily volume * 14-30 days buffer
        product_sales_vol = df_clean.groupby("product_name")["quantity"].transform("sum")
        total_days = max(1, (df_clean["date"].max() - df_clean["date"].min()).days) if df_clean["date"].notnull().any() else 30
        daily_rate = (product_sales_vol / total_days).clip(lower=0.5)
        # Random seed based on product name for deterministic baseline stock
        df_clean["current_stock"] = (daily_rate * 21).round(0)

    # Region
    if "region" in df_renamed.columns:
        df_clean["region"] = df_renamed["region"].astype(str).str.strip().str.title()
    else:
        df_clean["region"] = "National"

    return df_clean


def compute_data_quality_score(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates a comprehensive Data Quality Score (0 - 100%) and health report.
    Evaluates completeness, duplicate penalty, validity, and date regularity.
    """
    total_raw_cells = max(1, df_raw.shape[0] * df_raw.shape[1])
    missing_cells = df_raw.isna().sum().sum()
    completeness_score = max(0.0, 1.0 - (missing_cells / total_raw_cells)) * 30.0  # 30 pts

    # Duplicate penalty
    duplicate_rows = df_raw.duplicated().sum()
    dup_ratio = duplicate_rows / max(1, len(df_raw))
    uniqueness_score = max(0.0, 1.0 - dup_ratio) * 25.0  # 25 pts

    # Numeric Validity (positive values, price * qty matching)
    valid_numeric_rows = (
        (df_clean["quantity"] > 0) & 
        (df_clean["unit_price"] > 0) & 
        (df_clean["total_sales"] > 0)
    ).sum()
    validity_score = (valid_numeric_rows / max(1, len(df_clean))) * 25.0  # 25 pts

    # Date Regularity / Continuity
    valid_dates = df_clean["date"].notnull().sum()
    date_score = (valid_dates / max(1, len(df_clean))) * 20.0  # 20 pts

    overall_score = round(completeness_score + uniqueness_score + validity_score + date_score, 1)

    grade = "A+" if overall_score >= 90 else "A" if overall_score >= 80 else "B" if overall_score >= 70 else "C" if overall_score >= 60 else "D"

    return {
        "overall_score": overall_score,
        "grade": grade,
        "completeness_score": round(completeness_score, 1),
        "uniqueness_score": round(uniqueness_score, 1),
        "validity_score": round(validity_score, 1),
        "date_score": round(date_score, 1),
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "duplicates_removed": int(duplicate_rows),
        "missing_cells_imputed": int(missing_cells)
    }
