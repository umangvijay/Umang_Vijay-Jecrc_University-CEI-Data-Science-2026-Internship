"""
Data Loader — Handles CSV, Excel (.xlsx), and JSON file uploads.

Provides validation, error handling, and user-friendly messages
for all supported data formats.
"""

import pandas as pd
import json
import io
from typing import Tuple, Optional


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load an uploaded file into a Pandas DataFrame.

    Supports CSV, Excel (.xlsx/.xls), and JSON formats.
    Returns a tuple of (DataFrame or None, status_message).

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The file object from Streamlit's file_uploader widget.

    Returns
    -------
    tuple[pd.DataFrame | None, str]
        (DataFrame, success_message) on success,
        (None, error_message) on failure.
    """
    if uploaded_file is None:
        return None, "⚠️ No file uploaded."

    filename = uploaded_file.name.lower()
    file_size_mb = uploaded_file.size / (1024 * 1024)

    # Size limit: 200 MB
    if file_size_mb > 200:
        return None, f"❌ File too large ({file_size_mb:.1f} MB). Maximum is 200 MB."

    try:
        if filename.endswith(".csv"):
            df = _load_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = _load_excel(uploaded_file)
        elif filename.endswith(".json"):
            df = _load_json(uploaded_file)
        else:
            return None, f"❌ Unsupported file format: {filename.split('.')[-1]}. Use CSV, XLSX, or JSON."

        # Validate the loaded DataFrame
        is_valid, validation_msg = validate_dataframe(df)
        if not is_valid:
            return None, validation_msg

        rows, cols = df.shape
        return df, f"✅ Successfully loaded **{uploaded_file.name}** — {rows:,} rows × {cols} columns ({file_size_mb:.2f} MB)"

    except pd.errors.EmptyDataError:
        return None, "❌ The file is empty. Please upload a file with data."
    except pd.errors.ParserError as e:
        return None, f"❌ Failed to parse file: {str(e)}"
    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"


def _load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV file with encoding detection fallback."""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def _load_excel(uploaded_file) -> pd.DataFrame:
    """Load an Excel file using openpyxl engine."""
    return pd.read_excel(uploaded_file, engine="openpyxl")


def _load_json(uploaded_file) -> pd.DataFrame:
    """
    Load a JSON file. Handles both array-of-objects and
    nested structures by normalizing to flat table.
    """
    content = uploaded_file.read()
    uploaded_file.seek(0)

    data = json.loads(content)

    if isinstance(data, list):
        return pd.json_normalize(data)
    elif isinstance(data, dict):
        # Try common patterns: {"data": [...]} or {"records": [...]}
        for key in ["data", "records", "results", "items", "rows"]:
            if key in data and isinstance(data[key], list):
                return pd.json_normalize(data[key])
        # Fallback: normalize the entire dict
        return pd.json_normalize(data)
    else:
        raise ValueError("JSON must contain an array or object at the top level.")


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate a loaded DataFrame for basic sanity checks.

    Returns (True, "") if valid, (False, error_message) if invalid.
    """
    if df is None or df.empty:
        return False, "❌ The loaded data is empty (0 rows)."

    if len(df.columns) == 0:
        return False, "❌ The loaded data has no columns."

    if len(df.columns) > 500:
        return False, f"❌ Too many columns ({len(df.columns)}). Maximum supported is 500."

    if len(df) > 5_000_000:
        return False, f"❌ Too many rows ({len(df):,}). Maximum supported is 5 million."

    # Check for duplicate column names
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        # Auto-fix by appending suffix
        df.columns = pd.io.parsers.readers.ParserBase({"names": df.columns})._maybe_dedup_names(df.columns)

    return True, ""


def get_dataframe_info(df: pd.DataFrame) -> dict:
    """
    Get a summary dictionary of the DataFrame for display.

    Returns a dict with keys: rows, columns, dtypes, memory_mb,
    numeric_cols, categorical_cols, datetime_cols, missing_pct.
    """
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Try to detect datetime columns stored as strings
    for col in categorical_cols:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
            datetime_cols.append(col)
        except (ValueError, TypeError):
            pass

    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_mb": round(memory_mb, 2),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_pct": round(missing_pct, 2),
        "column_names": df.columns.tolist(),
    }
