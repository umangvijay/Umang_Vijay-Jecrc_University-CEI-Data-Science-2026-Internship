"""
Schema Analyzer — Auto-generates a comprehensive dataset summary upon upload.

Uses Pandas profiling + Gemini to produce a natural language summary
of the dataset's structure, quality, and suggested analysis approaches.
"""

import pandas as pd
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI


class SchemaAnalyzer:
    """Generates automatic dataset schema summaries using Pandas + Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = None
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.3,
            )

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive schema report for the DataFrame.

        Returns a dict with statistical summary, column info,
        quality metrics, and AI-generated natural language summary.
        """
        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "missing_values": {},
            "numeric_stats": {},
            "categorical_stats": {},
            "datetime_cols": [],
            "quality_score": 0.0,
            "suggested_use_cases": [],
            "ai_summary": "",
        }

        # ── Missing Values Analysis ──
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report["missing_values"] = {
            col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }
        report["missing_pct"] = round(
            (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2
        )

        # ── Numeric Column Statistics ──
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        report["numeric_cols"] = numeric_cols
        if numeric_cols:
            stats = df[numeric_cols].describe().round(3).to_dict()
            report["numeric_stats"] = stats

        # ── Categorical Column Statistics ──
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        report["categorical_cols"] = cat_cols
        for col in cat_cols[:20]:  # Limit to first 20
            vc = df[col].value_counts()
            report["categorical_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": vc.head(5).to_dict(),
                "sample": df[col].dropna().head(3).tolist(),
            }

        # ── DateTime Detection ──
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in cat_cols:
            if col not in datetime_cols:
                try:
                    pd.to_datetime(df[col].dropna().head(20), errors="raise")
                    datetime_cols.append(col)
                except (ValueError, TypeError):
                    pass
        report["datetime_cols"] = datetime_cols

        # ── Data Quality Score ──
        completeness = 1.0 - (report["missing_pct"] / 100)
        uniqueness = 1.0 - (df.duplicated().sum() / max(len(df), 1))
        report["duplicate_rows"] = int(df.duplicated().sum())
        report["quality_score"] = round((completeness * 0.6 + uniqueness * 0.4) * 100, 1)

        # ── Suggest Use Cases ──
        report["suggested_use_cases"] = self._suggest_use_cases(
            numeric_cols, cat_cols, datetime_cols, df
        )

        # ── AI Summary ──
        if self.llm:
            report["ai_summary"] = self._generate_ai_summary(report, df)

        return report

    def _suggest_use_cases(self, numeric_cols, cat_cols, datetime_cols, df) -> list:
        """Suggest analysis use cases based on column patterns."""
        suggestions = []

        # Check for revenue/sales columns
        revenue_keywords = ["revenue", "sales", "amount", "price", "total", "cost", "profit"]
        has_revenue = any(
            any(kw in col.lower() for kw in revenue_keywords)
            for col in numeric_cols
        )
        has_region = any(
            any(kw in col.lower() for kw in ["region", "state", "country", "city", "location", "area"])
            for col in cat_cols
        )

        if has_revenue:
            suggestions.append("📊 Sales Dashboard — Revenue analysis with bar/pie charts")
        if has_revenue and has_region:
            suggestions.append("🗺️ Regional Sales — Revenue breakdown by geography")

        # Time-series detection
        if datetime_cols:
            suggestions.append("📈 Trend Analysis — Time-series trends and patterns")
            suggestions.append("🔮 Predictive Forecasting — Forecast future values")

        # Customer/cohort detection
        customer_keywords = ["customer", "user", "client", "member", "subscriber"]
        has_customer = any(
            any(kw in col.lower() for kw in customer_keywords)
            for col in df.columns
        )
        if has_customer:
            suggestions.append("👥 Cohort Analysis — Customer segmentation")

        # Always available
        suggestions.append("🔍 Data Quality Audit — Missing values, outliers, duplicates")
        suggestions.append("💬 Ad-hoc Query — Ask any question about your data")

        return suggestions

    def _generate_ai_summary(self, report: dict, df: pd.DataFrame) -> str:
        """Generate a natural language summary using Gemini."""
        try:
            sample = df.head(5).to_string()
            prompt = f"""Analyze this dataset and provide a concise 3-4 sentence summary.

Dataset info:
- Rows: {report['rows']}, Columns: {report['columns']}
- Column names: {', '.join(report['column_names'][:20])}
- Numeric columns: {', '.join(report['numeric_cols'][:10])}
- Categorical columns: {', '.join(report['categorical_cols'][:10])}
- DateTime columns: {', '.join(report['datetime_cols'][:5])}
- Missing data: {report['missing_pct']}%
- Duplicate rows: {report.get('duplicate_rows', 0)}

Sample rows:
{sample}

Provide a brief, professional summary of what this dataset contains,
its quality, and what analyses would be most valuable. Keep it under 100 words."""

            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Auto-summary unavailable: {str(e)}"
