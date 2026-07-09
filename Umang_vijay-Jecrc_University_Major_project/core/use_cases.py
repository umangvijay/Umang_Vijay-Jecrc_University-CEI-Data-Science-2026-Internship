"""
Use Cases — Prompt templates for each analysis mode.

Each template provides the LLM with structured instructions
for generating Python/Pandas code tailored to the use case.
"""

USE_CASE_TEMPLATES = {
    "sales_dashboard": {
        "name": "📊 Sales Dashboard",
        "description": "Revenue analysis with bar charts, pie charts, and KPI metrics.",
        "prompt": """You are a data analyst. Generate Python code to create a Sales Dashboard.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Identify revenue/sales/amount columns and categorical grouping columns (region, product, category).
2. Create professional visualizations:
   - Bar chart showing revenue by the most relevant grouping (region, product, etc.)
   - If time data exists, add a monthly/quarterly revenue trend line
   - Pie chart for revenue distribution if there are <= 8 categories
3. Calculate KPI metrics: total revenue, average, top performer, growth rate if time data exists.
4. Use Plotly for interactive charts. Save with save_plotly_chart(fig, "name").
5. Print text insights using report_insights({{"key": "value"}}).

OUTPUT FORMAT:
- Charts saved via save_plotly_chart() or save_chart()
- Insights printed via report_insights()
- Print summary statistics to stdout

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },

    "data_quality": {
        "name": "🔍 Data Quality Audit",
        "description": "Missing values, outliers, duplicates detection and cleanliness report.",
        "prompt": """You are a data quality analyst. Generate Python code for a Data Quality Audit.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Missing Values Analysis:
   - Calculate missing count and percentage per column
   - Create a heatmap or bar chart of missing values using seaborn/matplotlib
2. Outlier Detection:
   - Use IQR method on numeric columns
   - Identify and count outliers per column
   - Create box plots for top outlier columns
3. Duplicate Detection:
   - Count exact duplicate rows
   - Identify near-duplicate patterns
4. Data Type Issues:
   - Check for mixed types, inconsistent formats
5. Generate a quality score (0-100) based on completeness and consistency.
6. Save charts with save_chart(fig, "name").
7. Print findings via report_insights().

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },

    "trend_analysis": {
        "name": "📈 Trend Analysis",
        "description": "Time-series trends, rolling averages, and seasonal patterns.",
        "prompt": """You are a time-series analyst. Generate Python code for Trend Analysis.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Identify datetime and numeric columns suitable for trend analysis.
2. Parse dates and sort chronologically.
3. Create visualizations:
   - Main trend line with actual values
   - Rolling average (7-day or 30-day depending on data granularity)
   - Highlight peaks and troughs
4. Calculate trend statistics:
   - Overall direction (increasing/decreasing/stable)
   - Rate of change
   - Volatility (standard deviation)
   - Period-over-period comparison
5. If seasonal patterns exist, note them.
6. Use Plotly for interactive charts. Save with save_plotly_chart(fig, "name").
7. Print insights via report_insights().

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },

    "cohort_analysis": {
        "name": "👥 Cohort Analysis",
        "description": "Customer segmentation by spend, activity, or behavior patterns.",
        "prompt": """You are a customer analytics expert. Generate Python code for Cohort Analysis.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Identify customer/user identifier and relevant metric columns.
2. Segment customers into cohorts based on:
   - Spending levels (quartiles or custom thresholds)
   - Activity frequency if applicable
   - Join date if time data exists
3. Create visualizations:
   - Grouped bar chart showing cohort sizes
   - Heatmap showing cohort metrics over time (if time data exists)
   - Box plot comparing cohort distributions
4. Calculate cohort statistics:
   - Size and percentage of each cohort
   - Average metrics per cohort
   - Key differentiators between cohorts
5. Use matplotlib/seaborn for charts. Save with save_chart(fig, "name").
6. Print insights via report_insights().

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },

    "predictive_forecast": {
        "name": "🔮 Predictive Forecasting",
        "description": "Linear Regression + ARIMA time-series forecasting.",
        "prompt": """You are a predictive analytics expert. Generate Python code for forecasting.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Identify the time column and target numeric column for forecasting.
2. Parse dates and ensure chronological ordering.
3. Apply TWO forecasting methods:

   METHOD 1 — Linear Regression (scikit-learn):
   - Convert dates to ordinal numbers
   - Fit LinearRegression
   - Predict next 30 data points
   - Calculate R² score

   METHOD 2 — ARIMA (statsmodels):
   - Fit ARIMA model (try order=(1,1,1), fallback to (0,1,1) on error)
   - Generate forecast with confidence intervals
   - If ARIMA fails, skip gracefully and note it

4. Create a combined visualization:
   - Historical data as solid line
   - Linear regression forecast as dashed line
   - ARIMA forecast as dotted line (if available)
   - Confidence intervals as shaded area
5. Report metrics: R², RMSE, forecast summary.
6. Use matplotlib for charts. Save with save_chart(fig, "name").
7. Print insights via report_insights().

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },

    "ad_hoc": {
        "name": "💬 Ad-hoc Query",
        "description": "Free-form analysis based on any plain English question.",
        "prompt": """You are an expert data analyst. Generate Python code to answer the user's question.

DATASET CONTEXT:
{schema_context}

USER QUERY: {query}

REQUIREMENTS:
1. Analyze the query and determine the best approach.
2. Write clean, efficient Pandas code to answer the question.
3. Create at least one relevant visualization (bar, line, scatter, pie, heatmap).
4. Provide text insights summarizing the findings.
5. Use Plotly or matplotlib as appropriate. Save charts with save_plotly_chart() or save_chart().
6. Print insights via report_insights().
7. Handle edge cases (missing data, wrong column names) gracefully.

CODE ONLY — no markdown, no explanations. Just valid Python.""",
    },
}


def get_use_case_prompt(use_case: str, schema_context: str, query: str) -> str:
    """Get the formatted prompt for a given use case."""
    template = USE_CASE_TEMPLATES.get(use_case, USE_CASE_TEMPLATES["ad_hoc"])
    return template["prompt"].format(schema_context=schema_context, query=query)


def get_use_case_names() -> dict:
    """Get a dict of use_case_key -> display_name."""
    return {key: val["name"] for key, val in USE_CASE_TEMPLATES.items()}


def get_use_case_descriptions() -> dict:
    """Get a dict of use_case_key -> description."""
    return {key: val["description"] for key, val in USE_CASE_TEMPLATES.items()}
