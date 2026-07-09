"""
Insights Formatter — Formats raw analysis output into styled
markdown and Streamlit-compatible display elements.
"""

import re
from typing import List, Dict, Optional


def format_insights(raw_text: str) -> str:
    """
    Format raw text insights into styled markdown for Streamlit.

    Handles:
    - Bullet point normalization
    - Key metric highlighting
    - Section header formatting
    - Number formatting with commas
    """
    if not raw_text or not raw_text.strip():
        return "_No insights generated._"

    lines = raw_text.strip().split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            continue

        # Format section headers (lines ending with colon or starting with ##)
        if line.endswith(":") and len(line) < 80:
            formatted_lines.append(f"\n**{line}**")
            continue

        # Normalize bullet points
        if line.startswith(("- ", "* ", "• ")):
            line = "• " + line[2:]
        elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            pass  # Keep numbered lists as-is

        # Highlight numbers and percentages
        line = _highlight_numbers(line)

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def _highlight_numbers(text: str) -> str:
    """
    Bold significant numbers and percentages in text.
    Avoids re-bolding already bold text.
    """
    # Bold percentages like 45.2% or 100%
    text = re.sub(r'(?<!\*\*)(\d+\.?\d*%)', r'**\1**', text)

    # Bold currency amounts like $1,234.56
    text = re.sub(r'(?<!\*\*)(\$[\d,]+\.?\d*)', r'**\1**', text)

    return text


def format_code_block(code: str, language: str = "python") -> str:
    """
    Wrap code in a syntax-highlighted markdown code block.
    """
    return f"```{language}\n{code.strip()}\n```"


def format_execution_log(iterations: List[Dict]) -> str:
    """
    Format the self-correction execution log for display.

    Each iteration dict should have:
    - attempt: int
    - success: bool
    - error: str (if failed)
    - fix_applied: str (if corrected)
    """
    if not iterations:
        return "_No execution log available._"

    lines = ["### 🔄 Execution Log\n"]

    for i, iteration in enumerate(iterations):
        attempt = iteration.get("attempt", i + 1)
        success = iteration.get("success", False)

        if success:
            lines.append(f"**Attempt {attempt}** — ✅ Success")
        else:
            error = iteration.get("error", "Unknown error")
            lines.append(f"**Attempt {attempt}** — ❌ Error")
            lines.append(f"  > `{_truncate(error, 150)}`")

            fix = iteration.get("fix_applied", "")
            if fix:
                lines.append(f"  > 🔧 Fix: {_truncate(fix, 100)}")

        lines.append("")

    total = len(iterations)
    successes = sum(1 for it in iterations if it.get("success"))
    lines.append(f"**Result:** {successes}/{total} attempts succeeded")

    return "\n".join(lines)


def format_schema_summary(schema_info: dict) -> str:
    """
    Format the auto-generated schema summary as styled markdown.
    """
    lines = [
        "### 📋 Dataset Schema Summary\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Rows** | {schema_info.get('rows', 'N/A'):,} |",
        f"| **Columns** | {schema_info.get('columns', 'N/A')} |",
        f"| **Memory** | {schema_info.get('memory_mb', 'N/A')} MB |",
        f"| **Missing Data** | {schema_info.get('missing_pct', 0)}% |",
        "",
    ]

    # Column breakdown
    numeric = schema_info.get("numeric_cols", [])
    categorical = schema_info.get("categorical_cols", [])
    datetime_cols = schema_info.get("datetime_cols", [])

    if numeric:
        lines.append(f"**📊 Numeric Columns ({len(numeric)}):** {', '.join(numeric[:10])}")
        if len(numeric) > 10:
            lines.append(f"  _...and {len(numeric) - 10} more_")

    if categorical:
        lines.append(f"**📝 Categorical Columns ({len(categorical)}):** {', '.join(categorical[:10])}")
        if len(categorical) > 10:
            lines.append(f"  _...and {len(categorical) - 10} more_")

    if datetime_cols:
        lines.append(f"**📅 Date/Time Columns ({len(datetime_cols)}):** {', '.join(datetime_cols[:5])}")

    return "\n".join(lines)


def create_insight_card(title: str, content: str, icon: str = "💡") -> str:
    """
    Create a styled insight card as markdown.
    """
    return f"""
> {icon} **{title}**
>
> {content}
"""


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
