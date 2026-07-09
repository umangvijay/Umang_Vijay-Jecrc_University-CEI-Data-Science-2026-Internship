"""
Chart Renderer — Renders Plotly and Matplotlib charts in Streamlit.

Applies consistent premium styling across all chart types
and handles both interactive (Plotly) and static (Matplotlib) outputs.
Uses the Fluid Analyst design system palette.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from typing import Optional

# Use non-interactive backend for Matplotlib in subprocess
matplotlib.use("Agg")

# ── Fluid Analyst Color Palette ─────────────────────────────────────
COLORS = {
    "primary": "#00F0FF",
    "secondary": "#D200FF",
    "accent": "#00E676",
    "warning": "#FFD93D",
    "info": "#6EC6FF",
    "dark_bg": "#0A1128",
    "card_bg": "#16203C",
    "surface_hover": "#1E2B4D",
    "text": "#F8F9FA",
    "text_muted": "#6B7A9C",
    "gradient": ["#00F0FF", "#D200FF", "#00E676", "#FFD93D", "#6EC6FF",
                  "#FF6B9D", "#A18AFF", "#FF9A76", "#00D4AA", "#FFC75F"],
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["dark_bg"],
        "plot_bgcolor": COLORS["card_bg"],
        "font": {"color": COLORS["text"], "family": "Space Grotesk, Inter, sans-serif", "size": 13},
        "title": {"font": {"size": 20, "color": COLORS["text"]}},
        "xaxis": {
            "gridcolor": "#1E2B4D",
            "zerolinecolor": "#1E2B4D",
            "title": {"font": {"size": 14}},
        },
        "yaxis": {
            "gridcolor": "#1E2B4D",
            "zerolinecolor": "#1E2B4D",
            "title": {"font": {"size": 14}},
        },
        "colorway": COLORS["gradient"],
        "margin": {"t": 60, "b": 40, "l": 50, "r": 30},
        "hoverlabel": {
            "bgcolor": COLORS["card_bg"],
            "font_size": 13,
            "font_color": COLORS["text"],
        },
    }
}


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """
    Apply the Fluid Analyst dark theme to a Plotly figure.
    """
    fig.update_layout(
        paper_bgcolor=COLORS["dark_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text"], family="Space Grotesk, Inter, sans-serif", size=13),
        xaxis=dict(gridcolor="#1E2B4D", zerolinecolor="#1E2B4D"),
        yaxis=dict(gridcolor="#1E2B4D", zerolinecolor="#1E2B4D"),
        margin=dict(t=60, b=40, l=50, r=30),
        hoverlabel=dict(
            bgcolor=COLORS["card_bg"],
            font_size=13,
            font_color=COLORS["text"],
        ),
    )
    return fig


def render_plotly_chart(fig: go.Figure, container=None, use_container_width: bool = True):
    """
    Render a Plotly figure in Streamlit with premium styling.
    """
    fig = apply_plotly_theme(fig)
    target = container or st
    target.plotly_chart(fig, use_container_width=use_container_width)


def render_matplotlib_chart(fig: plt.Figure, container=None):
    """
    Render a Matplotlib figure in Streamlit with premium styling.
    """
    apply_matplotlib_theme(fig)
    target = container or st
    target.pyplot(fig)
    plt.close(fig)


def apply_matplotlib_theme(fig: plt.Figure):
    """
    Apply the Fluid Analyst dark theme to a Matplotlib figure.
    """
    fig.patch.set_facecolor(COLORS["dark_bg"])
    for ax in fig.axes:
        ax.set_facecolor(COLORS["card_bg"])
        ax.tick_params(colors=COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color("#1E2B4D")
        ax.grid(True, color="#1E2B4D", alpha=0.5, linestyle="--")


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a Matplotlib figure to a base64-encoded PNG string.
    Useful for embedding charts in markdown or HTML.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["dark_bg"], edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def create_metric_card(label: str, value: str, delta: Optional[str] = None,
                       delta_color: str = "normal"):
    """
    Create a styled metric card in Streamlit.
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def get_color_palette(n: int = 10) -> list:
    """
    Get a list of N colors from the premium palette.
    Cycles through the gradient if n > len(gradient).
    """
    palette = COLORS["gradient"]
    return [palette[i % len(palette)] for i in range(n)]
