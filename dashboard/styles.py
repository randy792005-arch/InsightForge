"""
InsightForge Enterprise UI/UX Theme & Component Styler
Provides modern, SaaS-grade styling, custom CSS injection, typography hierarchy,
KPI card components, agent status steppers, and alert banners using data-URI SVG images.
"""

import streamlit as st

URI_HEADER = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMiIgaGVpZ2h0PSIyMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM2MEE1RkEiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxsaW5lIHgxPSIxOCIgeTE9IjIwIiB4Mj0iMTgiIHkyPSIxMCIvPjxsaW5lIHgxPSIxMiIgeTE9IjIwIiB4Mj0iMTIiIHkyPSI0Ii8+PGxpbmUgeDE9IjYiIHkxPSIyMCIgeDI9IjYiIHkyPSIxNCIvPjwvc3ZnPg=="
URI_CHECK = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwNTk2NjkiIHN0cm9rZS13aWR0aD0iMy41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5bGluZSBwb2ludHM9IjIwIDYgOSAxNyA0IDEyIi8+PC9zdmc+"
URI_REVENUE = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMyNTYzRUIiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik02IDNoMTIiLz48cGF0aCBkPSJNNiA4aDEyIi8+PHBhdGggZD0ibTYgMTMgOC41IDgiLz48cGF0aCBkPSJNNiAxM2gzIi8+PHBhdGggZD0iTTkgMTNjNi42NjcgMCA2LjY2Ny0xMCAwLTEwIi8+PC9zdmc+"
URI_UNITS = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwRDk0ODgiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxsaW5lIHgxPSIxNi41IiB5MT0iOS40IiB4Mj0iNy41IiB5Mj0iNC4yMSIvPjxwYXRoIGQ9Ik0yMSAxNlY4YTIgMiAwIDAgMC0xLTEuNzNsLTctNGEyIDIgMCAwIDAtMiAwbC03IDRBMiAyIDAgMCAwIDMgOHY4YTIgMiAwIDAgMCAxIDEuNzNsNyA0YTIgMiAwIDAgMCAyIDBsNy00QTIgMiAwIDAgMCAyMSAxNnoiLz48cG9seWxpbmUgcG9pbnRzPSIzLjI3IDYuOTYgMTIgMTIuMDEgMjAuNzMgNi45NiIvPjwvc3ZnPg=="
URI_ORDERS = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM0RjQ2RTUiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjkiIGN5PSIyMSIgcj0iMSIvPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjEiIHI9IjEiLz48cGF0aCBkPSJNMSAxaDRsMi42OCAxMy4zOWEyIDIgMCAwIDAgMiAxLjYxaDkuNzJhMiAyIDAgMCAwIDItMS42MUwyMyA2SDYiLz48L3N2Zz4="
URI_AOV = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNEOTc3MDYiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5bGluZSBwb2ludHM9IjIzIDYgMTMuNSAxNS41IDguNSAxMC41IDEgMTgiLz48cG9seWxpbmUgcG9pbnRzPSIxNyA2IDIzIDYgMjMgMTIiLz48L3N2Zz4="
URI_PRODUCTS = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwNTk2NjkiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yMC41OSAxMy40MWwtNy4xNyA3LjE3YTIgMiAwIDAgMS0yLjgzIDBMMiAxMlYyaDEwbDguNTkgOC41OWEyIDIgMCAwIDEgMCAyLjgyeiIvPjxsaW5lIHgxPSI3IiB5MT0iNyIgeDI9IjcuMDEiIHkyPSI3Ii8+PC9zdmc+"
URI_ALERT = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMiIgaGVpZ2h0PSIyMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNEOTc3MDYiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xMC4yOSAzLjg2TDEuODIgMThhMiAyIDAgMCAwIDEuNzEgM2gxNi45NGEyIDIgMCAwIDAgMS43MS0zTDEzLjcxIDMuODZhMiAyIDAgMCAwLTMuNDIgMHoiLz48bGluZSB4MT0iMTIiIHkxPSI5IiB4Mj0iMTIiIHkyPSIxMyIvPjxsaW5lIHgxPSIxMiIgeTE9IjE3IiB4Mj0iMTIuMDEiIHkyPSIxNyIvPjwvc3ZnPg=="


def inject_custom_css():
    """Injects high-end enterprise SaaS styling and custom CSS variables."""
    st.html("""
<style>
/* 1. Global Typography & Canvas */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Layout Spacing */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px;
}

/* 2. Enterprise Header Banner */
.enterprise-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 0.5rem;
    margin-bottom: 22px;
    color: #FFFFFF;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.header-title-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-icon-box {
    width: 38px;
    height: 38px;
    background: rgba(37, 99, 235, 0.25);
    border: 1px solid rgba(96, 165, 250, 0.4);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.header-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #F8FAFC;
}
.header-subtitle {
    font-size: 13px;
    color: #94A3B8;
    font-weight: 400;
    margin-top: 4px;
}
.header-badge {
    background: rgba(37, 99, 235, 0.25);
    border: 1px solid rgba(96, 165, 250, 0.4);
    color: #93C5FD;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-block;
}
.status-pill-live {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34D399;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #10B981;
}

/* 3. Redesigned 5-Agent Stepper Cards */
.agent-stepper-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 22px;
}
.agent-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}
.agent-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
}
.agent-card.agent-success {
    border-top: 3px solid #10B981;
}
.agent-card.agent-warning {
    border-top: 3px solid #F59E0B;
}
.agent-card.agent-error {
    border-top: 3px solid #EF4444;
}
.agent-card.agent-pending {
    border-top: 3px solid #94A3B8;
    opacity: 0.8;
}
.agent-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.agent-num-pill {
    font-size: 10px;
    font-weight: 700;
    color: #64748B;
    background: #F1F5F9;
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.04em;
}
.agent-badge-success {
    color: #059669;
    background: #ECFDF5;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.agent-badge-pending {
    color: #64748B;
    background: #F8FAFC;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
}
.agent-badge-error {
    color: #DC2626;
    background: #FEF2F2;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
}
.agent-name {
    font-size: 13px;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.agent-desc {
    font-size: 11px;
    color: #64748B;
    line-height: 1.3;
    margin: 0;
}

/* 4. Enterprise KPI Cards */
.kpi-container-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease-in-out;
    position: relative;
}
.kpi-card:hover {
    box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
}
.kpi-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.kpi-icon-box {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.icon-blue { background: #EFF6FF; }
.icon-teal { background: #F0FDFA; }
.icon-indigo { background: #EEF2FF; }
.icon-amber { background: #FFFBEB; }
.icon-emerald { background: #ECFDF5; }

.kpi-value {
    font-size: 24px;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 6px;
}
.kpi-subtext {
    font-size: 12px;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 4px;
}
.kpi-badge-pill {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    background: #F1F5F9;
    color: #475569;
}

/* 5. Custom Enterprise Alert Banner */
.enterprise-alert-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 5px solid #F59E0B;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
.alert-icon-box {
    flex-shrink: 0;
    margin-top: 2px;
}
.alert-body-title {
    font-size: 14px;
    font-weight: 700;
    color: #92400E;
    margin: 0 0 2px 0;
}
.alert-body-desc {
    font-size: 13px;
    color: #78350F;
    margin: 0;
    line-height: 1.4;
}

/* 6. Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    transition: background 0.15s ease !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}

.sidebar-brand-box {
    padding: 14px 4px 18px 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sidebar-brand-title {
    font-size: 18px;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
    margin: 0;
}
.sidebar-brand-sub {
    font-size: 11px;
    color: #94A3B8 !important;
    margin: 2px 0 0 0;
}
.sidebar-status-item {
    font-size: 12px;
    padding: 4px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dot-success {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    margin-right: 6px;
}
.dot-pending {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #94A3B8;
    margin-right: 6px;
}
.dot-error {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #EF4444;
    margin-right: 6px;
}

/* 7. Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}
</style>
""")

def render_dashboard_header(
    title: str = "InsightForge Executive BI Dashboard",
    subtitle: str = "Autonomous Multi-Agent Retail Business Intelligence and Decision Support System",
    badge_text: str = "InsightForge Enterprise"
):
    """Renders the sleek, high-contrast dark navy header banner with data-URI SVG icon."""
    html = f"""<div class="enterprise-header">
<div>
<div style="margin-bottom: 8px;"><span class="header-badge">{badge_text}</span></div>
<div class="header-title-wrapper">
<div class="header-icon-box">
<img src="{URI_HEADER}" width="22" height="22" alt="Dashboard Icon" />
</div>
<h1 class="header-title">{title}</h1>
</div>
<div class="header-subtitle">{subtitle}</div>
</div>
<div>
<div class="status-pill-live"><span class="status-dot"></span><span>AI Pipeline Active</span></div>
</div>
</div>"""
    st.html(html)


def render_agent_stepper_ui(agent_status: dict):
    """Renders the 5 specialized AI agents as modern status cards with SVG checkmarks."""
    agents_meta = [
        ("01", "Ingestion Agent", "ingestion", "Schema validation & raw persistence"),
        ("02", "Cleaning Agent", "cleaning", "Deduplication, types & quality score"),
        ("03", "Analytics Agent", "analytics", "KPIs, trends & category Pareto"),
        ("04", "Forecasting Agent", "forecasting", "Prophet / XGBoost demand model"),
        ("05", "Recommendation", "recommendation", "Replenishment & liquidation")
    ]

    cards_html = []
    for num, name, key, desc in agents_meta:
        info = agent_status.get(key, {"status": "PENDING", "msg": ""})
        status_val = info.get("status", "PENDING")

        if status_val in ["SUCCESS", "WARNING"]:
            card_class = "agent-success" if status_val == "SUCCESS" else "agent-warning"
            badge_html = f'<span class="agent-badge-success"><img src="{URI_CHECK}" width="10" height="10" alt="Success" /> {status_val}</span>'
        elif status_val == "ERROR":
            card_class = "agent-error"
            badge_html = '<span class="agent-badge-error">ERROR</span>'
        else:
            card_class = "agent-pending"
            badge_html = '<span class="agent-badge-pending">READY</span>'

        card = f"""<div class="agent-card {card_class}">
<div class="agent-header-row"><span class="agent-num-pill">{num}</span>{badge_html}</div>
<div class="agent-name">{name}</div>
<div class="agent-desc">{desc}</div>
</div>"""
        cards_html.append(card)

    container_html = f'<div class="agent-stepper-grid">{"".join(cards_html)}</div>'
    st.html(container_html)


def render_kpi_cards_grid(kpis: dict):
    """Renders 5 top-tier KPI metric cards with clean SVG icons and trend badges."""
    total_rev = f"₹{kpis.get('total_revenue', 0.0):,.2f}"
    units_sold = f"{kpis.get('total_units_sold', 0.0):,.0f}"
    transactions = f"{kpis.get('total_transactions', 0):,}"
    aov = f"₹{kpis.get('avg_order_value', 0.0):,.2f}"
    unique_skus = f"{kpis.get('unique_products', 0)}"

    total_days = kpis.get('total_days', 0)
    avg_daily_units = kpis.get('avg_daily_units', 0.0)
    categories_count = kpis.get('unique_categories', 0)
    avg_daily_rev = kpis.get('avg_daily_revenue', 0.0)

    html = f"""<div class="kpi-container-grid">
<div class="kpi-card">
<div class="kpi-top-row">
<span class="kpi-label">Total Revenue</span>
<div class="kpi-icon-box icon-blue">
<img src="{URI_REVENUE}" width="18" height="18" alt="Revenue" />
</div>
</div>
<div class="kpi-value">{total_rev}</div>
<div class="kpi-subtext"><span class="kpi-badge-pill">{total_days} Days Scope</span></div>
</div>

<div class="kpi-card">
<div class="kpi-top-row">
<span class="kpi-label">Units Sold</span>
<div class="kpi-icon-box icon-teal">
<img src="{URI_UNITS}" width="18" height="18" alt="Units" />
</div>
</div>
<div class="kpi-value">{units_sold}</div>
<div class="kpi-subtext"><span class="kpi-badge-pill">~{avg_daily_units:.0f} units/day</span></div>
</div>

<div class="kpi-card">
<div class="kpi-top-row">
<span class="kpi-label">Total Orders</span>
<div class="kpi-icon-box icon-indigo">
<img src="{URI_ORDERS}" width="18" height="18" alt="Orders" />
</div>
</div>
<div class="kpi-value">{transactions}</div>
<div class="kpi-subtext"><span class="kpi-badge-pill">₹{avg_daily_rev:,.0f} avg/day</span></div>
</div>

<div class="kpi-card">
<div class="kpi-top-row">
<span class="kpi-label">Average Order (AOV)</span>
<div class="kpi-icon-box icon-amber">
<img src="{URI_AOV}" width="18" height="18" alt="AOV" />
</div>
</div>
<div class="kpi-value">{aov}</div>
<div class="kpi-subtext"><span class="kpi-badge-pill">Basket Size</span></div>
</div>

<div class="kpi-card">
<div class="kpi-top-row">
<span class="kpi-label">Active Products</span>
<div class="kpi-icon-box icon-emerald">
<img src="{URI_PRODUCTS}" width="18" height="18" alt="Products" />
</div>
</div>
<div class="kpi-value">{unique_skus}</div>
<div class="kpi-subtext"><span class="kpi-badge-pill">{categories_count} Categories</span></div>
</div>
</div>"""
    st.html(html)


def render_alert_banner(critical_count: int, critical_items_preview: str = ""):
    """Renders a sleek enterprise warning alert for stockout risks with SVG warning icon."""
    html = f"""<div class="enterprise-alert-card">
<div class="alert-icon-box">
<img src="{URI_ALERT}" width="22" height="22" alt="Warning" />
</div>
<div>
<div class="alert-body-title">Urgent Replenishment Attention Required</div>
<div class="alert-body-desc">
The Recommendation Agent detected <b>{critical_count} critical product(s)</b> approaching immediate stockout (less than 7 days of inventory remaining).
{critical_items_preview}
Expedited purchase orders recommended to safeguard revenue.
</div>
</div>
</div>"""
    st.html(html)
