"""
Lights On / Lights Off  &  Consumption Intelligence Dashboard

Tab 1 — Lights On / Lights Off: Detects the largest increases and decreases
in Book of Business consumption, surfaces root causes, engagement gaps, and
open use cases for Overlay teams.

Tab 2 — Consumption Intelligence: Monitors consumption health, trends,
predictions, and anomalies across the Book of Business for proactive
intervention.

Data Sources (Snowflake):
- SALES.REPORTING.BOB_CONSUMPTION — trailing revenue windows, deltas, growth rates, predictions
- SALES.SE_REPORTING.ACCOUNT_BUSINESS_INDICATORS — assessment scores, context, growth tiers
- SALES.SE_REPORTING.ACCOUNT_BASE_METRICS — use case counts, meeting dates, run rates
- SALES.SALES_ENGINEERING.CONSUMPTION_RISK_MOVEMENTS — risk flags, mitigation notes
- SALES.REPORTING.ACTIVE_USE_CASE_PIPELINE — open use cases with health & probability
- SALES.REPORTING.CONSUMPTION_DAILY — daily account-level consumption with targets & predictions
- FINANCE.CUSTOMER.PRODUCT_CATEGORY_REV_ACTUALS_W_FORECAST_SFDC — product/feature-level revenue actuals & forecasts
"""

from datetime import timedelta

import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Overlay Dashboard",
    page_icon=":material/trending_up:",
    layout="wide",
)

CHART_HEIGHT = 340
DEFAULT_THRESHOLD = 10

# ── Reporting suite thresholds ───────────────────────────────────────────────
COVERAGE_WARNING  = 1.5    # Pipeline coverage below this → yellow
COVERAGE_HEALTHY  = 2.0    # Pipeline coverage at or above this → green
DAYS_NO_CONTACT   = 30     # Engagement gap threshold (days since last meeting)
DAYS_STALE_COMMENT = 30    # Days since comment → stale flag on use case
EACV_PCT_HIGH     = 0.75   # EACV percentile for "high value" deal classification

# ── Snowflake connection ─────────────────────────────────────────────────────

def get_conn():
    try:
        # In Streamlit in Snowflake (SiS), get_active_session() gives the owner's
        # Snowpark session. We activate secondary roles so warehouse grants from
        # the user's full role hierarchy are available, then set the warehouse.
        try:
            from snowflake.snowpark.context import get_active_session
            _session = get_active_session()
            try:
                _session.sql("USE SECONDARY ROLES ALL").collect()
            except Exception:
                pass
            for _wh in ["ACR_WH", "SNOWHOUSE", "SNOWADHOC"]:
                try:
                    _session.sql(f"USE WAREHOUSE {_wh}").collect()
                    break
                except Exception:
                    continue

            class _SISConn:
                def query(self, sql, **kwargs):
                    return _session.sql(sql).to_pandas()

            return _SISConn()
        except Exception:
            pass
        # Local development: fall back to standard Streamlit connection.
        return st.connection("snowflake", warehouse="SNOWADHOC")
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.info(
            "Configure your Snowflake connection in `.streamlit/secrets.toml` "
            "or via environment variables."
        )
        st.stop()


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner="Loading Book of Business data...")
def load_bob() -> pd.DataFrame:
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_bob")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading account indicators...")
def load_indicators() -> pd.DataFrame:
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_account_indicators")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading account metrics...")
def load_base_metrics() -> pd.DataFrame:
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_account_base_metrics")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading consumption risk data...")
def load_risk_movements() -> pd.DataFrame:
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_risk_movements")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading use case pipeline...")
def load_use_cases() -> pd.DataFrame:
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_active_use_cases")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading product categories...")
def load_product_accounts() -> pd.DataFrame:
    """Distinct account → product_category mapping (recent 3 months)."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_product_accounts")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading daily consumption trends...")
def load_daily_consumption_agg() -> pd.DataFrame:
    """Server-side aggregated daily consumption for current fiscal quarter."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_daily_consumption")
    df.columns = df.columns.str.lower()
    df["general_date"] = pd.to_datetime(df["general_date"])
    return df


@st.cache_data(ttl=1800, show_spinner="Loading previous quarter revenue...")
def load_prev_quarter_revenue() -> pd.DataFrame:
    """Total revenue for the 2nd previous fiscal quarter (pre-computed by dbt)."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_prev_quarter_revenue")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading monthly consumption trends...")
def load_monthly_consumption_trend() -> pd.DataFrame:
    """Monthly consumption by product category — last 6 months, all accounts."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_monthly_trend")
    df.columns = df.columns.str.lower()
    df["consumption_month"] = pd.to_datetime(df["consumption_month"])
    return df


@st.cache_data(ttl=1800, show_spinner="Loading functional areas...")
def load_functional_areas() -> pd.DataFrame:
    """Distinct account → functional area mapping (latest WLC week)."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_functional_areas")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading product-level consumption...")
def load_product_consumption() -> pd.DataFrame:
    """Product-category-level monthly revenue with MoM daily-rate comparisons."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_product_consumption")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading functional area credit shares...")
def load_func_area_shares() -> pd.DataFrame:
    """Per-account functional area credit share (6-month average)."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_func_area_shares")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading monthly functional area shares...")
def load_monthly_func_area_shares() -> pd.DataFrame:
    """Per-account, per-month functional area credit share from weekly WLC snapshots."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_monthly_func_area_shares")
    df.columns = df.columns.str.lower()
    df["share_month"] = pd.to_datetime(df["share_month"])
    return df


@st.cache_data(ttl=1800, show_spinner="Loading feature credit shares...")
def load_feature_shares() -> pd.DataFrame:
    """Per-account feature revenue share (last 3 months of actuals)."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_feature_shares")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading use case forecast pipeline...")
def load_forecast_pipeline() -> pd.DataFrame:
    """Full use case pipeline (stages 1–7) with health, overlay metadata, and text sentiment."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_forecast_pipeline")
    df.columns = df.columns.str.lower()
    for c in ["decision_date", "go_live_date", "technical_win_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner="Loading specialist overlay...")
def load_specialist_overlay() -> pd.DataFrame:
    """Per-specialist use case assignments with role, involvement, and comments."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_specialist_pipeline")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=1800, show_spinner="Loading specialist management hierarchy...")
def load_specialist_mgmt_hierarchy() -> pd.DataFrame:
    """Map each specialist to their Workday first-line manager."""
    conn = get_conn()
    df = conn.query("SELECT * FROM AFE.DBT_DEV_overlay.mart_overlay_specialist_mgmt_hierarchy")
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=600, show_spinner="Loading specialist engagement data...")
def load_specialist_engagement_data() -> pd.DataFrame:
    """Specialist engagement snapshot with activity metrics and staleness indicators."""
    conn = get_conn()
    df = conn.query("""
        SELECT
            PREFERRED_NAME,
            MANAGER_NAME,
            IS_PEOPLE_MANAGER,
            ORIGINAL_HIRE_DATE,
            TENURE,
            HIERARCHY_3,
            HIERARCHY_4,
            HIERARCHY_5,
            SFDC_ID,
            SPECIALIST_GROUP,
            SPECIALIST_COMMENTS_14D AS COMMENTS_14D,
            SPECIALIST_COMMENTS_7D  AS COMMENTS_7D,
            ACTIVITIES_14D,
            W_UC_ACTIVITY_14D,
            NO_UC_ACTIVITY_14D,
            ACTIVITIES_7D,
            W_UC_ACTIVITY_7D,
            NO_UC_ACTIVITY_7D,
            STATUS                  AS UPDATE_NEEDED_STATUS
        FROM AFE.DBT_DEV_MARTS.SPECIALIST_ENGAGEMENT_STATUS
    """)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=600, show_spinner="Loading use case detail data...")
def load_use_case_detail_data() -> pd.DataFrame:
    """Use case details for specialist drill-down (joins via ACTIVE_USE_CASE_LIST)."""
    conn = get_conn()
    df = conn.query("""
        SELECT *
        FROM AFE.DBT_DEV_OVERLAY.MART_OVERLAY_ACTIVE_USE_CASES
    """)
    df.columns = df.columns.str.lower()
    # Ensure use_case_id is string for join
    df["use_case_id"] = df["use_case_id"].astype(str)
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

PERIOD_MAP = {
    "30 Days": {
        "current": "revenue_trailing_30d",
        "prior": "revenue_trailing_30_60d",
        "delta": "revenue_delta_30",
        "growth": "growth_rate_30d",
    },
    "60 Days": {
        "current": "revenue_trailing_60d",
        "prior": "revenue_trailing_60_120d",
        "delta": "revenue_delta_60",
        "growth": None,
    },
    "90 Days": {
        "current": "revenue_trailing_90d",
        "prior": "revenue_trailing_90_180d",
        "delta": "revenue_delta_90",
        "growth": "growth_rate_90d",
    },
    "180 Days": {
        "current": "revenue_trailing_180d",
        "prior": "revenue_trailing_180_360d",
        "delta": "revenue_delta_180",
        "growth": "growth_rate_180d",
    },
}


def fmt_currency(val):
    if val is None or pd.isna(val):
        return "$0.00"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:,.2f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:,.2f}K"
    return f"${val:,.2f}"


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "0.00%"
    return f"{val * 100:+.2f}%"


def compute_change_pct(row, cols):
    """Compute % change: (current - prior) / prior."""
    prior = row[cols["prior"]]
    if prior is None or pd.isna(prior) or prior == 0:
        return None
    return (row[cols["current"]] - prior) / abs(prior)


# ── Load all data ────────────────────────────────────────────────────────────

bob = load_bob()
indicators = load_indicators()
base_metrics = load_base_metrics()
risk_movements = load_risk_movements()
use_cases = load_use_cases()
product_accounts = load_product_accounts()
daily_consumption = load_daily_consumption_agg()
prev2_fq_revenue = load_prev_quarter_revenue()
monthly_trend = load_monthly_consumption_trend()
functional_areas = load_functional_areas()
product_consumption = load_product_consumption()
func_area_shares = load_func_area_shares()
monthly_func_area_shares = load_monthly_func_area_shares()
feature_shares = load_feature_shares()
forecast_pipeline = load_forecast_pipeline()
specialist_overlay = load_specialist_overlay()
specialist_mgmt_hierarchy = load_specialist_mgmt_hierarchy()
specialist_engagement = load_specialist_engagement_data()
use_case_detail = load_use_case_detail_data()


# ── Resolve pending navigation (must run before any widget renders) ──────────
# Buttons set _nav_module/_nav_page instead of the widget-owned keys directly,
# because Streamlit forbids modifying a widget key after it has been instantiated.
if "_nav_module" in st.session_state:
    st.session_state["module_selector"] = st.session_state.pop("_nav_module")
if "_nav_page" in st.session_state:
    _resolved_mod = st.session_state.get("module_selector", "Home")
    st.session_state[f"page_selector_{_resolved_mod}"] = st.session_state.pop("_nav_page")

# ── Sidebar filters ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Navigation")
    module = st.selectbox("Module", ["Home", "Consumption", "Use Case", "Flags", "Guide", "About"], key="module_selector")

    MODULE_PAGES = {
        "Home": ["Dashboard Overview"],
        "Consumption": ["Lights On / Lights Off", "Consumption Intelligence"],
        "Use Case": ["Use Case Forecasting", "Use Case Hygiene"],
        "Flags": ["Field Flags"],
        "Guide": ["Operational Manual"],
        "About": ["About This App"],
    }
    page = st.radio(
        "Page",
        MODULE_PAGES[module],
        key=f"page_selector_{module}",
    )
    if module not in ("Home", "Guide", "About"):
        st.divider()
        st.header("Filters")

    if module == "Consumption":
        all_regions = sorted(bob["region"].dropna().unique().tolist())
        selected_regions = st.multiselect("Region", all_regions, default=[])

        all_theaters = sorted(bob["theater"].dropna().unique().tolist())
        selected_theaters = st.multiselect("Theater", all_theaters, default=[])

        all_products = sorted(product_accounts["product_category"].dropna().unique().tolist())
        selected_products = st.multiselect("Product Category", all_products, default=[])

        # Feature filter — dynamically populated based on selected Product Category
        if selected_products:
            all_features = sorted(
                feature_shares.loc[
                    feature_shares["product_category"].isin(selected_products), "feature"
                ].dropna().unique().tolist()
            )
        else:
            all_features = sorted(feature_shares["feature"].dropna().unique().tolist())
        selected_features = st.multiselect("Feature", all_features, default=[])

        all_func_areas = sorted(functional_areas["functional_area"].dropna().unique().tolist())
        selected_func_areas = st.multiselect("Functional Area", all_func_areas, default=[])

        all_industries = sorted(bob["industry"].dropna().unique().tolist())
        selected_industries = st.multiselect("Industry", all_industries, default=[])

        owner_search = st.text_input("Account / Owner Search")

    elif module == "Use Case":
        if page == "Use Case Forecasting":
            # Determine current FQ from the pipeline data
            _cq_fqs = forecast_pipeline.loc[
                forecast_pipeline["is_current_fq_go_live"] == True, "go_live_fq"
            ].dropna().unique()
            current_fq = _cq_fqs[0] if len(_cq_fqs) > 0 else "FY2027-Q1"

            all_fq_go_live = sorted(
                forecast_pipeline["go_live_fq"].dropna().unique().tolist(), reverse=True
            )
            fc_time_period = st.selectbox(
                "Fiscal Quarter",
                all_fq_go_live,
                index=all_fq_go_live.index(current_fq) if current_fq in all_fq_go_live else 0,
                key="fc_time_period",
            )

            fc_all_theaters = sorted(
                forecast_pipeline["theater_name"].dropna().unique().tolist()
            )
            fc_selected_theaters = st.multiselect(
                "Theater", fc_all_theaters, default=[], key="fc_theaters"
            )

            fc_all_regions = sorted(
                forecast_pipeline["region_name"].dropna().unique().tolist()
            )
            fc_selected_regions = st.multiselect(
                "Region", fc_all_regions, default=[], key="fc_regions"
            )

            fc_all_reps = sorted(
                forecast_pipeline["rep_name"].dropna().unique().tolist()
            )
            fc_selected_reps = st.multiselect(
                "Account Owner", fc_all_reps, default=[], key="fc_reps"
            )

            fc_all_specialists = sorted(
                specialist_overlay.loc[
                    specialist_overlay["is_team_member"] == 1, "specialist_name"
                ].dropna().unique().tolist()
            )
            fc_selected_specialists = st.multiselect(
                "Specialist Name", fc_all_specialists, default=[], key="fc_specialists"
            )

            fc_all_prod_cats = sorted(
                {v.strip() for val in forecast_pipeline["product_category"].dropna()
                 for v in str(val).split(";") if v.strip()}
            )
            fc_selected_prod_cats = st.multiselect(
                "Product Categories", fc_all_prod_cats, default=[], key="fc_prod_cats"
            )

            # Specialist Manager dropdown: top-level managers + first-line managers (who have reports)
            _top_mgrs = set(specialist_overlay["specialist_manager_name"].dropna().unique())
            _first_line_mgrs = set(specialist_mgmt_hierarchy["workday_manager"].dropna().unique())
            # First-line managers are people who manage specialists but aren't top-level
            # Include both levels in the dropdown
            fc_all_spec_mgrs = sorted(_top_mgrs | _first_line_mgrs)
            fc_selected_spec_mgrs = st.multiselect(
                "Specialist Manager", fc_all_spec_mgrs, default=[], key="fc_spec_mgrs"
            )

            fc_all_tech_uc = sorted(
                {v.strip() for val in forecast_pipeline["technical_use_case"].dropna()
                 for v in str(val).split(";") if v.strip()}
            )
            fc_selected_tech_uc = st.multiselect(
                "Technical Use Case", fc_all_tech_uc, default=[], key="fc_tech_uc"
            )

            fc_all_features = sorted(
                {v.strip() for val in forecast_pipeline["prioritized_features"].dropna()
                 for v in str(val).split(";") if v.strip()}
            )
            fc_selected_features = st.multiselect(
                "Prioritized Feature", fc_all_features, default=[], key="fc_features"
            )

            fc_high_acv_no_tw = st.checkbox(
                "High ACV / No Technical Win",
                help="Show only high-EACV use cases without a technical win",
                key="fc_high_acv_no_tw",
            )

            fc_missing_specialist = st.checkbox(
                "Missing Specialist Comments",
                help="Show use cases where specialist has not provided comments",
                key="fc_missing_specialist",
            )

        elif page == "Use Case Hygiene":
            hy_active_only = st.checkbox(
                "Active Specialists Only", value=True, key="hy_active_only"
            )

            hy_all_groups = sorted(
                specialist_engagement["specialist_group"].dropna().unique().tolist()
            )
            hy_selected_groups = st.multiselect(
                "Specialist Group", hy_all_groups, default=hy_all_groups, key="hy_groups"
            )

            hy_all_theaters = sorted(
                specialist_engagement["hierarchy_3"].dropna().unique().tolist()
            )
            hy_selected_theaters = st.multiselect(
                "Theater", hy_all_theaters, default=hy_all_theaters, key="hy_theaters"
            )

            hy_all_statuses = sorted(
                specialist_engagement["update_needed_status"].dropna().unique().tolist()
            )
            hy_selected_statuses = st.multiselect(
                "Update Status", hy_all_statuses, default=hy_all_statuses, key="hy_statuses"
            )

            hy_all_managers = sorted(
                specialist_engagement["manager_name"].dropna().unique().tolist()
            )
            hy_selected_managers = st.multiselect(
                "Manager", hy_all_managers, default=[], key="hy_managers"
            )

    elif module == "Flags":
        fl_all_theaters = sorted(specialist_engagement["hierarchy_3"].dropna().unique().tolist())
        fl_selected_theaters = st.multiselect("Theater", fl_all_theaters, default=fl_all_theaters, key="fl_theaters")
        fl_all_managers = sorted(specialist_engagement["manager_name"].dropna().unique().tolist())
        fl_selected_managers = st.multiselect("Manager", fl_all_managers, default=[], key="fl_managers")
        fl_all_groups = sorted(specialist_engagement["specialist_group"].dropna().unique().tolist())
        fl_selected_groups = st.multiselect("Specialist Group", fl_all_groups, default=fl_all_groups, key="fl_groups")

    st.divider()
    st.caption(f"Data refresh: cached 30 min TTL")


# ── Home module ───────────────────────────────────────────────────────────────

if module == "Home":
    st.markdown("# :material/dashboard: Business Health Command Center")
    st.caption("Full book of business — unfiltered · Cached 30 min")

    # ── Compute consumption KPIs ──────────────────────────────────────────────
    _home_bob = bob.merge(
        base_metrics[["account_id", "last_account_meeting_date"]],
        on="account_id", how="left",
    )
    _today_h = pd.Timestamp.now().normalize()
    _home_bob["_days_mtg"] = (
        _today_h - pd.to_datetime(_home_bob["last_account_meeting_date"])
    ).dt.days.fillna(999).astype(int)

    _qtd          = _home_bob["current_fiscal_quarter_revenue"].fillna(0).sum()
    _prev_fq      = _home_bob["previous_fiscal_quarter_revenue"].fillna(0).sum()
    _predicted_fq = _home_bob["revenue_prediction_current_fiscal_quarter"].fillna(0).sum()
    _qoq_pct      = (_qtd - _prev_fq) / _prev_fq if _prev_fq > 0 else 0.0
    _lights_off   = int((_home_bob["revenue_delta_30"].fillna(0) < 0).sum())
    _rev_at_risk  = _home_bob.loc[
        _home_bob["revenue_delta_30"].fillna(0) < 0, "revenue_delta_30"
    ].fillna(0).sum()
    _urgent_dec   = int((
        (_home_bob["revenue_delta_30"].fillna(0) < 0) &
        (_home_bob["_days_mtg"] >= DAYS_NO_CONTACT)
    ).sum())

    # ── Compute pipeline KPIs ─────────────────────────────────────────────────
    _fp_h = forecast_pipeline.copy()
    _fp_cur = _fp_h[_fp_h["is_current_fq_go_live"] == True]

    # Go-Live pipeline coverage (current FQ go-live rows)
    _gl_open_eacv = _fp_cur[_fp_cur["stage_number"].between(1, 6)]["use_case_eacv"].fillna(0).sum()
    _gl_won_eacv  = _fp_cur[_fp_cur["stage_number"] == 7]["use_case_eacv"].fillna(0).sum()
    _gl_cov = _gl_open_eacv / _gl_won_eacv if _gl_won_eacv > 0 else float("inf")

    # Win pipeline coverage (decision date in current FQ — derive date range from FQ string)
    _cq_fqs = _fp_h.loc[_fp_h["is_current_fq_go_live"] == True, "go_live_fq"].dropna()
    _cq_str = _cq_fqs.iloc[0] if len(_cq_fqs) > 0 else None
    if _cq_str:
        try:
            _p = _cq_str.replace("FY", "").split("-Q")
            _fy, _q = int(_p[0]), int(_p[1])
            _cy = _fy - 1
            _ms = (_q - 1) * 3 + 2
            if _ms > 12:
                _ms -= 12
                _cy += 1
            _fqs = pd.Timestamp(year=_cy, month=_ms, day=1)
            _fqe = _fqs + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        except Exception:
            _fqs, _fqe = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-30")
        _wins_h = _fp_h[(_fp_h["decision_date"] >= _fqs) & (_fp_h["decision_date"] <= _fqe)]
        _win_open = _wins_h[_wins_h["stage_number"].between(1, 3)]["use_case_eacv"].fillna(0).sum()
        _win_won  = _wins_h[_wins_h["stage_number"] >= 4]["use_case_eacv"].fillna(0).sum()
        _win_cov  = _win_open / _win_won if _win_won > 0 else float("inf")
        _qtd_won  = _win_won
    else:
        _win_cov, _qtd_won = 0.0, 0.0

    _at_risk_gl = int((_fp_h["health_status"] == "At Risk").sum())
    _se_overdue = int((specialist_engagement["update_needed_status"] == "Needed Now").sum())

    # Missing comments EACV
    _sp_miss_ids = specialist_overlay.loc[
        (specialist_overlay["is_team_member"] == 1) &
        (specialist_overlay["specialist_comments"].isna() |
         (specialist_overlay["specialist_comments"].str.strip() == "")),
        "use_case_id",
    ].unique()
    _fp_active_h = _fp_h[_fp_h["stage_number"].between(1, 6)]
    _missing_cmt_eacv = _fp_active_h[
        _fp_active_h["use_case_id"].isin(_sp_miss_ids)
    ]["use_case_eacv"].fillna(0).sum()

    # ── Row 1: Consumption Health ─────────────────────────────────────────────
    st.divider()
    st.subheader(":material/trending_up: Consumption Health")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("QTD Consumption", fmt_currency(_qtd))
        if st.button("→ Consumption Intelligence", key="h_qtd", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Consumption Intelligence"
            st.rerun()
    with h2:
        st.metric("Predicted FQ Finish", fmt_currency(_predicted_fq))
        if st.button("→ Consumption Intelligence", key="h_pred", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Consumption Intelligence"
            st.rerun()
    with h3:
        st.metric("QoQ vs Prior FQ", fmt_pct(_qoq_pct))
        if st.button("→ Consumption Intelligence", key="h_qoq", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Consumption Intelligence"
            st.rerun()

    h4, h5, h6 = st.columns(3)
    with h4:
        st.metric("Lights Off Accounts", _lights_off)
        if st.button("→ Lights On / Lights Off", key="h_lo", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Lights On / Lights Off"
            st.rerun()
    with h5:
        st.metric("Urgent Decliners", _urgent_dec)
        if st.button("→ Lights On / Lights Off", key="h_ud", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Lights On / Lights Off"
            st.rerun()
    with h6:
        st.metric("Revenue at Risk (30d)", fmt_currency(_rev_at_risk))
        if st.button("→ Lights On / Lights Off", key="h_rar", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Lights On / Lights Off"
            st.rerun()

    # ── Row 2: Pipeline Health ────────────────────────────────────────────────
    st.divider()
    st.subheader(":material/query_stats: Pipeline Health")

    p1, p2, p3 = st.columns(3)
    with p1:
        _gl_cov_disp = f"{_gl_cov:.1f}x" if _gl_cov != float("inf") else "∞"
        _gl_color = "normal" if _gl_cov >= COVERAGE_HEALTHY else ("off" if _gl_cov < COVERAGE_WARNING else "normal")
        st.metric("Go-Live Coverage", _gl_cov_disp)
        if st.button("→ Use Case Forecasting", key="h_glc", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()
    with p2:
        _win_cov_disp = f"{_win_cov:.1f}x" if _win_cov != float("inf") else "∞"
        st.metric("Win Coverage", _win_cov_disp)
        if st.button("→ Use Case Forecasting", key="h_wc", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()
    with p3:
        st.metric("QTD Won", fmt_currency(_qtd_won))
        if st.button("→ Use Case Forecasting", key="h_won", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()

    p4, p5, p6 = st.columns(3)
    with p4:
        st.metric("At-Risk Go-Lives", _at_risk_gl)
        if st.button("→ Use Case Forecasting", key="h_argl", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()
    with p5:
        st.metric("Specialists Overdue", _se_overdue)
        if st.button("→ Use Case Hygiene", key="h_seo", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Hygiene"
            st.rerun()
    with p6:
        st.metric("Missing Comments ($)", fmt_currency(_missing_cmt_eacv))
        if st.button("→ Use Case Forecasting", key="h_mc", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()

    st.divider()
    if st.button(":material/flag: View All Field Flags", type="primary", key="h_flags"):
        st.session_state["_nav_module"] = "Flags"
        st.rerun()


# ── Apply filters (Consumption module) ──────────────────────────────────────

if module == "Consumption":
    df = bob.copy()

    # Default comparison period for shared pipeline (used by scope overlay).
    # The Lights On / Lights Off page will recompute with the user-selected period.
    _default_cols = PERIOD_MAP["30 Days"]
    df["change_pct"] = df.apply(lambda r: compute_change_pct(r, _default_cols), axis=1)
    df["abs_change_pct"] = df["change_pct"].abs()
    df["current_revenue"] = df[_default_cols["current"]]
    df["prior_revenue"] = df[_default_cols["prior"]]
    df["delta_revenue"] = df[_default_cols["delta"]]

    # Join indicators
    df = df.merge(
        indicators[["account_id", "assessment_tier", "consumption_context",
                    "growth_tier", "use_case_health_context",
                    "se_engagement_context", "consumption_risk", "account_risk",
                    "district_name"]],
        on="account_id", how="left", suffixes=("", "_ind")
    )

    # Join base metrics
    df = df.merge(
        base_metrics[["account_id", "last_account_meeting_date", "active_use_cases",
                      "in_pursuit_use_cases", "in_implementation_use_cases",
                      "high_risk_use_cases", "annual_run_rate_l30d",
                      "run_rate_acceleration_pct"]],
        on="account_id", how="left"
    )

    # Region filter
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]

    # Theater filter
    if selected_theaters:
        df = df[df["theater"].isin(selected_theaters)]

    # Industry filter
    if selected_industries:
        df = df[df["industry"].isin(selected_industries)]

    # Product filter — keep accounts that have consumption in selected product(s)
    if selected_products:
        accts_with_product = product_accounts.loc[
            product_accounts["product_category"].isin(selected_products), "account_id"
        ].unique()
        df = df[df["account_id"].isin(accts_with_product)]

    # Functional Area filter — keep accounts with workloads in selected area(s)
    if selected_func_areas:
        accts_with_func = functional_areas.loc[
            functional_areas["functional_area"].isin(selected_func_areas), "account_id"
        ].unique()
        df = df[df["account_id"].isin(accts_with_func)]

    # Feature filter — keep accounts with usage in selected feature(s)
    if selected_features:
        accts_with_feature = feature_shares.loc[
            feature_shares["feature"].isin(selected_features), "account_id"
        ].unique()
        df = df[df["account_id"].isin(accts_with_feature)]

    # ── Consumption scope overlay ────────────────────────────────────────────────
    # When Feature, Product Category, or Functional Area filters are active,
    # replace the account-total consumption with scoped consumption so KPIs,
    # leaderboard, and detail tables reflect only the selected slice.
    # Priority: Feature > Product Category > Functional Area.

    scope_label = ""  # empty = total account consumption

    if selected_features:
        # Use credit-share allocation: feature % of credits × BOB trailing revenue
        fs = feature_shares[
            feature_shares["feature"].isin(selected_features)
        ].copy()
        fs_agg = fs.groupby("account_id", as_index=False)["credit_share"].sum()
        fs_agg["credit_share"] = fs_agg["credit_share"].clip(upper=1.0)
        df = df.merge(fs_agg, on="account_id", how="inner")
        df["current_revenue"] = df["current_revenue"] * df["credit_share"]
        df["prior_revenue"] = df["prior_revenue"] * df["credit_share"]
        df["delta_revenue"] = df["current_revenue"] - df["prior_revenue"]
        df["change_pct"] = df.apply(
            lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
            if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
            axis=1,
        )
        df["abs_change_pct"] = df["change_pct"].abs()
        df = df.drop(columns=["credit_share"])
        scope_label = f"Feature: {', '.join(selected_features)}"

    elif selected_products:
        # Aggregate product-level revenue for selected categories per account
        pc_agg = (
            product_consumption[
                product_consumption["product_category"].isin(selected_products)
            ]
            .groupby("account_id", as_index=False)
            .agg(
                current_revenue=("monthly_revenue", "sum"),
                prior_revenue=("monthly_revenue_1m_ago", "sum"),
                avg_daily_current=("avg_daily_revenue", "sum"),
                avg_daily_3m=("avg_daily_revenue_3m_ago", "sum"),
                avg_daily_6m=("avg_daily_revenue_6m_ago", "sum"),
            )
        )
        pc_agg["delta_revenue"] = pc_agg["current_revenue"] - pc_agg["prior_revenue"].fillna(0)
        pc_agg["change_pct"] = pc_agg.apply(
            lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
            if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
            axis=1,
        )

        # Overlay onto df
        overlay = pc_agg[["account_id", "current_revenue", "prior_revenue", "delta_revenue", "change_pct"]]
        df = df.drop(columns=["current_revenue", "prior_revenue", "delta_revenue", "change_pct", "abs_change_pct"])
        df = df.merge(overlay, on="account_id", how="inner")
        df["abs_change_pct"] = df["change_pct"].abs()
        scope_label = f"Product: {', '.join(selected_products)}"

    elif selected_func_areas:
        # Use credit-share allocation: functional area % of credits × BOB trailing revenue
        fa = func_area_shares[
            func_area_shares["functional_area"].isin(selected_func_areas)
        ].copy()
        # Sum share across selected functional areas per account
        fa_agg = fa.groupby("account_id", as_index=False)["credit_share"].sum()
        # Cap at 1.0 (shouldn't exceed, but safety)
        fa_agg["credit_share"] = fa_agg["credit_share"].clip(upper=1.0)
        # Merge share into df and scale revenue columns
        df = df.merge(fa_agg, on="account_id", how="inner")
        df["current_revenue"] = df["current_revenue"] * df["credit_share"]
        df["prior_revenue"] = df["prior_revenue"] * df["credit_share"]
        df["delta_revenue"] = df["current_revenue"] - df["prior_revenue"]
        df["change_pct"] = df.apply(
            lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
            if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
            axis=1,
        )
        df["abs_change_pct"] = df["change_pct"].abs()
        df = df.drop(columns=["credit_share"])
        scope_label = f"Functional Area: {', '.join(selected_func_areas)}"

    # Owner / account search
    if owner_search:
        mask = (
            df["account_name"].str.contains(owner_search, case=False, na=False)
            | df["account_owner_name"].str.contains(owner_search, case=False, na=False)
            | df["lead_sales_engineer_name"].str.contains(owner_search, case=False, na=False)
        )
        df = df[mask]

    # Derive engagement gap flag
    df["last_account_meeting_date"] = pd.to_datetime(
        df["last_account_meeting_date"], errors="coerce"
    )
    today = pd.Timestamp.now().normalize()
    df["days_since_meeting"] = (today - df["last_account_meeting_date"]).dt.days
    df["engagement_gap"] = df["days_since_meeting"] >= 30


    # ── Page header ──────────────────────────────────────────────────────────────

    _hdr_col, _btn_col = st.columns([8, 1])
    _hdr_col.markdown("# :material/trending_up: Overlay Dashboard")
    if _btn_col.button(":material/restart_alt: Reset", type="secondary"):
        st.session_state.clear()
        st.rerun()

    if scope_label:
        st.info(f"Consumption scoped to **{scope_label}** — revenue figures reflect this slice only.")


    # ── Page Routing ─────────────────────────────────────────────────────────────


    # ══════════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Lights On / Lights Off
    # ══════════════════════════════════════════════════════════════════════════════

    if page == "Lights On / Lights Off":

        # ── Page-specific filters ───────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
        with fc1:
            time_period = st.selectbox(
                "Comparison Period",
                list(PERIOD_MAP.keys()),
                index=0,
                key="lo_time_period",
            )
        with fc2:
            direction = st.radio(
                "Change Direction",
                ["All", "Lights On (Up)", "Lights Off (Down)"],
                index=0,
                key="lo_direction",
            )
        with fc3:
            threshold = st.slider(
                "Min % Change Threshold",
                min_value=0,
                max_value=100,
                value=DEFAULT_THRESHOLD,
                step=1,
                help="Only show accounts with absolute % change above this threshold",
                key="lo_threshold",
            )
        with fc4:
            urgent_filter = st.checkbox(
                "Urgent Decliner, High ACV Only",
                key="lo_urgent",
            )

        # ── Build page-local dataframe with selected comparison period ───────
        cols = PERIOD_MAP[time_period]
        ldf = df.copy()
        ldf["current_revenue"] = ldf[cols["current"]]
        ldf["prior_revenue"] = ldf[cols["prior"]]
        ldf["delta_revenue"] = ldf[cols["delta"]]
        ldf["change_pct"] = ldf.apply(lambda r: compute_change_pct(r, cols), axis=1)
        ldf["abs_change_pct"] = ldf["change_pct"].abs()

        # Re-apply scope overlay with user-selected period for Product filter
        if selected_products:
            pc_agg = (
                product_consumption[
                    product_consumption["product_category"].isin(selected_products)
                ]
                .groupby("account_id", as_index=False)
                .agg(
                    current_revenue=("monthly_revenue", "sum"),
                    prior_revenue=("monthly_revenue_1m_ago", "sum"),
                    avg_daily_current=("avg_daily_revenue", "sum"),
                    avg_daily_3m=("avg_daily_revenue_3m_ago", "sum"),
                    avg_daily_6m=("avg_daily_revenue_6m_ago", "sum"),
                )
            )
            pc_agg["delta_revenue"] = pc_agg["current_revenue"] - pc_agg["prior_revenue"].fillna(0)
            pc_agg["change_pct"] = pc_agg.apply(
                lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
                if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
                axis=1,
            )
            if time_period in ("90 Days", "180 Days"):
                prior_col = "avg_daily_3m" if time_period == "90 Days" else "avg_daily_6m"
                pc_agg["change_pct"] = pc_agg.apply(
                    lambda r: (r["avg_daily_current"] - r[prior_col]) / abs(r[prior_col])
                    if pd.notna(r[prior_col]) and r[prior_col] != 0 else None,
                    axis=1,
                )
                pc_agg["prior_revenue"] = pc_agg[prior_col] * (90 if time_period == "90 Days" else 180)
                pc_agg["current_revenue"] = pc_agg["avg_daily_current"] * (90 if time_period == "90 Days" else 180)
                pc_agg["delta_revenue"] = pc_agg["current_revenue"] - pc_agg["prior_revenue"]
            overlay = pc_agg[["account_id", "current_revenue", "prior_revenue", "delta_revenue", "change_pct"]]
            ldf = ldf.drop(columns=["current_revenue", "prior_revenue", "delta_revenue", "change_pct", "abs_change_pct"])
            ldf = ldf.merge(overlay, on="account_id", how="inner")
            ldf["abs_change_pct"] = ldf["change_pct"].abs()

        elif selected_func_areas:
            fa = func_area_shares[
                func_area_shares["functional_area"].isin(selected_func_areas)
            ].copy()
            fa_agg = fa.groupby("account_id", as_index=False)["credit_share"].sum()
            fa_agg["credit_share"] = fa_agg["credit_share"].clip(upper=1.0)
            ldf = ldf.merge(fa_agg, on="account_id", how="inner")
            ldf["current_revenue"] = ldf["current_revenue"] * ldf["credit_share"]
            ldf["prior_revenue"] = ldf["prior_revenue"] * ldf["credit_share"]
            ldf["delta_revenue"] = ldf["current_revenue"] - ldf["prior_revenue"]
            ldf["change_pct"] = ldf.apply(
                lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
                if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
                axis=1,
            )
            ldf["abs_change_pct"] = ldf["change_pct"].abs()
            ldf = ldf.drop(columns=["credit_share"])

        elif selected_features:
            fs = feature_shares[
                feature_shares["feature"].isin(selected_features)
            ].copy()
            fs_agg = fs.groupby("account_id", as_index=False)["credit_share"].sum()
            fs_agg["credit_share"] = fs_agg["credit_share"].clip(upper=1.0)
            ldf = ldf.merge(fs_agg, on="account_id", how="inner")
            ldf["current_revenue"] = ldf["current_revenue"] * ldf["credit_share"]
            ldf["prior_revenue"] = ldf["prior_revenue"] * ldf["credit_share"]
            ldf["delta_revenue"] = ldf["current_revenue"] - ldf["prior_revenue"]
            ldf["change_pct"] = ldf.apply(
                lambda r: (r["current_revenue"] - r["prior_revenue"]) / abs(r["prior_revenue"])
                if pd.notna(r["prior_revenue"]) and r["prior_revenue"] != 0 else None,
                axis=1,
            )
            ldf["abs_change_pct"] = ldf["change_pct"].abs()
            ldf = ldf.drop(columns=["credit_share"])

        # Apply threshold
        ldf = ldf.dropna(subset=["change_pct"])
        ldf = ldf[ldf["abs_change_pct"] >= threshold / 100]

        # Apply direction filter
        if direction == "Lights On (Up)":
            ldf = ldf[ldf["change_pct"] > 0]
        elif direction == "Lights Off (Down)":
            ldf = ldf[ldf["change_pct"] < 0]

        # Apply urgent filter
        if urgent_filter:
            acv_p75 = bob["acv_purchased"].quantile(0.75)
            ldf = ldf[(ldf["change_pct"] < 0) & (ldf["acv_purchased"].fillna(0) >= acv_p75)]

        # ── KPI Row ──────────────────────────────────────────────────────────────

        lights_on = ldf[ldf["change_pct"] > 0]
        lights_off = ldf[ldf["change_pct"] < 0]
        total_accounts_in_bob = len(bob)
        significant_count = len(ldf)
        alerts_count = int(ldf["consumption_risk"].fillna(False).sum())

        with st.container():
            st.metric(
                "Lights On (Growth)",
                f"{len(lights_on)} accounts",
                fmt_currency(lights_on["delta_revenue"].sum()),
            )
            st.metric(
                "Lights Off (Decline)",
                f"{len(lights_off)} accounts",
                fmt_currency(lights_off["delta_revenue"].sum()),
                delta_color="inverse",
            )
            st.metric(
                "Net Consumption Change",
                fmt_currency(ldf["delta_revenue"].sum()),
            )
            st.metric(
                "Significant Changes",
                f"{significant_count} / {total_accounts_in_bob}",
                f"{significant_count / max(total_accounts_in_bob, 1) * 100:.2f}% of BoB",
            )
            st.metric(
                "Risk Alerts",
                f"{alerts_count}",
            )

        # ── Section A: Leaderboard ───────────────────────────────────────────────

        st.divider()
        st.subheader(":material/leaderboard: Lights On & Lights Off Leaderboard")

        top_n = st.slider("Top N movers", min_value=5, max_value=50, value=10, key="top_n")

        col_on, col_off = st.columns(2)

        # Top Lights On
        with col_on:
            with st.container():
                st.markdown("**Top Lights On (Growth)**")
                top_on = (
                    lights_on.nlargest(top_n, "delta_revenue")[
                        ["account_name", "delta_revenue", "change_pct",
                         "prior_revenue", "current_revenue",
                         "account_owner_name", "region", "industry"]
                    ]
                    .reset_index(drop=True)
                )
                if not top_on.empty:
                    chart_on = (
                        alt.Chart(top_on.head(top_n))
                        .mark_bar(color="#2ecc71")
                        .encode(
                            y=alt.Y("account_name:N", sort="-x", title=None),
                            x=alt.X("delta_revenue:Q", title="Revenue Change ($)"),
                            tooltip=[
                                alt.Tooltip("account_name:N", title="Account"),
                                alt.Tooltip("delta_revenue:Q", title="Revenue Change", format="$,.2f"),
                                alt.Tooltip("change_pct:Q", title="Revenue Change %", format=".2%"),
                            ],
                        )
                        .properties(height=max(top_n * 28, 200))
                    )
                    st.altair_chart(chart_on, use_container_width=True)
                else:
                    st.info("No upward movers matching filters.")

        # Top Lights Off
        with col_off:
            with st.container():
                st.markdown("**Top Lights Off (Decline)**")
                top_off = (
                    lights_off.nsmallest(top_n, "delta_revenue")[
                        ["account_name", "delta_revenue", "change_pct",
                         "prior_revenue", "current_revenue",
                         "account_owner_name", "region", "industry"]
                    ]
                    .reset_index(drop=True)
                )
                if not top_off.empty:
                    top_off_chart = top_off.copy()
                    top_off_chart["abs_delta"] = top_off_chart["delta_revenue"].abs()
                    chart_off = (
                        alt.Chart(top_off_chart.head(top_n))
                        .mark_bar(color="#e74c3c")
                        .encode(
                            y=alt.Y("account_name:N", sort="-x", title=None),
                            x=alt.X("abs_delta:Q", title="Revenue Decline ($)"),
                            tooltip=[
                                alt.Tooltip("account_name:N", title="Account"),
                                alt.Tooltip("delta_revenue:Q", title="Revenue Change", format="$,.2f"),
                                alt.Tooltip("change_pct:Q", title="Revenue Change %", format=".2%"),
                            ],
                        )
                        .properties(height=max(top_n * 28, 200))
                    )
                    st.altair_chart(chart_off, use_container_width=True)
                else:
                    st.info("No downward movers matching filters.")

        # Leaderboard data table
        with st.container():
            st.markdown("**Detailed Leaderboard**")
            leaderboard = (
                ldf.sort_values("abs_change_pct", ascending=False)
                .head(top_n * 2)[
                    ["account_name", "account_owner_name", "lead_sales_engineer_name",
                     "region", "current_revenue", "prior_revenue",
                     "delta_revenue", "change_pct"]
                ]
                .reset_index(drop=True)
            )
            leaderboard["change_pct"] = leaderboard["change_pct"] * 100
            st.dataframe(
                leaderboard,
                column_config={
                    "account_name": st.column_config.TextColumn("Account"),
                    "account_owner_name": st.column_config.TextColumn("Account Owner"),
                    "lead_sales_engineer_name": st.column_config.TextColumn("Sales Engineer"),
                    "region": st.column_config.TextColumn("Region"),
                    "current_revenue": st.column_config.NumberColumn("Current Period Revenue", format="dollar", step=0.01),
                    "prior_revenue": st.column_config.NumberColumn("Prior Period Revenue", format="dollar", step=0.01),
                    "delta_revenue": st.column_config.NumberColumn("Revenue Change", format="dollar", step=0.01),
                    "change_pct": st.column_config.NumberColumn("Revenue Change %", format="%.2f%%"),
                },
                hide_index=True,
                use_container_width=True,
            )

        # ── Section B: Root Cause & Engagement ───────────────────────────────────

        st.divider()
        st.subheader(":material/search_insights: Root Cause & Engagement Details")

        # Merge risk info
        detail = ldf.merge(
            risk_movements[["account_id", "account_comments_c",
                             "consumption_risk_mitigation_steps_c",
                             "consumption_risk_c"]],
            on="account_id", how="left"
        )

        # Build detail table
        detail_display = (
            detail.sort_values("abs_change_pct", ascending=False)[
                ["account_name", "change_pct", "delta_revenue",
                 "consumption_context",
                 "account_owner_name", "lead_sales_engineer_name",
                 "last_account_meeting_date", "days_since_meeting",
                 "engagement_gap", "active_use_cases",
                 "in_pursuit_use_cases", "in_implementation_use_cases",
                 "high_risk_use_cases", "se_engagement_context"]
            ]
            .reset_index(drop=True)
        )

        # Flag urgent decliners with no recent engagement
        detail_display["alert"] = detail_display.apply(
            lambda r: "Decliner, No Contact 30+ days"
            if r["change_pct"] < 0 and r.get("engagement_gap", False)
            else "",
            axis=1,
        )

        # Convert to display-friendly percentage (multiply by 100)
        detail_display["change_pct"] = detail_display["change_pct"] * 100

        with st.container():
            st.markdown("**Engagement & Risk Overview**")
            st.dataframe(
                detail_display,
                column_config={
                    "account_name": st.column_config.TextColumn("Account"),
                    "change_pct": st.column_config.NumberColumn("Revenue Change %", format="%.2f%%"),
                    "delta_revenue": st.column_config.NumberColumn("Revenue Change ($)", format="dollar", step=0.01),
                    "consumption_context": st.column_config.TextColumn("Consumption Context", width="large"),
                    "account_owner_name": st.column_config.TextColumn("AE"),
                    "lead_sales_engineer_name": st.column_config.TextColumn("Lead SE"),
                    "last_account_meeting_date": st.column_config.DateColumn("Last Meeting", format="MMM DD, YYYY"),
                    "days_since_meeting": st.column_config.NumberColumn("Days Since Meeting"),
                    "engagement_gap": None,
                    "active_use_cases": st.column_config.NumberColumn("Active UCs"),
                    "in_pursuit_use_cases": st.column_config.NumberColumn("Pursuit UCs"),
                    "in_implementation_use_cases": st.column_config.NumberColumn("Impl UCs"),
                    "high_risk_use_cases": st.column_config.NumberColumn("High Risk UCs"),
                    "se_engagement_context": st.column_config.TextColumn("SE Engagement"),
                    "alert": st.column_config.TextColumn("Alert"),
                },
                hide_index=True,
                use_container_width=True,
                height=500,
            )

        # ── Section C: Open Use Cases for Selected Accounts ──────────────────────

        st.divider()
        st.subheader(":material/cases: Open Use Cases on Filtered Accounts")

        filtered_account_ids = ldf["account_id"].unique().tolist()
        account_use_cases = use_cases[use_cases["account_id"].isin(filtered_account_ids)]

        # Let user pick a specific account to drill into
        account_options = sorted(ldf["account_name"].dropna().unique().tolist())
        selected_account = st.selectbox(
            "Drill into account", ["All filtered accounts"] + account_options
        )

        if selected_account != "All filtered accounts":
            acct_id = ldf.loc[ldf["account_name"] == selected_account, "account_id"].iloc[0]
            account_use_cases = account_use_cases[account_use_cases["account_id"] == acct_id]

        if account_use_cases.empty:
            st.info("No open use cases for the selected account(s).")
        else:
            uc_display = (
                account_use_cases.merge(
                    ldf[["account_id", "account_name"]].drop_duplicates(),
                    on="account_id", how="left",
                )[
                    ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
                     "health_status", "industry_use_case", "prioritized_feature",
                     "workloads", "risk_factors", "days_in_stage", "use_case_lead_se_name"]
                ]
                .sort_values(["account_name", "use_case_eacv"], ascending=[True, False])
                .reset_index(drop=True)
            )
            with st.container():
                st.dataframe(
                    uc_display,
                    column_config={
                        "account_name": st.column_config.TextColumn("Account"),
                        "use_case_name": st.column_config.TextColumn("Use Case"),
                        "use_case_stage": st.column_config.TextColumn("Stage"),
                        "use_case_eacv": st.column_config.NumberColumn("EACV", format="dollar", step=0.01),
                        "health_status": st.column_config.TextColumn("Health"),
                        "industry_use_case": st.column_config.TextColumn("Industry / Functional Use Case", width="large"),
                        "prioritized_feature": st.column_config.TextColumn("Prioritized Feature(s)", width="large"),
                        "workloads": st.column_config.TextColumn("Workloads"),
                        "risk_factors": st.column_config.TextColumn("Risk Factors", width="large"),
                        "days_in_stage": st.column_config.NumberColumn("Days in Stage"),
                        "use_case_lead_se_name": st.column_config.TextColumn("Lead SE"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )


    # ══════════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Consumption Intelligence
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Consumption Intelligence":

        # ── Filter daily consumption to match sidebar selections ─────────────────
        dc = daily_consumption.copy()
        if selected_regions:
            dc = dc[dc["region"].isin(selected_regions)]
        if selected_theaters:
            dc = dc[dc["theater"].isin(selected_theaters)]
        # Restrict to accounts in the filtered BOB set
        bob_account_ids = df["account_id"].unique()
        dc = dc[dc["account_id"].isin(bob_account_ids)]

        # Filter monthly trend to matched accounts
        mt = monthly_trend[monthly_trend["account_id"].isin(bob_account_ids)].copy()
        if selected_products:
            mt = mt[mt["product_category"].isin(selected_products)]

        # Apply scope overlay to monthly trend data (scale revenue by credit share)
        if selected_features:
            fs = feature_shares[
                feature_shares["feature"].isin(selected_features)
            ].copy()
            fs_agg_mt = fs.groupby("account_id", as_index=False)["credit_share"].sum()
            fs_agg_mt["credit_share"] = fs_agg_mt["credit_share"].clip(upper=1.0)
            mt = mt.merge(fs_agg_mt, on="account_id", how="inner")
            mt["monthly_revenue"] = mt["monthly_revenue"] * mt["credit_share"]
            mt = mt.drop(columns=["credit_share"])
        elif selected_func_areas:
            # Use month-specific credit shares for mt
            mfa = monthly_func_area_shares[
                monthly_func_area_shares["functional_area"].isin(selected_func_areas)
            ].copy()
            mfa_agg = mfa.groupby(["account_id", "share_month"], as_index=False)["credit_share"].sum()
            mfa_agg["credit_share"] = mfa_agg["credit_share"].clip(upper=1.0)
            mt = mt.merge(
                mfa_agg,
                left_on=["account_id", "consumption_month"],
                right_on=["account_id", "share_month"],
                how="inner",
            )
            mt["monthly_revenue"] = mt["monthly_revenue"] * mt["credit_share"]
            mt = mt.drop(columns=["credit_share", "share_month"])

        # ── Prepare BOB-level aggregates for KPIs ────────────────────────────────
        bob_filtered = bob[bob["account_id"].isin(bob_account_ids)].copy()

        # Join district_name from indicators onto bob_filtered
        bob_filtered = bob_filtered.merge(
            indicators[["account_id", "district_name"]],
            on="account_id", how="left",
        )

        # ── Apply consumption scope overlay (same priority as Tab 1) ──────────
        # Revenue columns to scale when a scope filter is active
        _tab2_rev_cols = [
            "current_fiscal_quarter_revenue",
            "previous_fiscal_quarter_revenue",
            "revenue_prediction_current_fiscal_quarter",
            "revenue_prediction_cfyq1",
            "revenue_prediction_cfyq2",
            "revenue_prediction_cfyq3",
            "revenue_prediction_cfyq4",
            "current_fiscal_year_revenue",
            "previous_fiscal_year_revenue",
        ]

        if selected_features:
            fs = feature_shares[
                feature_shares["feature"].isin(selected_features)
            ].copy()
            fs_agg = fs.groupby("account_id", as_index=False)["credit_share"].sum()
            fs_agg["credit_share"] = fs_agg["credit_share"].clip(upper=1.0)
            bob_filtered = bob_filtered.merge(fs_agg, on="account_id", how="inner")
            for col in _tab2_rev_cols:
                if col in bob_filtered.columns:
                    bob_filtered[col] = bob_filtered[col] * bob_filtered["credit_share"]
            bob_filtered = bob_filtered.drop(columns=["credit_share"])

        elif selected_products:
            # Derive product-category credit shares from feature_shares
            ps = feature_shares[
                feature_shares["product_category"].isin(selected_products)
            ].copy()
            ps_agg = ps.groupby("account_id", as_index=False)["credit_share"].sum()
            ps_agg["credit_share"] = ps_agg["credit_share"].clip(upper=1.0)
            bob_filtered = bob_filtered.merge(ps_agg, on="account_id", how="inner")
            for col in _tab2_rev_cols:
                if col in bob_filtered.columns:
                    bob_filtered[col] = bob_filtered[col] * bob_filtered["credit_share"]
            bob_filtered = bob_filtered.drop(columns=["credit_share"])

        elif selected_func_areas:
            fa = func_area_shares[
                func_area_shares["functional_area"].isin(selected_func_areas)
            ].copy()
            fa_agg = fa.groupby("account_id", as_index=False)["credit_share"].sum()
            fa_agg["credit_share"] = fa_agg["credit_share"].clip(upper=1.0)
            bob_filtered = bob_filtered.merge(fa_agg, on="account_id", how="inner")
            for col in _tab2_rev_cols:
                if col in bob_filtered.columns:
                    bob_filtered[col] = bob_filtered[col] * bob_filtered["credit_share"]
            bob_filtered = bob_filtered.drop(columns=["credit_share"])

        total_qtd = bob_filtered["current_fiscal_year_revenue"].sum()
        prev_fq = bob_filtered["previous_fiscal_quarter_revenue"].sum()
        total_predicted = bob_filtered["revenue_prediction_current_fiscal_quarter"].sum()
        active_accounts = int(bob_filtered["current_fiscal_year_revenue"].gt(0).sum())

        qoq_growth = (total_predicted - prev_fq) / abs(prev_fq) if prev_fq and prev_fq != 0 else 0

        # ── KPIs ─────────────────────────────────────────────────────────────────

        st.subheader(":material/monitoring: Consumption Intelligence — Executive Summary")

        with st.container():
            st.metric(
                "Total QTD Consumption",
                fmt_currency(total_qtd),
            )
            st.metric(
                "QoQ Growth",
                fmt_pct(qoq_growth),
            )
            st.metric(
                "Predicted FQ Finish",
                fmt_currency(total_predicted),
            )
            st.metric(
                "Active Accounts",
                f"{active_accounts:,}",
            )

        # ── Section A: Consumption Trends & Forecasting ──────────────────────────

        st.divider()
        st.subheader(":material/insights: Section A — Consumption Trends & Forecasting")

        # A1: Stacked bar — Actual vs Predicted Revenue by Quarter
        # Previous 2 FQ actuals, Current FQ (actual stacked with predicted), Next 3 FQ predicted
        prev2_fq_total = prev2_fq_revenue["prev2_fq_revenue"].iloc[0] if not prev2_fq_revenue.empty else 0
        prev_fq_total = bob_filtered["previous_fiscal_quarter_revenue"].sum()

        # Scale prev2_fq_total when scope filters are active.
        # Use the ratio of scaled-to-unscaled previous_fq revenue as a proxy.
        if selected_features or selected_products or selected_func_areas:
            unscaled_prev_fq = bob[bob["account_id"].isin(bob_account_ids)]["previous_fiscal_quarter_revenue"].sum()
            scope_ratio = prev_fq_total / abs(unscaled_prev_fq) if unscaled_prev_fq and unscaled_prev_fq != 0 else 0
            prev2_fq_total = prev2_fq_total * scope_ratio
        curr_fq_actual = bob_filtered["current_fiscal_year_revenue"].sum()
        curr_fq_predicted = bob_filtered["revenue_prediction_cfyq1"].sum()
        q2_predicted = bob_filtered["revenue_prediction_cfyq2"].sum()
        q3_predicted = bob_filtered["revenue_prediction_cfyq3"].sum()
        q4_predicted = bob_filtered["revenue_prediction_cfyq4"].sum()

        quarterly_data = pd.DataFrame([
            {"quarter": "FQ-2", "series": "Actual", "revenue": prev2_fq_total},
            {"quarter": "FQ-1", "series": "Actual", "revenue": prev_fq_total},
            {"quarter": "Current FQ", "series": "Actual", "revenue": curr_fq_actual},
            {"quarter": "Current FQ", "series": "Predicted", "revenue": curr_fq_predicted - curr_fq_actual},
            {"quarter": "FQ+1", "series": "Predicted", "revenue": q2_predicted},
            {"quarter": "FQ+2", "series": "Predicted", "revenue": q3_predicted},
            {"quarter": "FQ+3", "series": "Predicted", "revenue": q4_predicted},
        ])
        # Ensure no negative predicted remainder
        quarterly_data.loc[quarterly_data["revenue"] < 0, "revenue"] = 0

        color_scale = alt.Scale(
            domain=["Actual", "Predicted"],
            range=["#2563eb", "#f59e0b"],
        )

        quarter_order = ["FQ-2", "FQ-1", "Current FQ", "FQ+1", "FQ+2", "FQ+3"]

        quarterly_bar = (
            alt.Chart(quarterly_data)
            .mark_bar()
            .encode(
                x=alt.X("quarter:N", title="Quarter", sort=quarter_order,
                         axis=alt.Axis(labelAngle=0)),
                y=alt.Y("revenue:Q", title="Revenue ($)", stack="zero"),
                color=alt.Color("series:N", scale=color_scale, title=""),
                order=alt.Order("series:N", sort="descending"),
                tooltip=[
                    alt.Tooltip("quarter:N", title="Quarter"),
                    alt.Tooltip("series:N", title="Type"),
                    alt.Tooltip("revenue:Q", title="Revenue", format="$,.2f"),
                ],
            )
            .properties(height=CHART_HEIGHT, title="Quarterly Revenue: Actual vs Predicted")
        )

        with st.container():
            st.altair_chart(quarterly_bar, use_container_width=True)

        # A2: Stacked bar — Monthly consumption by product category
        if not mt.empty:
            monthly_product_agg = (
                mt.groupby(["consumption_month", "product_category"], as_index=False)
                .agg(revenue=("monthly_revenue", "sum"))
            )

            stacked_bar = (
                alt.Chart(monthly_product_agg)
                .mark_bar()
                .encode(
                    x=alt.X("yearmonth(consumption_month):T", title="Month"),
                    y=alt.Y("revenue:Q", title="Monthly Revenue ($)", stack="zero"),
                    color=alt.Color("product_category:N", title="Product Category"),
                    tooltip=[
                        alt.Tooltip("yearmonth(consumption_month):T", title="Month"),
                        alt.Tooltip("product_category:N", title="Product"),
                        alt.Tooltip("revenue:Q", title="Revenue", format="$,.2f"),
                    ],
                )
                .properties(height=CHART_HEIGHT, title="Monthly Consumption by Product Category")
            )

            with st.container():
                st.altair_chart(stacked_bar, use_container_width=True)
        else:
            st.info("No monthly trend data available for the current filters.")

        # ── Section B: Account-Level Consumption Details ─────────────────────────

        st.divider()
        st.subheader(":material/table_chart: Section B — Consumption Details")

        # Group-by selector — controls Sections B, C, and D
        GROUP_BY_OPTIONS = {
            "Account": {"col": "account_name", "label": "Account"},
            "Owner": {"col": "account_owner_name", "label": "Account Owner"},
            "District": {"col": "district_name", "label": "District"},
            "Region": {"col": "region", "label": "Region"},
            "Theater": {"col": "theater", "label": "Theater"},
        }
        group_by_choice = st.selectbox(
            "Group rows by",
            list(GROUP_BY_OPTIONS.keys()),
            index=0,
            key="section_b_group_by",
        )
        group_cfg = GROUP_BY_OPTIONS[group_by_choice]
        group_col = group_cfg["col"]

        # Build Section B source data with all needed columns
        _rev_cols_b = [
            "previous_fiscal_quarter_revenue", "current_fiscal_year_revenue",
            "revenue_prediction_current_fiscal_quarter",
            "revenue_prediction_cfyq2", "revenue_prediction_cfyq3", "revenue_prediction_cfyq4",
        ]
        acct_detail = bob_filtered[[
            "account_name", "account_id", "account_owner_name",
            "region", "district_name", "theater",
        ] + _rev_cols_b + ["date_of_last_consumption"]].copy()

        if group_by_choice == "Account":
            # Account-level: compute QoQ per row
            acct_detail["qoq_growth"] = np.where(
                acct_detail["previous_fiscal_quarter_revenue"].abs() > 0,
                (acct_detail["revenue_prediction_current_fiscal_quarter"]
                 - acct_detail["previous_fiscal_quarter_revenue"])
                / acct_detail["previous_fiscal_quarter_revenue"].abs(),
                np.nan,
            )
            acct_detail = acct_detail.sort_values(
                "current_fiscal_year_revenue", ascending=False
            ).reset_index(drop=True)

            acct_display = acct_detail[[
                "account_name", "region", "district_name",
            ] + _rev_cols_b[:3] + ["qoq_growth"] + _rev_cols_b[3:] + [
                "date_of_last_consumption",
            ]].copy()
            acct_display["qoq_growth"] = acct_display["qoq_growth"] * 100

            col_config_b = {
                "account_name": st.column_config.TextColumn("Account"),
                "region": st.column_config.TextColumn("Region"),
                "district_name": st.column_config.TextColumn("District"),
            }
        else:
            # Grouped: aggregate revenue columns, compute QoQ from aggregates
            acct_detail["date_of_last_consumption"] = pd.to_datetime(
                acct_detail["date_of_last_consumption"], errors="coerce"
            )
            grp = acct_detail.groupby(group_col, as_index=False).agg(
                previous_fiscal_quarter_revenue=("previous_fiscal_quarter_revenue", "sum"),
                current_fiscal_year_revenue=("current_fiscal_year_revenue", "sum"),
                revenue_prediction_current_fiscal_quarter=("revenue_prediction_current_fiscal_quarter", "sum"),
                revenue_prediction_cfyq2=("revenue_prediction_cfyq2", "sum"),
                revenue_prediction_cfyq3=("revenue_prediction_cfyq3", "sum"),
                revenue_prediction_cfyq4=("revenue_prediction_cfyq4", "sum"),
                date_of_last_consumption=("date_of_last_consumption", "max"),
            )
            grp["qoq_growth"] = np.where(
                grp["previous_fiscal_quarter_revenue"].abs() > 0,
                (grp["revenue_prediction_current_fiscal_quarter"]
                 - grp["previous_fiscal_quarter_revenue"])
                / grp["previous_fiscal_quarter_revenue"].abs(),
                np.nan,
            )
            grp = grp.sort_values(
                "current_fiscal_year_revenue", ascending=False
            ).reset_index(drop=True)

            acct_display = grp[[
                group_col,
            ] + _rev_cols_b[:3] + ["qoq_growth"] + _rev_cols_b[3:] + [
                "date_of_last_consumption",
            ]].copy()
            acct_display["qoq_growth"] = acct_display["qoq_growth"] * 100

            col_config_b = {
                group_col: st.column_config.TextColumn(group_cfg["label"]),
            }

        # Shared column config for revenue/date columns
        col_config_b.update({
            "previous_fiscal_quarter_revenue": st.column_config.NumberColumn(
                "Prev FQ Revenue", format="dollar", step=0.01
            ),
            "current_fiscal_year_revenue": st.column_config.NumberColumn(
                "QTD Revenue", format="dollar", step=0.01
            ),
            "revenue_prediction_current_fiscal_quarter": st.column_config.NumberColumn(
                "Predicted FQ Finish", format="dollar", step=0.01
            ),
            "qoq_growth": st.column_config.NumberColumn("QoQ Growth %", format="%.2f%%"),
            "revenue_prediction_cfyq2": st.column_config.NumberColumn(
                "FQ + 1 Predicted", format="dollar", step=0.01
            ),
            "revenue_prediction_cfyq3": st.column_config.NumberColumn(
                "FQ + 2 Predicted", format="dollar", step=0.01
            ),
            "revenue_prediction_cfyq4": st.column_config.NumberColumn(
                "FQ + 3 Predicted", format="dollar", step=0.01
            ),
            "date_of_last_consumption": st.column_config.DateColumn(
                "Last Consumption", format="MMM DD, YYYY"
            ),
        })

        with st.container():
            st.dataframe(
                acct_display,
                column_config=col_config_b,
                hide_index=True,
                use_container_width=True,
                height=500,
            )

        # ── Section C: Monthly Consumption Table ──────────────────────────────────

        st.divider()
        st.subheader(":material/calendar_month: Section C — Monthly Consumption")

        if not mt.empty:
            # Join the grouping column from bob_filtered onto monthly trend data
            meta_cols = ["account_id", "account_name", "account_owner_name",
                         "district_name", "region", "theater"]
            acct_meta = bob_filtered[[c for c in meta_cols if c in bob_filtered.columns]].drop_duplicates()
            mt_with_meta = mt.merge(acct_meta, on="account_id", how="left")

            # Aggregate revenue by group column + month
            mt_agg = (
                mt_with_meta.groupby([group_col, "consumption_month"], as_index=False)
                .agg(monthly_revenue=("monthly_revenue", "sum"))
            )

            # Compute unscoped monthly totals per group row for % calculation
            mt_unscoped = monthly_trend[
                monthly_trend["account_id"].isin(bob_account_ids)
            ].copy()
            mt_unscoped_with_meta = mt_unscoped.merge(acct_meta, on="account_id", how="left")
            unscoped_agg = (
                mt_unscoped_with_meta.groupby([group_col, "consumption_month"], as_index=False)
                .agg(total_revenue=("monthly_revenue", "sum"))
            )
            unscoped_pivot = unscoped_agg.pivot_table(
                index=group_col,
                columns="consumption_month",
                values="total_revenue",
                aggfunc="sum",
            ).fillna(0)

            # Pivot: rows = group, columns = months
            mt_pivot = mt_agg.pivot_table(
                index=group_col,
                columns="consumption_month",
                values="monthly_revenue",
                aggfunc="sum",
            ).fillna(0)

            # Store raw month timestamps before renaming
            raw_month_cols = list(mt_pivot.columns)
            month_labels = [c.strftime("%b %Y") for c in raw_month_cols]

            # Build percentage pivot: each cell = scoped / row's unscoped total
            unscoped_aligned = unscoped_pivot.reindex(
                index=mt_pivot.index, columns=raw_month_cols
            ).fillna(0)
            pct_pivot = mt_pivot.copy()
            for month_ts in raw_month_cols:
                pct_pivot[month_ts] = pct_pivot[month_ts].where(
                    unscoped_aligned[month_ts] == 0,
                    mt_pivot[month_ts] / unscoped_aligned[month_ts].replace(0, 1) * 100
                )
                pct_pivot.loc[unscoped_aligned[month_ts] == 0, month_ts] = 0.0

            # Rename columns to month labels
            mt_pivot.columns = month_labels
            pct_pivot.columns = month_labels

            # Add Total columns
            mt_pivot["Total"] = mt_pivot.sum(axis=1)

            # Compute unscoped row totals for percentage Total column
            unscoped_row_totals = unscoped_pivot.reindex(
                index=mt_pivot.index
            ).fillna(0).sum(axis=1)

            # Sort both by Total descending
            mt_pivot = mt_pivot.sort_values("Total", ascending=False).reset_index()
            sort_order = mt_pivot[group_col].tolist()
            pct_pivot = pct_pivot.reset_index()
            pct_pivot = pct_pivot.set_index(group_col).loc[sort_order].reset_index()

            # Compute Total % for Section D (scoped row total / unscoped row total)
            total_pct_vals = []
            for idx_val in mt_pivot[group_col]:
                scoped_row_total = mt_agg[mt_agg[group_col] == idx_val]["monthly_revenue"].sum()
                unscoped_row_total = unscoped_row_totals.get(idx_val, 0)
                total_pct_vals.append(
                    scoped_row_total / unscoped_row_total * 100
                    if unscoped_row_total > 0 else 0.0
                )
            pct_pivot["Total"] = total_pct_vals

            # ── Section C display: dollar values ──
            month_cols = month_labels
            for col in month_cols + ["Total"]:
                mt_pivot[col] = mt_pivot[col].apply(fmt_currency)

            display_cols_c = [group_col] + month_cols + ["Total"]
            col_config_c = {
                group_col: st.column_config.TextColumn(group_cfg["label"]),
                "Total": st.column_config.TextColumn("Total"),
            }
            for mc in month_cols:
                col_config_c[mc] = st.column_config.TextColumn(mc)

            with st.container():
                st.caption(f"Showing **{len(mt_pivot)}** rows grouped by **{group_by_choice}** — monthly consumption over the last 6 months.")
                st.dataframe(
                    mt_pivot[display_cols_c],
                    column_config=col_config_c,
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

            # ── Section D: Percentage of Total Consumption ──
            st.divider()
            st.subheader(":material/percent: Section D — % of Row Total Consumption")

            # Format pct values
            for col in month_cols + ["Total"]:
                pct_pivot[col] = pct_pivot[col].apply(lambda v: f"{v:.1f}%")

            display_cols_d = [group_col] + month_cols + ["Total"]
            col_config_d = {
                group_col: st.column_config.TextColumn(group_cfg["label"]),
                "Total": st.column_config.TextColumn("Total"),
            }
            for mc in month_cols:
                col_config_d[mc] = st.column_config.TextColumn(mc)

            with st.container():
                st.caption(f"Showing **{len(pct_pivot)}** rows grouped by **{group_by_choice}** — each cell shows the filtered consumption as a percentage of that row's total (unscoped) consumption for the month.")
                st.dataframe(
                    pct_pivot[display_cols_d],
                    column_config=col_config_d,
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )
        else:
            st.info("No monthly consumption data available for the current filters.")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE: Use Case
# ══════════════════════════════════════════════════════════════════════════════

elif module == "Use Case":

    if page == "Use Case Forecasting":

        # ── Page header ───────────────────────────────────────────────────────────

        _hdr_col, _btn_col = st.columns([8, 1])
        _hdr_col.markdown("# :material/query_stats: Use Case Forecasting")
        if _btn_col.button(":material/restart_alt: Reset", type="secondary", key="fc_reset"):
            st.session_state.clear()
            st.rerun()

        # ── Apply Forecast filters ────────────────────────────────────────────────

        fp = forecast_pipeline.copy()
        sp = specialist_overlay.copy()

        # Filter to selected fiscal quarter (go-live FQ)
        fp_fq = fp[fp["go_live_fq"] == fc_time_period].copy()

        # Also keep a broader "all active" set for total pipeline KPIs
        fp_all = fp.copy()

        # Apply hierarchy filters
        if fc_selected_theaters:
            fp_fq = fp_fq[fp_fq["theater_name"].isin(fc_selected_theaters)]
            fp_all = fp_all[fp_all["theater_name"].isin(fc_selected_theaters)]
        if fc_selected_regions:
            fp_fq = fp_fq[fp_fq["region_name"].isin(fc_selected_regions)]
            fp_all = fp_all[fp_all["region_name"].isin(fc_selected_regions)]
        if fc_selected_reps:
            fp_fq = fp_fq[fp_fq["rep_name"].isin(fc_selected_reps)]
            fp_all = fp_all[fp_all["rep_name"].isin(fc_selected_reps)]
        if fc_selected_prod_cats:
            _pc_pat = "|".join(re.escape(v) for v in fc_selected_prod_cats)
            _pc_mask_fq = fp_fq["product_category"].fillna("").str.contains(_pc_pat, case=False, regex=True)
            _pc_mask_all = fp_all["product_category"].fillna("").str.contains(_pc_pat, case=False, regex=True)
            fp_fq = fp_fq[_pc_mask_fq]
            fp_all = fp_all[_pc_mask_all]
        if fc_selected_spec_mgrs:
            smh = specialist_mgmt_hierarchy
            _top_mgr_names = set(sp["specialist_manager_name"].dropna().unique())
            # Resolve selected names to specialist names
            _mgr_resolved_spec_names = set()
            for mgr in fc_selected_spec_mgrs:
                if mgr in _top_mgr_names:
                    # Top-level manager: include all specialists under them
                    _mgr_resolved_spec_names.update(
                        sp.loc[sp["specialist_manager_name"] == mgr, "specialist_name"].unique()
                    )
                # First-line manager: include their direct reports (and themselves)
                _reports = smh.loc[smh["workday_manager"] == mgr, "specialist_name"].unique()
                _mgr_resolved_spec_names.update(_reports)
                _mgr_resolved_spec_names.add(mgr)
            spec_mgr_uc_ids = sp.loc[
                sp["specialist_name"].isin(_mgr_resolved_spec_names)
                & (sp["is_team_member"] == 1), "use_case_id"
            ].unique()
            fp_fq = fp_fq[fp_fq["use_case_id"].isin(spec_mgr_uc_ids)]
            fp_all = fp_all[fp_all["use_case_id"].isin(spec_mgr_uc_ids)]
        else:
            _mgr_resolved_spec_names = None
        if fc_selected_tech_uc:
            _tu_pat = "|".join(re.escape(v) for v in fc_selected_tech_uc)
            _tu_mask_fq = fp_fq["technical_use_case"].fillna("").str.contains(_tu_pat, case=False, regex=True)
            _tu_mask_all = fp_all["technical_use_case"].fillna("").str.contains(_tu_pat, case=False, regex=True)
            fp_fq = fp_fq[_tu_mask_fq]
            fp_all = fp_all[_tu_mask_all]
        if fc_selected_features:
            _ft_pat = "|".join(re.escape(v) for v in fc_selected_features)
            _ft_mask_fq = fp_fq["prioritized_features"].fillna("").str.contains(_ft_pat, case=False, regex=True)
            _ft_mask_all = fp_all["prioritized_features"].fillna("").str.contains(_ft_pat, case=False, regex=True)
            fp_fq = fp_fq[_ft_mask_fq]
            fp_all = fp_all[_ft_mask_all]

        # Specialist filter — narrow to use cases where selected specialist is on the team
        if fc_selected_specialists:
            specialist_uc_ids = sp.loc[
                sp["specialist_name"].isin(fc_selected_specialists)
                & (sp["is_team_member"] == 1), "use_case_id"
            ].unique()
            fp_fq = fp_fq[fp_fq["use_case_id"].isin(specialist_uc_ids)]
            fp_all = fp_all[fp_all["use_case_id"].isin(specialist_uc_ids)]

        # Operational filters
        if fc_high_acv_no_tw:
            eacv_p75 = fp_all["use_case_eacv"].quantile(0.75) if len(fp_all) > 0 else 0
            fp_fq = fp_fq[
                (fp_fq["use_case_eacv"].fillna(0) >= eacv_p75)
                & (fp_fq["is_tech_won"] != True)
            ]
            fp_all = fp_all[
                (fp_all["use_case_eacv"].fillna(0) >= eacv_p75)
                & (fp_all["is_tech_won"] != True)
            ]

        # Filter specialist overlay to match filtered use cases (both go-live and wins sets)
        all_uc_ids = set(fp_fq["use_case_id"].unique()) | set(fp_all["use_case_id"].unique())
        sp_filtered = sp[sp["use_case_id"].isin(all_uc_ids)]

        if fc_missing_specialist:
            # Show only use cases where specialist has no comments
            missing_ids = sp_filtered.loc[
                sp_filtered["specialist_comments"].isna()
                | (sp_filtered["specialist_comments"].str.strip() == ""),
                "use_case_id",
            ].unique()
            # Also include use cases with no specialist at all
            no_specialist_ids = set(fp_fq["use_case_id"].unique()) - set(sp["use_case_id"].unique())
            target_ids = set(missing_ids) | no_specialist_ids
            fp_fq = fp_fq[fp_fq["use_case_id"].isin(target_ids)]

        # ── Determine CQ wins vs go-lives ────────────────────────────────────────
        # "Wins" = Decision Date falls within the selected FQ
        # "Go Lives" = Go-Live Date falls within the selected FQ

        # Helper: format EACV as $xx.xM or $xx.xk
        def _fmt_eacv(v):
            if pd.isna(v) or v == 0:
                return "$0"
            abs_v = abs(v)
            if abs_v >= 1_000_000:
                return f"${v / 1_000_000:,.1f}M"
            if abs_v >= 1_000:
                return f"${v / 1_000:,.1f}k"
            return f"${v:,.0f}"

        # Parse FQ string to date range (e.g., FY2027-Q1 → Snowflake FY starts Feb 1)
        # Snowflake fiscal year: FY20XX starts Feb 1 of calendar year 20XX-1
        def _fq_to_date_range(fq_str):
            """Convert 'FY2027-Q1' to (start_date, end_date)."""
            try:
                parts = fq_str.replace("FY", "").split("-Q")
                fy = int(parts[0])
                q = int(parts[1])
                # Snowflake FY starts Feb 1: FY2027-Q1 = Feb 1 2026 – Apr 30 2026
                cal_year = fy - 1
                month_start = (q - 1) * 3 + 2  # Q1→2, Q2→5, Q3→8, Q4→11
                if month_start > 12:
                    month_start -= 12
                    cal_year += 1
                start = pd.Timestamp(year=cal_year, month=month_start, day=1)
                end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
                return start, end
            except Exception:
                return pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-30")

        fq_start, fq_end = _fq_to_date_range(fc_time_period)

        wins = fp_all[
            (fp_all["decision_date"] >= fq_start) & (fp_all["decision_date"] <= fq_end)
        ].copy()

        go_lives = fp_fq.copy()  # Already filtered to go_live_fq == fc_time_period

        # ── KPI Row ───────────────────────────────────────────────────────────────

        # Go-Live bucket KPIs
        gl_open = go_lives[go_lives["stage_number"].between(1, 6)]
        gl_total_pipeline = gl_open["use_case_eacv"].fillna(0).sum()
        gl_weighted_pipeline = go_lives["expected_value"].fillna(0).sum()
        gl_won = go_lives[go_lives["stage_number"] == 7]
        gl_qtd_won_eacv = gl_won["use_case_eacv"].fillna(0).sum()
        gl_pipeline_coverage = (
            gl_total_pipeline / gl_qtd_won_eacv if gl_qtd_won_eacv > 0 else 0
        )

        # Win bucket KPIs
        win_open = wins[wins["stage_number"].between(1, 3)]
        win_total_pipeline = win_open["use_case_eacv"].fillna(0).sum()
        win_weighted_pipeline = wins["expected_value"].fillna(0).sum()
        win_won = wins[wins["stage_number"] >= 4]
        win_qtd_won_eacv = win_won["use_case_eacv"].fillna(0).sum()
        win_pipeline_coverage = (
            win_total_pipeline / win_qtd_won_eacv if win_qtd_won_eacv > 0 else 0
        )

        st.markdown(f"**Selected FQ:** {fc_time_period}  ({fq_start.strftime('%b %d')} – {fq_end.strftime('%b %d, %Y')})")

        st.caption("Go-Live Pipeline (Go-Live Date in FQ)")
        with st.container():
            st.metric(
                "Total Open Pipeline (Stage 1-6)",
                fmt_currency(gl_total_pipeline),
                f"{len(gl_open)} use cases",
            )
            st.metric(
                "Weighted Pipeline",
                fmt_currency(gl_weighted_pipeline),
            )
            st.metric(
                "Total Closed Won (Stage 7)",
                fmt_currency(gl_qtd_won_eacv),
                f"{len(gl_won)} use cases",
            )
            st.metric(
                "Pipeline Coverage",
                f"{gl_pipeline_coverage:.1f}x",
            )

        st.caption("Win Pipeline (Decision Date in FQ)")
        with st.container():
            st.metric(
                "Total Open Pipeline (Stage 1-3)",
                fmt_currency(win_total_pipeline),
                f"{len(win_open)} use cases",
            )
            st.metric(
                "Weighted Pipeline",
                fmt_currency(win_weighted_pipeline),
            )
            st.metric(
                "QTD Won (Stage 4+)",
                fmt_currency(win_qtd_won_eacv),
                f"{len(win_won)} won",
            )
            st.metric(
                "Pipeline Coverage",
                f"{win_pipeline_coverage:.1f}x",
            )

        # ── Section A: Pipeline Overview ──────────────────────────────────────────

        st.divider()
        st.subheader(":material/bar_chart: Section A — Pipeline Overview")

        col_wins_chart, col_golive_chart = st.columns(2)

        with col_wins_chart:
            st.markdown("**Use Case Wins** (Decision Date in FQ)")
            if len(wins) > 0:
                wins_by_stage = (
                    wins.groupby("use_case_stage", as_index=False)["use_case_eacv"]
                    .sum()
                )
                _stage_order = sorted(wins_by_stage["use_case_stage"].tolist())
                chart_wins = (
                    alt.Chart(wins_by_stage)
                    .mark_bar()
                    .encode(
                        x=alt.X("use_case_eacv:Q", title="Total EACV ($)"),
                        y=alt.Y("use_case_stage:N", title="Stage", sort=_stage_order),
                        tooltip=[
                            alt.Tooltip("use_case_stage:N", title="Stage"),
                            alt.Tooltip("use_case_eacv:Q", title="EACV", format="$,.0f"),
                        ],
                        color=alt.Color(
                            "use_case_stage:N",
                            legend=None,
                            scale=alt.Scale(scheme="tableau10"),
                        ),
                    )
                    .properties(height=CHART_HEIGHT)
                )
                st.altair_chart(chart_wins, use_container_width=True)
            else:
                st.info("No use case wins with a Decision Date in this FQ.")

        with col_golive_chart:
            st.markdown("**Go Lives** (Go-Live Date in FQ)")
            if len(go_lives) > 0:
                gl_by_stage = (
                    go_lives.groupby("use_case_stage", as_index=False)["use_case_eacv"]
                    .sum()
                )
                _gl_stage_order = sorted(gl_by_stage["use_case_stage"].tolist())
                chart_gl = (
                    alt.Chart(gl_by_stage)
                    .mark_bar()
                    .encode(
                        x=alt.X("use_case_eacv:Q", title="Total EACV ($)"),
                        y=alt.Y("use_case_stage:N", title="Stage", sort=_gl_stage_order),
                        tooltip=[
                            alt.Tooltip("use_case_stage:N", title="Stage"),
                            alt.Tooltip("use_case_eacv:Q", title="EACV", format="$,.0f"),
                        ],
                        color=alt.Color(
                            "use_case_stage:N",
                            legend=None,
                            scale=alt.Scale(scheme="tableau10"),
                        ),
                    )
                    .properties(height=CHART_HEIGHT)
                )
                st.altair_chart(chart_gl, use_container_width=True)
            else:
                st.info("No use cases with a Go-Live Date in this FQ.")

        # ── Section B: Use Case Wins Details ──────────────────────────────────────

        st.divider()
        st.subheader(":material/emoji_events: Section B — Use Case Wins Details")

        # Join specialist comments (aggregate per use case)
        # When filtering by specialist manager, narrow to their org's team members only
        sp_for_detail = sp_filtered.copy()
        if _mgr_resolved_spec_names is not None:
            sp_for_detail = sp_for_detail[
                sp_for_detail["specialist_name"].isin(_mgr_resolved_spec_names)
                & (sp_for_detail["is_team_member"] == 1)
            ]
        sp_comments = (
            sp_for_detail.groupby("use_case_id", as_index=False)
            .agg(
                specialist_names=("specialist_name", lambda x: ", ".join(x.dropna().unique())),
                specialist_comments=("specialist_comments", lambda x: " | ".join(x.dropna().unique())),
            )
        )

        wins_detail = wins.merge(sp_comments, on="use_case_id", how="left")

        # Inline stage filter for Section B
        _wins_stages = sorted(wins_detail["use_case_stage"].dropna().unique().tolist())
        wins_stage_filter = st.multiselect(
            "Filter by Stage", _wins_stages, default=_wins_stages, key="wins_stage_filter"
        )
        wins_detail = wins_detail[wins_detail["use_case_stage"].isin(wins_stage_filter)]

        wins_detail["eacv_display"] = wins_detail["use_case_eacv"].apply(_fmt_eacv)

        wins_display_cols = [
            "account_name", "use_case_name", "use_case_stage", "decision_date",
            "product_category", "technical_use_case", "prioritized_features",
            "eacv_display", "se_comments", "specialist_comments",
        ]
        if fc_selected_spec_mgrs:
            wins_display_cols.insert(0, "specialist_names")
        # Only keep columns that exist
        wins_display_cols = [c for c in wins_display_cols if c in wins_detail.columns]

        wins_col_config = {
            "specialist_names": st.column_config.TextColumn("Specialist Name"),
            "account_name": st.column_config.TextColumn("Account"),
            "use_case_name": st.column_config.TextColumn("Use Case Name", width="medium"),
            "use_case_stage": st.column_config.TextColumn("Stage"),
            "decision_date": st.column_config.DateColumn("Decision Date"),
            "product_category": st.column_config.TextColumn("Product Categories"),
            "technical_use_case": st.column_config.TextColumn("Technical Use Case"),
            "prioritized_features": st.column_config.TextColumn("Prioritized Features"),
            "eacv_display": st.column_config.TextColumn("EACV"),
            "se_comments": st.column_config.TextColumn("SE Comments", width="large"),
            "specialist_comments": st.column_config.TextColumn("Specialist Comments", width="large"),
        }

        with st.container():
            st.caption(
                f"Showing **{len(wins_detail)}** use cases with Decision Date in **{fc_time_period}**"
            )
            st.dataframe(
                wins_detail.sort_values("use_case_eacv", ascending=False)[wins_display_cols],
                column_config=wins_col_config,
                hide_index=True,
                use_container_width=True,
                height=500,
            )

        # ── Section C: Use Case Go Lives Details ──────────────────────────────────

        st.divider()
        st.subheader(":material/rocket_launch: Section C — Use Case Go Lives Details")

        gl_detail = go_lives.merge(sp_comments, on="use_case_id", how="left")

        # Inline stage filter for Section C
        _gl_stages = sorted(gl_detail["use_case_stage"].dropna().unique().tolist())
        gl_stage_filter = st.multiselect(
            "Filter by Stage", _gl_stages, default=_gl_stages, key="gl_stage_filter"
        )
        gl_detail = gl_detail[gl_detail["use_case_stage"].isin(gl_stage_filter)]

        gl_detail["eacv_display"] = gl_detail["use_case_eacv"].apply(_fmt_eacv)
        gl_detail["gl_prob_display"] = gl_detail["go_live_probability"].apply(
            lambda v: f"{v * 100:.1f}%" if pd.notna(v) else ""
        )

        gl_display_cols = [
            "account_name", "use_case_name", "use_case_stage", "go_live_date",
            "product_category", "technical_use_case", "prioritized_features",
            "eacv_display", "health_status", "gl_prob_display",
            "se_comments", "specialist_comments",
        ]
        if fc_selected_spec_mgrs:
            gl_display_cols.insert(0, "specialist_names")
        gl_display_cols = [c for c in gl_display_cols if c in gl_detail.columns]

        gl_col_config = {
            "specialist_names": st.column_config.TextColumn("Specialist Name"),
            "account_name": st.column_config.TextColumn("Account"),
            "use_case_name": st.column_config.TextColumn("Use Case Name", width="medium"),
            "use_case_stage": st.column_config.TextColumn("Stage"),
            "go_live_date": st.column_config.DateColumn("Go-Live Date"),
            "product_category": st.column_config.TextColumn("Product Categories"),
            "technical_use_case": st.column_config.TextColumn("Technical Use Case"),
            "prioritized_features": st.column_config.TextColumn("Prioritized Features"),
            "eacv_display": st.column_config.TextColumn("EACV"),
            "health_status": st.column_config.TextColumn("Health"),
            "gl_prob_display": st.column_config.TextColumn("Go-Live Probability"),
            "se_comments": st.column_config.TextColumn("SE Comments", width="large"),
            "specialist_comments": st.column_config.TextColumn("Specialist Comments", width="large"),
        }

        with st.container():
            st.caption(
                f"Showing **{len(gl_detail)}** use cases with Go-Live in **{fc_time_period}**"
            )
            st.dataframe(
                gl_detail.sort_values("use_case_eacv", ascending=False)[gl_display_cols],
                column_config=gl_col_config,
                hide_index=True,
                use_container_width=True,
                height=500,
            )

    elif page == "Use Case Hygiene":

        # ── Page header ───────────────────────────────────────────────────────────

        _hdr_col, _btn_col = st.columns([8, 1])
        _hdr_col.markdown("# :material/health_and_safety: Use Case Hygiene")
        if _btn_col.button(":material/restart_alt: Reset", type="secondary", key="hy_reset"):
            st.session_state.clear()
            st.rerun()

        # ── Apply Hygiene filters ─────────────────────────────────────────────────

        hy_df = specialist_engagement.copy()

        if hy_active_only and "active_status" in hy_df.columns:
            hy_df = hy_df[hy_df["active_status"] == True]
        if hy_selected_groups:
            hy_df = hy_df[hy_df["specialist_group"].isin(hy_selected_groups)]
        if hy_selected_theaters:
            hy_df = hy_df[hy_df["hierarchy_3"].isin(hy_selected_theaters)]
        if hy_selected_statuses:
            hy_df = hy_df[hy_df["update_needed_status"].isin(hy_selected_statuses)]
        if hy_selected_managers:
            hy_df = hy_df[hy_df["manager_name"].isin(hy_selected_managers)]

        # Compute active use case count per specialist
        if "active_use_case_list" in hy_df.columns:
            hy_df["active_uc_count"] = hy_df["active_use_case_list"].apply(
                lambda x: len([v.strip() for v in str(x).split(",") if v.strip() and v.strip().lower() != "nan"]) if pd.notna(x) else 0
            )
        else:
            hy_df["active_uc_count"] = 0

        # ── KPI row ──────────────────────────────────────────────────────────────

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Specialists", len(hy_df))
        with k2:
            _active_count = len(hy_df[hy_df["active_status"] == True]) if "active_status" in hy_df.columns else len(hy_df)
            st.metric("Active Specialists", _active_count)
        with k3:
            _needing = len(hy_df[hy_df["update_needed_status"].isin(["Needed Soon", "Needed Now"])])
            st.metric("Needing Update", _needing)
        with k4:
            _overdue = len(hy_df[hy_df["update_needed_status"] == "Needed Now"])
            st.metric("Overdue", _overdue)

        st.divider()

        # ── Specialist engagement table ──────────────────────────────────────────

        st.subheader(":material/group: Specialist Engagement Overview")

        display_cols = [
            "preferred_name", "specialist_group", "hierarchy_3", "manager_name",
            "active_status", "active_reason",
            "comments_14d", "comments_7d", "activities_14d", "activities_7d",
            "last_update", "days_until_update_needed", "update_needed_status",
            "active_uc_count",
        ]
        display_cols = [c for c in display_cols if c in hy_df.columns]

        col_config = {
            "preferred_name": st.column_config.TextColumn("Specialist"),
            "specialist_group": st.column_config.TextColumn("Group"),
            "hierarchy_3": st.column_config.TextColumn("Theater"),
            "manager_name": st.column_config.TextColumn("Manager"),
            "active_status": st.column_config.CheckboxColumn("Active"),
            "active_reason": st.column_config.TextColumn("Active Reason"),
            "comments_14d": st.column_config.NumberColumn("Comments 14d"),
            "comments_7d": st.column_config.NumberColumn("Comments 7d"),
            "activities_14d": st.column_config.NumberColumn("Activities 14d"),
            "activities_7d": st.column_config.NumberColumn("Activities 7d"),
            "last_update": st.column_config.TextColumn("Last Update"),
            "days_until_update_needed": st.column_config.NumberColumn("Days Until Update"),
            "update_needed_status": st.column_config.TextColumn("Update Status"),
            "active_uc_count": st.column_config.NumberColumn("Active UCs"),
        }

        sort_col = "days_until_update_needed" if "days_until_update_needed" in hy_df.columns else display_cols[0]
        hy_sorted = hy_df.sort_values(sort_col, ascending=True, na_position="last")

        with st.container():
            st.caption(f"Showing **{len(hy_sorted)}** specialists")
            st.dataframe(
                hy_sorted[display_cols],
                column_config=col_config,
                hide_index=True,
                use_container_width=True,
                height=500,
            )

        st.divider()

        # ── Use case drill-down ──────────────────────────────────────────────────

        st.subheader(":material/assignment: Use Case Drill-Down")

        specialist_names = sorted(hy_df["preferred_name"].dropna().unique().tolist())
        if specialist_names:
            selected_specialist = st.selectbox(
                "Select Specialist", specialist_names, key="hy_specialist_select"
            )

            spec_row = hy_df[hy_df["preferred_name"] == selected_specialist].iloc[0]
            uc_list_raw = spec_row.get("active_use_case_list", "")

            if pd.notna(uc_list_raw) and str(uc_list_raw).strip():
                uc_ids = [v.strip() for v in str(uc_list_raw).split(",") if v.strip() and v.strip().lower() != "nan"]

                if uc_ids:
                    uc_matches = use_case_detail[use_case_detail["use_case_id"].isin(uc_ids)].copy()

                    if not uc_matches.empty:
                        uc_matches["sf_link"] = uc_matches["use_case_id"].apply(
                            lambda uid: f"https://snowflake.lightning.force.com/lightning/r/vh__Deliverable__c/{uid}/view"
                        )

                        uc_display_cols = [
                            "use_case_name", "account_name", "use_case_eacv",
                            "use_case_stage", "use_case_status", "workloads", "sf_link",
                        ]
                        uc_display_cols = [c for c in uc_display_cols if c in uc_matches.columns]

                        uc_col_config = {
                            "use_case_name": st.column_config.TextColumn("Use Case"),
                            "account_name": st.column_config.TextColumn("Account"),
                            "use_case_eacv": st.column_config.NumberColumn("EACV", format="$%,.0f"),
                            "use_case_stage": st.column_config.TextColumn("Stage"),
                            "use_case_status": st.column_config.TextColumn("Status"),
                            "workloads": st.column_config.TextColumn("Workloads"),
                            "sf_link": st.column_config.LinkColumn("Salesforce", display_text="Open"),
                        }

                        with st.container():
                            st.caption(
                                f"**{selected_specialist}** — {len(uc_matches)} active use case(s)"
                            )
                            st.dataframe(
                                uc_matches.sort_values("use_case_eacv", ascending=False)[uc_display_cols],
                                column_config=uc_col_config,
                                hide_index=True,
                                use_container_width=True,
                            )
                    else:
                        st.info(f"No matching use case records found for {selected_specialist}.")
                else:
                    st.info(f"No active use cases listed for {selected_specialist}.")
            else:
                st.info(f"No active use cases listed for {selected_specialist}.")
        else:
            st.info("No specialists match the current filters.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: Flags — Field Flags Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif module == "Flags":

    _hdr_col, _btn_col = st.columns([8, 1])
    _hdr_col.markdown("# :material/flag: Field Flags")
    if _btn_col.button(":material/restart_alt: Reset", type="secondary", key="fl_reset"):
        st.session_state.clear()
        st.rerun()

    st.caption("Data quality actions the field needs to take. Goal: all counts = 0.")

    # ── Build filtered specialist engagement data ─────────────────────────────
    fl_eng = specialist_engagement.copy()
    if fl_selected_theaters:
        fl_eng = fl_eng[fl_eng["hierarchy_3"].isin(fl_selected_theaters)]
    if fl_selected_managers:
        fl_eng = fl_eng[fl_eng["manager_name"].isin(fl_selected_managers)]
    if fl_selected_groups:
        fl_eng = fl_eng[fl_eng["specialist_group"].isin(fl_selected_groups)]

    if "active_use_case_list" in fl_eng.columns:
        fl_eng["active_uc_count"] = fl_eng["active_use_case_list"].apply(
            lambda x: len([v.strip() for v in str(x).split(",")
                           if v.strip() and v.strip().lower() != "nan"])
            if pd.notna(x) else 0
        )
    else:
        fl_eng["active_uc_count"] = 0

    # ── Build use case pipeline data ─────────────────────────────────────────
    fl_fp = forecast_pipeline.copy()
    fl_fp_active = fl_fp[fl_fp["stage_number"].between(1, 6)].copy()
    _eacv_p75 = fl_fp_active["use_case_eacv"].quantile(EACV_PCT_HIGH) if len(fl_fp_active) > 0 else 0

    # ── Build account engagement data ─────────────────────────────────────────
    fl_bob = bob.merge(
        base_metrics[["account_id", "last_account_meeting_date",
                      "active_use_cases", "high_risk_use_cases", "in_pursuit_use_cases"]],
        on="account_id", how="left",
    )
    fl_bob = fl_bob.merge(
        indicators[["account_id", "consumption_risk", "account_risk"]],
        on="account_id", how="left", suffixes=("", "_ind"),
    )
    _today_fl = pd.Timestamp.now().normalize()
    fl_bob["days_since_meeting"] = (
        _today_fl - pd.to_datetime(fl_bob["last_account_meeting_date"])
    ).dt.days.fillna(999).astype(int)

    # ── Helper: add Salesforce link column ───────────────────────────────────
    def _add_sf_link(df):
        df = df.copy()
        df["sf_link"] = df["use_case_id"].apply(
            lambda uid: f"https://snowflake.lightning.force.com/lightning/r/vh__Deliverable__c/{uid}/view"
        )
        return df

    _uc_col_cfg = {
        "account_name":             st.column_config.TextColumn("Account"),
        "use_case_name":            st.column_config.TextColumn("Use Case", width="medium"),
        "use_case_stage":           st.column_config.TextColumn("Stage"),
        "use_case_eacv":            st.column_config.NumberColumn("EACV", format="$%,.0f"),
        "decision_date":            st.column_config.DateColumn("Decision Date"),
        "go_live_date":             st.column_config.DateColumn("Go-Live Date"),
        "health_status":            st.column_config.TextColumn("Health"),
        "go_live_probability":      st.column_config.NumberColumn("GL Prob %", format="%.1f"),
        "is_tech_won":              st.column_config.CheckboxColumn("Tech Won"),
        "has_specialist_coverage":  st.column_config.CheckboxColumn("Has Specialist"),
        "days_since_comment_update":st.column_config.NumberColumn("Days Since Comment"),
        "comment_stale_flag":       st.column_config.CheckboxColumn("Stale"),
        "theater_name":             st.column_config.TextColumn("Theater"),
        "rep_name":                 st.column_config.TextColumn("AE"),
        "sf_link":                  st.column_config.LinkColumn("Salesforce", display_text="Open"),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP A — SPECIALIST DATA QUALITY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(":material/person_alert: Group A — Specialist Data Quality")

    a_overdue   = fl_eng[fl_eng["update_needed_status"] == "Needed Now"]
    a_due_soon  = fl_eng[fl_eng["update_needed_status"] == "Needed Soon"]
    a_zero_act  = fl_eng[(fl_eng["activities_14d"].fillna(0) == 0) & (fl_eng["active_uc_count"] > 0)]
    a_zero_cmt  = fl_eng[(fl_eng["comments_14d"].fillna(0) == 0) & (fl_eng["active_uc_count"] > 0)]
    a_no_ucs    = fl_eng[fl_eng["active_uc_count"] == 0]

    ga1, ga2, ga3, ga4, ga5 = st.columns(5)
    with ga1:
        st.metric("Overdue (Needed Now)", len(a_overdue),
                  help="Specialists whose SFDC update is past due — update today")
    with ga2:
        st.metric("Due Soon (Needed Soon)", len(a_due_soon),
                  help="Update window closing — must update before next forecast call")
    with ga3:
        st.metric("Zero Activity 14d", len(a_zero_act),
                  help="Active specialists with zero SFDC activities in 14 days")
    with ga4:
        st.metric("Zero Comments 14d", len(a_zero_cmt),
                  help="Active specialists with zero SFDC comments in 14 days")
    with ga5:
        st.metric("No Active UCs", len(a_no_ucs),
                  help="Specialists with no use cases assigned — pipeline gap")

    _a_cols = ["preferred_name", "specialist_group", "hierarchy_3", "manager_name",
               "comments_7d", "comments_14d", "activities_14d", "update_needed_status", "active_uc_count"]
    _a_cfg = {
        "preferred_name":       st.column_config.TextColumn("Specialist"),
        "specialist_group":     st.column_config.TextColumn("Group"),
        "hierarchy_3":          st.column_config.TextColumn("Theater"),
        "manager_name":         st.column_config.TextColumn("Manager"),
        "comments_7d":          st.column_config.NumberColumn("Comments 7d"),
        "comments_14d":         st.column_config.NumberColumn("Comments 14d"),
        "activities_14d":       st.column_config.NumberColumn("Activities 14d"),
        "update_needed_status": st.column_config.TextColumn("Status"),
        "active_uc_count":      st.column_config.NumberColumn("Active UCs"),
    }

    with st.expander(f"**Overdue ({len(a_overdue)})** — action: update SFDC use case comments today",
                     expanded=len(a_overdue) > 0):
        if len(a_overdue) > 0:
            _cols = [c for c in _a_cols if c in a_overdue.columns]
            st.dataframe(a_overdue[_cols].sort_values("comments_14d"),
                         column_config=_a_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No overdue specialists.")

    with st.expander(f"**Due Soon ({len(a_due_soon)})** — action: update before next forecast call",
                     expanded=False):
        if len(a_due_soon) > 0:
            _cols = [c for c in _a_cols if c in a_due_soon.columns]
            st.dataframe(a_due_soon[_cols].sort_values("comments_14d"),
                         column_config=_a_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No specialists due soon.")

    with st.expander(f"**Zero Activity or Comments 14d ({max(len(a_zero_act), len(a_zero_cmt))})** "
                     f"— action: log an activity or comment in SFDC", expanded=False):
        _combined_a = pd.concat([a_zero_act, a_zero_cmt]).drop_duplicates(subset=["preferred_name"])
        if len(_combined_a) > 0:
            _zac_cols = ["preferred_name", "specialist_group", "hierarchy_3", "manager_name",
                         "comments_7d", "comments_14d", "activities_14d", "activities_7d", "active_uc_count"]
            _zac_cols = [c for c in _zac_cols if c in _combined_a.columns]
            _zac_cfg = dict(_a_cfg)
            _zac_cfg["activities_7d"] = st.column_config.NumberColumn("Activities 7d")
            st.dataframe(_combined_a[_zac_cols].sort_values("activities_14d"),
                         column_config=_zac_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All active specialists have recent activity and comments.")

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP B — USE CASE RECORD QUALITY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(":material/assignment_late: Group B — Use Case Record Quality")

    b_no_specialist = fl_fp_active[fl_fp_active["has_specialist_coverage"] == False]
    b_no_dec_date   = fl_fp_active[
        (fl_fp_active["stage_number"] >= 3) & fl_fp_active["decision_date"].isna()
    ]
    b_no_gl_date    = fl_fp_active[
        (fl_fp_active["stage_number"] >= 4) & fl_fp_active["go_live_date"].isna()
    ]
    b_no_tw_hi      = fl_fp_active[
        (fl_fp_active["stage_number"] >= 3) &
        (fl_fp_active["is_tech_won"] == False) &
        (fl_fp_active["use_case_eacv"].fillna(0) >= _eacv_p75)
    ]
    b_stale         = fl_fp_active[fl_fp_active["comment_stale_flag"] == 1]
    b_at_risk       = fl_fp[fl_fp["health_status"] == "At Risk"]
    b_low_prob      = fl_fp[
        (fl_fp["is_current_fq_go_live"] == True) &
        (fl_fp["stage_number"].between(1, 6)) &
        (fl_fp["go_live_probability"].fillna(1.0) < 0.5)
    ]

    gb1, gb2, gb3, gb4, gb5, gb6, gb7 = st.columns(7)
    with gb1:
        st.metric("No Specialist", len(b_no_specialist),
                  help="Active use cases with no specialist assigned")
    with gb2:
        st.metric("No Decision Date (S3+)", len(b_no_dec_date),
                  help="Stage 3+ use cases missing a decision date in SFDC")
    with gb3:
        st.metric("No Go-Live Date (S4+)", len(b_no_gl_date),
                  help="Stage 4+ use cases missing a go-live date in SFDC")
    with gb4:
        st.metric("No Tech Win – Hi EACV", len(b_no_tw_hi),
                  help=f"Stage 3+ without technical win, EACV ≥ 75th pct (≥{fmt_currency(_eacv_p75)})")
    with gb5:
        st.metric(f"Stale Comments (>{DAYS_STALE_COMMENT}d)", len(b_stale),
                  help="Use cases with no comment update in 30+ days")
    with gb6:
        st.metric("At-Risk Go-Lives", len(b_at_risk),
                  help="Use cases flagged health_status = At Risk")
    with gb7:
        st.metric("Low Go-Live Prob (<50%)", len(b_low_prob),
                  help="Current FQ go-lives with go-live probability below 50%")

    _b_base = ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
               "decision_date", "go_live_date", "health_status", "theater_name", "rep_name", "sf_link"]

    with st.expander(f"**No Specialist ({len(b_no_specialist)})** — action: assign a specialist in SFDC",
                     expanded=len(b_no_specialist) > 0):
        if len(b_no_specialist) > 0:
            _df = _add_sf_link(b_no_specialist)
            _cols = [c for c in _b_base if c in _df.columns]
            st.dataframe(_df.sort_values("use_case_eacv", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All active use cases have specialist coverage.")

    with st.expander(f"**Missing Decision Date ({len(b_no_dec_date)})** "
                     f"— action: confirm and log decision date in SFDC",
                     expanded=len(b_no_dec_date) > 0):
        if len(b_no_dec_date) > 0:
            _df = _add_sf_link(b_no_dec_date)
            _cols = [c for c in _b_base if c in _df.columns]
            st.dataframe(_df.sort_values("use_case_eacv", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All Stage 3+ use cases have a decision date.")

    with st.expander(f"**Missing Go-Live Date ({len(b_no_gl_date)})** "
                     f"— action: log planned go-live date in SFDC",
                     expanded=len(b_no_gl_date) > 0):
        if len(b_no_gl_date) > 0:
            _df = _add_sf_link(b_no_gl_date)
            _cols = [c for c in _b_base if c in _df.columns]
            st.dataframe(_df.sort_values("use_case_eacv", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All Stage 4+ use cases have a go-live date.")

    with st.expander(f"**No Tech Win – High EACV ({len(b_no_tw_hi)})** "
                     f"— action: complete technical win criteria in SFDC",
                     expanded=len(b_no_tw_hi) > 0):
        if len(b_no_tw_hi) > 0:
            _df = _add_sf_link(b_no_tw_hi)
            _cols = [c for c in ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
                                  "is_tech_won", "decision_date", "theater_name", "rep_name", "sf_link"]
                     if c in _df.columns]
            st.dataframe(_df.sort_values("use_case_eacv", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All high-EACV Stage 3+ use cases have a technical win.")

    with st.expander(f"**Stale Comments ({len(b_stale)})** "
                     f"— action: update SFDC use case comment",
                     expanded=len(b_stale) > 0):
        if len(b_stale) > 0:
            _df = _add_sf_link(b_stale)
            _cols = [c for c in ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
                                  "days_since_comment_update", "theater_name", "rep_name", "sf_link"]
                     if c in _df.columns]
            st.dataframe(_df.sort_values("days_since_comment_update", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("All active use cases have recent comments.")

    with st.expander(f"**At-Risk Go-Lives ({len(b_at_risk)})** "
                     f"— action: document mitigation plan in SE Comments",
                     expanded=len(b_at_risk) > 0):
        if len(b_at_risk) > 0:
            _df = _add_sf_link(b_at_risk)
            _cols = [c for c in ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
                                  "health_status", "go_live_date", "theater_name", "rep_name", "sf_link"]
                     if c in _df.columns]
            st.dataframe(_df.sort_values("use_case_eacv", ascending=False)[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No at-risk go-lives.")

    with st.expander(f"**Low Go-Live Probability ({len(b_low_prob)})** "
                     f"— action: update health assessment or escalate",
                     expanded=len(b_low_prob) > 0):
        if len(b_low_prob) > 0:
            _df = _add_sf_link(b_low_prob)
            _df["go_live_probability"] = (_df["go_live_probability"] * 100).round(1)
            _cols = [c for c in ["account_name", "use_case_name", "use_case_stage", "use_case_eacv",
                                  "go_live_probability", "go_live_date", "health_status",
                                  "theater_name", "sf_link"]
                     if c in _df.columns]
            st.dataframe(_df.sort_values("go_live_probability")[_cols],
                         column_config=_uc_col_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No current-FQ go-lives with low probability.")

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP C — ACCOUNT ENGAGEMENT GAPS
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(":material/warning: Group C — Account Engagement Gaps")

    c_urgent      = fl_bob[
        (fl_bob["revenue_delta_30"].fillna(0) < 0) &
        (fl_bob["days_since_meeting"] >= DAYS_NO_CONTACT)
    ]
    c_high_risk   = fl_bob[fl_bob["high_risk_use_cases"].fillna(0) > 0]
    c_dec_pursuit = fl_bob[
        (fl_bob["revenue_delta_30"].fillna(0) < 0) &
        (fl_bob["in_pursuit_use_cases"].fillna(0) > 0)
    ]

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.metric("Urgent Decliners (No Contact 30d)", len(c_urgent),
                  help="Declining accounts with no meeting in 30+ days — same-day AE outreach required")
    with gc2:
        st.metric("High-Risk Use Cases", len(c_high_risk),
                  help="Accounts with ≥1 use case flagged High Risk in SFDC — specialist review needed")
    with gc3:
        st.metric("Declining + Open Pursuit UCs", len(c_dec_pursuit),
                  help="Declining accounts with Pursuit-stage use cases — pipeline validity at risk")

    _c_base = ["account_name", "account_owner_name", "lead_sales_engineer_name",
               "theater", "region", "revenue_delta_30", "days_since_meeting",
               "last_account_meeting_date", "active_use_cases", "in_pursuit_use_cases",
               "high_risk_use_cases"]
    _c_cfg = {
        "account_name":              st.column_config.TextColumn("Account"),
        "account_owner_name":        st.column_config.TextColumn("AE"),
        "lead_sales_engineer_name":  st.column_config.TextColumn("Lead SE"),
        "theater":                   st.column_config.TextColumn("Theater"),
        "region":                    st.column_config.TextColumn("Region"),
        "revenue_delta_30":          st.column_config.NumberColumn("Rev Delta 30d", format="$%,.0f"),
        "days_since_meeting":        st.column_config.NumberColumn("Days Since Meeting"),
        "last_account_meeting_date": st.column_config.DateColumn("Last Meeting"),
        "active_use_cases":          st.column_config.NumberColumn("Active UCs"),
        "in_pursuit_use_cases":      st.column_config.NumberColumn("Pursuit UCs"),
        "high_risk_use_cases":       st.column_config.NumberColumn("High Risk UCs"),
    }

    with st.expander(f"**Urgent Decliners ({len(c_urgent)})** "
                     f"— action: AE outreach same day; log meeting in CRM",
                     expanded=len(c_urgent) > 0):
        if len(c_urgent) > 0:
            _cols = [c for c in _c_base if c in c_urgent.columns]
            st.dataframe(c_urgent[_cols].sort_values("revenue_delta_30"),
                         column_config=_c_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No urgent decliners with engagement gaps.")

    with st.expander(f"**High-Risk Use Cases ({len(c_high_risk)})** "
                     f"— action: specialist reviews and updates health in SFDC",
                     expanded=len(c_high_risk) > 0):
        if len(c_high_risk) > 0:
            _cols = [c for c in _c_base if c in c_high_risk.columns]
            st.dataframe(c_high_risk[_cols].sort_values("high_risk_use_cases", ascending=False),
                         column_config=_c_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No accounts with high-risk use cases.")

    with st.expander(f"**Declining + Open Pursuit UCs ({len(c_dec_pursuit)})** "
                     f"— action: AE + Specialist sync; validate whether pipeline is real",
                     expanded=False):
        if len(c_dec_pursuit) > 0:
            _cols = [c for c in _c_base if c in c_dec_pursuit.columns]
            st.dataframe(c_dec_pursuit[_cols].sort_values("revenue_delta_30"),
                         column_config=_c_cfg, hide_index=True, use_container_width=True)
        else:
            st.success("No declining accounts with open pursuit use cases.")


# ── Guide module ──────────────────────────────────────────────────────────────

elif module == "Guide":
    st.markdown("# :material/menu_book: Operational Manual")
    st.caption("How to use the Overlay Dashboard — reference for field teams and managers")

    (
        tab_overview,
        tab_home,
        tab_consumption,
        tab_usecase,
        tab_flags,
        tab_ref,
    ) = st.tabs([
        "Overview",
        "Home — Command Center",
        "Consumption",
        "Use Case",
        "Field Flags",
        "Thresholds & Data",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_overview:
        st.markdown("""
## What Is This App?

The **Overlay Dashboard** is the single source of truth for Overlay Specialists,
Managers, and Area Field Executives (AFEs) to monitor the health of their book of
business, track use case pipeline, and ensure field data quality in Salesforce.

### Who Uses It
| Role | Primary Use |
|------|-------------|
| Specialist | Daily hygiene — update comments, review flag counts |
| Manager | Weekly forecast prep — coverage ratios, at-risk go-lives |
| AFE / VP | QBR readiness — QoQ trends, pipeline sufficiency, urgent decliners |

### How It's Organised
| Module | Purpose |
|--------|---------|
| **Home** | 12-KPI command center — snapshot of the entire business |
| **Consumption** | Revenue trends, Lights On / Lights Off, account-level intelligence |
| **Use Case** | Pipeline forecasting, tech win tracking, hygiene scoring |
| **Flags** | Data-quality action list — goal is zero on every counter |
| **Guide** | This manual |

### Data Refresh
All data is cached for **30 minutes**. Each page shows a cache timestamp in the
sidebar caption. Force a refresh by reloading the browser tab.
""")
        st.info(
            "**Warehouse:** The app runs on `SNOWADHOC`. "
            "If queries time out, check that the warehouse is not suspended."
        )

    # ── Tab 2: Home — Command Center ─────────────────────────────────────────
    with tab_home:
        st.markdown("""
## Business Health Command Center

The Home page is unfiltered — it covers the **full book of business** so
leadership can see total health at a glance.

### Row 1 — Consumption Health

| KPI | Definition | Act When |
|-----|-----------|---------|
| **QTD Consumption** | Sum of `current_fiscal_quarter_revenue` across all accounts | Below plan → escalate to Consumption Intelligence |
| **Predicted FQ Finish** | Model prediction for end-of-quarter revenue | < QTD → forecast risk; engage at-risk accounts |
| **QoQ vs Prior FQ** | % change vs previous fiscal quarter | Negative → review Lights Off accounts |
| **Lights Off Accounts** | Accounts with zero or near-zero consumption in the current period | > 0 → open Lights On / Lights Off immediately |
| **Urgent Decliners** | Accounts with 30-day revenue decline **and** no meeting in 30+ days | Any → schedule outreach this week |
| **Revenue at Risk (30d)** | Sum of negative 30-day revenue deltas across all accounts | Rising → assign specialist coverage |

### Row 2 — Pipeline Health

| KPI | Definition | Act When |
|-----|-----------|---------|
| **Go-Live Coverage** | Open pipeline EACV ÷ QTD Won target | < 1.5x (yellow) or < 2.0x (red) → build pipeline |
| **Win Coverage** | Open + won pipeline ÷ target | < 1.5x → urgently qualify new use cases |
| **QTD Won** | Sum of EACV for tech-won use cases this quarter | Tracking behind plan → focus NTB accounts |
| **At-Risk Go-Lives** | Use cases with `health_status = At Risk` and current-FQ go-live | Any → review with AE and Specialist |
| **Specialists Overdue** | Specialists with `update_needed_status = Needed Now` | Any → direct manager action today |
| **Missing Comments ($)** | EACV of active use cases with stale or absent SFDC comments | High → Group B Flags → No Specialist / Stale Comments |

### Navigation Buttons
Each KPI card has a **→ button** that jumps directly to the relevant detail page.
The "View All Field Flags" button at the bottom opens the Flags module.
""")

    # ── Tab 3: Consumption ────────────────────────────────────────────────────
    with tab_consumption:
        st.markdown("""
## Consumption Module

### Page 1 — Lights On / Lights Off

**Purpose:** Identify the biggest movers in consumption — accounts accelerating
(Lights On) and accounts declining (Lights Off).

**Filters:** Region · Theater · Product Category · Feature · Functional Area ·
Industry · Account / Owner Search

**Period selector:** Compare the current window (7, 14, 30, or 90 days) against
the prior equivalent window.

| Section | What to Look For |
|---------|-----------------|
| **Lights Off table** | Largest negative revenue deltas — sort by delta to find the worst accounts |
| **Lights On table** | Largest positive deltas — candidates for expansion motions |
| **Delta chart** | Trend of net daily delta — a worsening slope needs immediate engagement |

**Action:** For any Lights Off account, click the account name to pull up its
risk context, then assign or confirm Specialist coverage.

---

### Page 2 — Consumption Intelligence

**Purpose:** Deep-dive consumption health, trend analysis, and predictive signals
across the filtered book of business.

| Section | What to Look For |
|---------|-----------------|
| **QoQ trend chart** | Quarter-over-quarter revenue progression — declining slope needs executive attention |
| **Prediction vs Actual** | Gap between `revenue_prediction_current_fiscal_quarter` and QTD actuals |
| **Account heat map** | Concentration of risk by region / theater |
| **Risk movements** | Accounts that moved risk tiers — new `HIGH` or `CRITICAL` entries need same-week action |

**Action:** Filter to your theater before reviewing. Export the heat map view
for your QBR or weekly manager call.
""")

    # ── Tab 4: Use Case ───────────────────────────────────────────────────────
    with tab_usecase:
        st.markdown("""
## Use Case Module

### Page 1 — Use Case Forecasting

**Purpose:** Track pipeline coverage and go-live commitments for the current and
future fiscal quarters.

**Filters:** Fiscal Quarter · Theater · Region · Account Owner · Specialist ·
Product Category · Specialist Manager

| Metric | Definition |
|--------|-----------|
| **Go-Live Coverage** | Open pipeline ÷ won target. Healthy ≥ 2.0x, warning < 1.5x |
| **Win Coverage** | (Open + won) ÷ target |
| **Tech Won** | Use cases with `is_tech_won = True` — counts toward closed revenue |
| **Pipeline table** | Full list of open use cases with stage, EACV, go-live date, health status |

**How to Use:**
1. Set the **Fiscal Quarter** filter to the current FQ.
2. Check Go-Live and Win coverage ratios. Below 2.0x → actively qualify new use cases.
3. Sort the pipeline table by **Go-Live Date** ascending — anything without a
   Specialist assigned in the next 30 days is an immediate risk.
4. Use the **Specialist Name** filter to review individual specialist pipelines
   on 1:1 calls.

---

### Page 2 — Use Case Hygiene

**Purpose:** Surface use cases that are missing critical data fields or have
stale Specialist engagement.

| Flag | Meaning | Action |
|------|---------|--------|
| **No Specialist** | `has_specialist_coverage = False` | Assign coverage in SFDC today |
| **Stale Comment** | `comment_stale_flag = True` (> 30 days) | Specialist updates comment this week |
| **At-Risk** | `health_status = At Risk` | AE + Specialist joint call; update MEDIC |
| **Missing Decision Date** | `decision_date` is null | AE fills in SFDC; required for forecast |
| **Missing Go-Live Date** | `go_live_date` is null | Required for FQ planning; block this use case from pipeline reporting |

**Goal:** Every in-flight use case should have a Specialist, a Decision Date,
a Go-Live Date, and a comment updated within 30 days.
""")

    # ── Tab 5: Field Flags ────────────────────────────────────────────────────
    with tab_flags:
        st.markdown("""
## Field Flags Dashboard

**Goal: every counter = 0.** Each flag group surfaces a specific category of
data-quality debt. Managers review this page at the start of every forecast week.

---

### Group A — Specialist Data Quality

Filters: Theater · Manager · Specialist Group (sidebar)

| Flag | Definition | Action |
|------|-----------|--------|
| **Overdue (Needed Now)** | `update_needed_status = Needed Now` | Manager escalation — update SFDC comments **today** |
| **Due Soon (Needed Soon)** | `update_needed_status = Needed Soon` | Update before the next forecast call |
| **Zero Activity 14d** | No SFDC activities logged in 14 days, has active UCs | Specialist re-engages or manager re-assigns |
| **Zero Comments 14d** | No SFDC comments in 14 days, has active UCs | Add a comment for every active use case this week |
| **No Active UCs** | Specialist has no use cases assigned | Manager reviews capacity and assigns pipeline |

**Cadence:** Review Group A every Monday morning. Target = 0 Overdue before
each Thursday forecast call.

---

### Group B — Use Case Record Quality

| Flag | Definition | Action |
|------|-----------|--------|
| **No Specialist** | `has_specialist_coverage = False` | Assign in SFDC — no UC should be uncovered |
| **Missing Decision Date** | `decision_date` is null | AE enters date; required for forecast accuracy |
| **Missing Go-Live Date** | `go_live_date` is null | AE enters date; blocks FQ planning |
| **No Tech Win — High EACV** | `is_tech_won = False` and EACV ≥ 75th percentile | High-value deal not yet technically qualified — prioritise |
| **Stale Comments** | `comment_stale_flag = True` | Specialist adds comment this week |
| **At-Risk Go-Lives** | `health_status = At Risk`, current FQ go-live | Joint AE + Specialist action plan this week |
| **Low Go-Live Probability** | `go_live_probability < 50%` | Review blocking issues; escalate or slip the date |

---

### Group C — Account Engagement Gaps

| Flag | Definition | Action |
|------|-----------|--------|
| **Urgent Decliners** | 30-day revenue decline **and** last meeting > 30 days ago | Schedule outreach this week — do not let these age |
| **High-Risk UCs** | Accounts with `high_risk_use_cases > 0` | Consumption + Use Case joint review |
| **Declining + Open Pursuit** | Declining revenue and active pursuit use case | Validate whether the pipeline is real; flag for manager |

---
""")
        st.info(
            "All expanders in the Flags page include **direct Salesforce links** "
            "— click 'Open' in any table row to jump straight to the record."
        )
        st.warning(
            "**Reset button:** The Reset button in the top-right of the Flags page "
            "clears all session state and reloads the app. Use it if filters appear stuck."
        )

    # ── Tab 6: Thresholds & Data ──────────────────────────────────────────────
    with tab_ref:
        st.markdown("""
## Thresholds & Constants

These values drive all KPI colours, flag logic, and action directives across
every module. Changes require a code update and re-deploy.

| Constant | Value | Used In |
|----------|-------|---------|
| `COVERAGE_WARNING` | **1.5x** | Go-Live and Win Coverage KPIs — turns yellow below this |
| `COVERAGE_HEALTHY` | **2.0x** | Coverage KPIs — green at or above this |
| `DAYS_NO_CONTACT` | **30 days** | Urgent Decliner flag — meeting gap threshold |
| `DAYS_STALE_COMMENT` | **30 days** | Stale comment flag on use cases |
| `EACV_PCT_HIGH` | **75th percentile** | "High EACV" threshold for No Tech Win flag |

---

## Data Sources

| Table | Module | Contents |
|-------|--------|---------|
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_BOB` | Consumption, Home | Account-level trailing revenue, deltas, predictions, growth rates |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_ACCOUNT_BASE_METRICS` | Home, Flags | Meeting dates, active/risk/pursuit use case counts |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_ACCOUNT_INDICATORS` | Flags | Consumption risk tier, account risk, assessment tier |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_FORECAST_PIPELINE` | Use Case, Flags | Open use case pipeline — stage, EACV, go-live date, health, tech win |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_RISK_MOVEMENTS` | Consumption | Accounts that changed risk tier — with mitigation notes |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_DAILY_CONSUMPTION` | Consumption | Daily account consumption for trend charts |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_PRODUCT_ACCOUNTS` | Consumption | Account → product category mapping |
| `AFE.DBT_DEV_MARTS.SPECIALIST_ENGAGEMENT_STATUS` | Flags | Specialist activity, comment counts, update status |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_SPECIALIST_PIPELINE` | Use Case | Specialist ↔ use case assignments |
| `AFE.DBT_DEV_OVERLAY.MART_OVERLAY_ACTIVE_USE_CASES` | Use Case | Full active use case list |

---

## Cache Policy

All tables are cached for **30 minutes** via `@st.cache_data(ttl=1800)`.
To force an immediate refresh, reload the browser tab or press **R** in Streamlit.

## Deployment

```
snow streamlit deploy --replace --connection afe_deploy --role AFE_ADMIN_RL
```

App location: `AFE.DBT_DEV.OVERLAY_DASHBOARD`
""")


# ── About module ──────────────────────────────────────────────────────────────

elif module == "About":
    st.markdown("# :material/info: About This App")
    st.divider()

    st.markdown("""
## Overlay Dashboard

Welcome to the Overlay Dashboard. Use the **Module** selector in the sidebar to
navigate between modules, and the **Page** selector to switch between pages
within each module.
""")

    st.divider()
    st.markdown("## :material/bolt: Consumption")
    st.markdown(
        "Track and analyze account-level consumption trends, identify movers, "
        "and drill into root causes."
    )

    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown("### Lights On / Lights Off")
        st.markdown("""
Tracks account-level consumption changes across comparison periods. Surfaces the
top movers (up and down), provides root cause and engagement details for each
account, and lists open use cases on filtered accounts.
""")
        if st.button("Open Lights On / Lights Off", key="ab_lo", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Lights On / Lights Off"
            st.rerun()
    with ab2:
        st.markdown("### Consumption Intelligence")
        st.markdown("""
Executive summary with headline KPIs, consumption trend charts with forecasting,
account-level consumption details by product category, monthly product breakdown,
and percentage-of-total analysis across the filtered account set.
""")
        if st.button("Open Consumption Intelligence", key="ab_ci", use_container_width=True):
            st.session_state["_nav_module"] = "Consumption"
            st.session_state["_nav_page"] = "Consumption Intelligence"
            st.rerun()

    st.divider()
    st.markdown("## :material/task_alt: Use Case")
    st.markdown(
        "Monitor the use case pipeline, track wins and go-lives, and assess "
        "specialist engagement health."
    )

    ab3, ab4 = st.columns(2)
    with ab3:
        st.markdown("### Use Case Forecasting")
        st.markdown("""
Pipeline overview with Go-Live and Win KPIs (Total Pipeline, Weighted Pipeline,
QTD Won/Deployed, Pipeline Coverage). Includes bar charts showing pipeline by
stage, a detailed wins table with decision dates and specialist comments, and a
go-lives table with health status and go-live probability tracking.
""")
        if st.button("Open Use Case Forecasting", key="ab_ucf", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Forecasting"
            st.rerun()
    with ab4:
        st.markdown("### Use Case Hygiene")
        st.markdown("""
Specialist engagement dashboard showing update staleness, activity metrics
(comments and activities over 7-day and 14-day windows), and drill-down into
active use cases per specialist with EACV, stage, and Salesforce links.
""")
        if st.button("Open Use Case Hygiene", key="ab_uch", use_container_width=True):
            st.session_state["_nav_module"] = "Use Case"
            st.session_state["_nav_page"] = "Use Case Hygiene"
            st.rerun()

    st.divider()
    ab5, ab6 = st.columns(2)
    with ab5:
        st.markdown("## :material/flag: Field Flags")
        st.markdown("""
Data-quality action list for the field. Three groups cover Specialist Data
Quality, Use Case Record Quality, and Account Engagement Gaps. Each group
surfaces a count of records needing attention with expandable tables and direct
Salesforce links. **Goal: all counts = 0.**
""")
        if st.button("Open Field Flags", key="ab_flags", use_container_width=True):
            st.session_state["_nav_module"] = "Flags"
            st.rerun()
    with ab6:
        st.markdown("## :material/menu_book: Operational Guide")
        st.markdown("""
Detailed reference covering how to interpret every KPI, when to act, threshold
definitions, data source inventory, and deployment instructions.
""")
        if st.button("Open Operational Guide", key="ab_guide", use_container_width=True):
            st.session_state["_nav_module"] = "Guide"
            st.rerun()


