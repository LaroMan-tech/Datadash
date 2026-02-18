from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Data dashboard status"
LOGO_URL = "https://www.uptraplift.nl/wp-content/uploads/2025/04/up-traplift_payoff-wit.png"
BRAND_DARK = "#060b14"
BRAND_DARK_2 = "#0b1424"
BRAND_ACCENT = "#f39c12"
STATUS_COLORS = {
    "Open": "#1f77b4",
    "Closed": "#2ca02c",
    "Final": "#9467bd",
    "Credit": "#d62728",
    "Repair": "#ff7f0e",
    "Waiting for return": "#17becf",
    "Disassembly": "#8c564b",
    "Send spare return olc": "#bcbd22",
    "Send paid": "#7f7f7f",
    "Send free spare 778": "#e377c2",
    "return to dealer with": "#aec7e8",
    "Rejected": "#c49c94",
    "<NA>": "#bbbbbb",
}
DEPARTMENT_COLORS = {
    "Technical support": "#1f77b4",
    "Financial": "#2ca02c",
    "Repair": "#ff7f0e",
    "Onbekend": "#7f7f7f",
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def normalize_status_name(value: str) -> str:
    out = "".join(ch for ch in str(value).lower().strip() if ch.isalnum() or ch.isspace())
    out = " ".join(out.split())
    return out


def load_status_map(folder: Path) -> dict[int, str]:
    mapping_path = folder / "status_mapping.csv"
    if not mapping_path.exists():
        return {}

    try:
        m = pd.read_csv(mapping_path)
        if {"code", "label"}.issubset(m.columns):
            return {
                int(float(row["code"])): str(row["label"]).strip()
                for _, row in m[["code", "label"]].dropna().iterrows()
            }
    except Exception:
        return {}
    return {}


def status_label(value, status_map: dict[int, str]) -> str:
    if pd.isna(value):
        return "<NA>"
    try:
        n = float(value)
        if n.is_integer():
            code = int(n)
            return status_map.get(code, str(code))
    except Exception:
        pass
    return str(value)


def build_color_scale(statuses: list[str]) -> alt.Scale:
    colors = [STATUS_COLORS.get(s, "#4c78a8") for s in statuses]
    return alt.Scale(domain=statuses, range=colors)


def apply_brand_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top right, {BRAND_DARK_2} 0%, {BRAND_DARK} 55%);
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0c1627 0%, #09111d 100%);
            border-right: 1px solid rgba(243, 156, 18, 0.25);
        }}
        div[data-testid="stMetric"] {{
            border: 1px solid rgba(243, 156, 18, 0.25);
            border-radius: 12px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.02);
        }}
        .brand-title {{
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff;
        }}
        .brand-subtitle {{
            margin: 0;
            color: #c6d1e6;
            font-size: 0.95rem;
        }}
        .block-container {{
            padding-top: 1.3rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_department_mapping(folder: Path) -> dict[str, str]:
    mapping_file = folder / "workload per status per afdeling.txt"
    if not mapping_file.exists():
        return {}

    mapping: dict[str, str] = {}
    lines = mapping_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        afdeling = right.strip()
        statuses = [s.strip() for s in left.split(",") if s.strip()]
        for s in statuses:
            mapping[normalize_status_name(s)] = afdeling

    # Aliasen voor kleine schrijfverschillen
    aliases = {
        "dissasembly": "disassembly",
        "send spare return old 777": "send spare return olc",
        "return to dealer with next order": "return to dealer with",
    }
    for src, target in aliases.items():
        src_n = normalize_status_name(src)
        target_n = normalize_status_name(target)
        if target_n in mapping:
            mapping[src_n] = mapping[target_n]
        elif src_n in mapping:
            mapping[target_n] = mapping[src_n]

    return mapping


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_brand_style()

    h1, h2 = st.columns([1, 3])
    with h1:
        st.image(LOGO_URL, use_container_width=True)
    with h2:
        st.markdown('<p class="brand-title">UP Traplift - Status Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="brand-subtitle">Support processen en workload per afdeling</p>', unsafe_allow_html=True)

    folder = Path(__file__).resolve().parent
    data_path = folder / "iso_service_report_202602131508.csv"

    if not data_path.exists():
        st.error(f"Bestand niet gevonden: {data_path.name}")
        st.stop()

    df = read_csv_auto(data_path)
    status_map = load_status_map(folder)
    department_mapping = load_department_mapping(folder)

    if "StatusFlag_ID" not in df.columns:
        st.error("Kolom StatusFlag_ID ontbreekt in service report data.")
        st.stop()

    if "CreatedAt" in df.columns:
        df["CreatedAt"] = pd.to_datetime(df["CreatedAt"], errors="coerce")
    if "UpdatedAt" in df.columns:
        df["UpdatedAt"] = pd.to_datetime(df["UpdatedAt"], errors="coerce")

    st.sidebar.header("Filters")
    include_na = st.sidebar.checkbox("Toon N/A status", value=True)
    include_final_in_charts = st.sidebar.checkbox("Toon Final in diagrammen", value=True)

    if "CreatedAt" in df.columns and df["CreatedAt"].notna().any():
        min_dt = df["CreatedAt"].min()
        max_dt = df["CreatedAt"].max()
        d_from, d_to = st.sidebar.date_input(
            "Datumbereik (CreatedAt)",
            value=(min_dt.date(), max_dt.date()),
        )
        df = df[(df["CreatedAt"] >= pd.Timestamp(d_from)) & (df["CreatedAt"] <= pd.Timestamp(d_to))]

    df["status_label"] = df["StatusFlag_ID"].map(lambda v: status_label(v, status_map)).astype(str)

    status_options = sorted(df["status_label"].dropna().unique().tolist())
    selected_statuses = st.sidebar.multiselect("Status", options=status_options, default=status_options)
    filtered = df[df["status_label"].isin(selected_statuses)].copy()

    chart_source = filtered.copy()
    if not include_final_in_charts:
        chart_source = chart_source[~chart_source["status_label"].str.lower().eq("final")]

    status_dist = chart_source["status_label"].value_counts(dropna=False)
    if not include_na:
        status_dist = status_dist[~status_dist.index.isin(["<NA>", "nan", "None"])]

    filtered["afdeling"] = filtered["status_label"].map(
        lambda s: department_mapping.get(normalize_status_name(s), "Onbekend")
    )

    st.subheader("KPI's")
    # KPI's op datasetniveau (niet afhankelijk van statusfilter)
    kpi_source = df.copy()
    traplift_col = "Location_ID" if "Location_ID" in kpi_source.columns else None
    created_col = "CreatedAt" if "CreatedAt" in kpi_source.columns else None

    errors_per_traplift = None
    service_reports_per_traplift = None
    reports_per_month_value = None
    reports_per_month_delta = None
    forecast_next_month = None
    forecast_note = None

    traplift_count = 0
    if traplift_col and kpi_source[traplift_col].notna().any():
        traplift_count = int(kpi_source[traplift_col].nunique(dropna=True))
        if traplift_count > 0:
            service_reports_per_traplift = len(kpi_source) / traplift_count

    # Errors per traplift alleen wanneer er echt een error-code kolom is.
    error_candidates = [
        "ErrorCode",
        "Error_Code",
        "ErrorCode_ID",
        "ErrorCodeId",
        "FaultCode",
        "Fault_Code",
    ]
    error_col = next((c for c in error_candidates if c in kpi_source.columns), None)
    if error_col and traplift_count > 0 and kpi_source[error_col].notna().any():
        errors_per_traplift = kpi_source[error_col].notna().sum() / traplift_count

    # Servicerapporten per maand = laatste maand vs maand ervoor.
    if created_col and kpi_source[created_col].notna().any():
        monthly_counts = (
            kpi_source.dropna(subset=[created_col])
            .groupby(kpi_source[created_col].dt.to_period("M"))
            .size()
            .sort_index()
        )
        if len(monthly_counts) >= 2:
            latest_period = monthly_counts.index[-1]
            prev_period = monthly_counts.index[-2]
            latest_count = int(monthly_counts.iloc[-1])
            prev_count = int(monthly_counts.iloc[-2])
            reports_per_month_value = latest_count
            diff = latest_count - prev_count
            sign = "+" if diff >= 0 else ""
            reports_per_month_delta = f"{sign}{diff} vs {prev_period.strftime('%b %Y')}"
        elif len(monthly_counts) == 1:
            reports_per_month_value = int(monthly_counts.iloc[-1])
            reports_per_month_delta = "geen vorige maand in dataset"

        if len(monthly_counts) >= 3:
            x = np.arange(len(monthly_counts))
            y = monthly_counts.values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * len(monthly_counts) + intercept
            forecast_next_month = max(0, int(round(pred)))
            next_period = (monthly_counts.index[-1] + 1).strftime("%b %Y")
            forecast_note = f"lineaire forecast voor {next_period}"

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric(
        "Errors per traplift",
        f"{errors_per_traplift:.2f}" if errors_per_traplift is not None else "n.v.t.",
    )
    kc2.metric(
        "Servicerapporten per traplift",
        f"{service_reports_per_traplift:.2f}" if service_reports_per_traplift is not None else "n.v.t.",
    )
    kc3.metric(
        "Servicerapporten per maand",
        f"{reports_per_month_value}" if reports_per_month_value is not None else "n.v.t.",
        reports_per_month_delta if reports_per_month_delta is not None else None,
    )
    kc4.metric(
        "Forecast volgende maand",
        f"{forecast_next_month}" if forecast_next_month is not None else "n.v.t.",
        forecast_note if forecast_note is not None else None,
    )

    if created_col and kpi_source[created_col].notna().any():
        monthly_counts_plot = (
            kpi_source.dropna(subset=[created_col])
            .groupby(kpi_source[created_col].dt.to_period("M"))
            .size()
            .sort_index()
        )
        if not monthly_counts_plot.empty:
            st.subheader("Forecast grafiek servicerapporten per maand")
            hist_df = pd.DataFrame(
                {
                    "periode": monthly_counts_plot.index.to_timestamp(),
                    "aantal": monthly_counts_plot.values,
                    "type": "Historisch",
                }
            )
            layers = []
            hist_line = (
                alt.Chart(hist_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("periode:T", title="Maand"),
                    y=alt.Y("aantal:Q", title="Aantal servicerapporten"),
                    color=alt.Color(
                        "type:N",
                        scale=alt.Scale(domain=["Historisch", "Forecast"], range=["#66b3ff", "#f39c12"]),
                        legend=alt.Legend(title="Reeks"),
                    ),
                    tooltip=["periode:T", "aantal:Q", "type:N"],
                )
            )
            layers.append(hist_line)

            if forecast_next_month is not None:
                next_period_ts = (monthly_counts_plot.index[-1] + 1).to_timestamp()
                last_hist_ts = monthly_counts_plot.index[-1].to_timestamp()
                last_hist_val = int(monthly_counts_plot.iloc[-1])
                forecast_line_df = pd.DataFrame(
                    {
                        "periode": [last_hist_ts, next_period_ts],
                        "aantal": [last_hist_val, int(forecast_next_month)],
                        "type": ["Forecast", "Forecast"],
                    }
                )
                forecast_point_df = pd.DataFrame(
                    {
                        "periode": [next_period_ts],
                        "aantal": [int(forecast_next_month)],
                        "type": ["Forecast"],
                    }
                )
                forecast_line = (
                    alt.Chart(forecast_line_df)
                    .mark_line(strokeDash=[6, 4], point=False)
                    .encode(
                        x="periode:T",
                        y="aantal:Q",
                        color=alt.Color(
                            "type:N",
                            scale=alt.Scale(domain=["Historisch", "Forecast"], range=["#66b3ff", "#f39c12"]),
                            legend=alt.Legend(title="Reeks"),
                        ),
                        tooltip=["periode:T", "aantal:Q", "type:N"],
                    )
                )
                forecast_point = (
                    alt.Chart(forecast_point_df)
                    .mark_circle(size=120)
                    .encode(
                        x="periode:T",
                        y="aantal:Q",
                        color=alt.Color(
                            "type:N",
                            scale=alt.Scale(domain=["Historisch", "Forecast"], range=["#66b3ff", "#f39c12"]),
                            legend=alt.Legend(title="Reeks"),
                        ),
                        tooltip=["periode:T", "aantal:Q", "type:N"],
                    )
                )
                layers.extend([forecast_line, forecast_point])

            forecast_chart = alt.layer(*layers).properties(height=340)
            st.altair_chart(forecast_chart, use_container_width=True)

    st.subheader("Workload per afdeling")
    workload_source = filtered[
        (~filtered["status_label"].str.lower().eq("final"))
        & (~filtered["status_label"].isin(["<NA>", "nan", "None"]))
    ].copy()
    workload_df = (
        workload_source.groupby("afdeling")
        .agg(
            totaal_werkload=("status_label", "count"),
            open_statussen=("status_label", lambda s: int((s.str.lower() == "open").sum())),
        )
        .reset_index()
        .sort_values("totaal_werkload", ascending=False)
    )
    st.dataframe(workload_df, use_container_width=True, height=220)
    dept_order = workload_df["afdeling"].tolist()
    dept_colors = [DEPARTMENT_COLORS.get(d, "#4c78a8") for d in dept_order]
    dept_chart = (
        alt.Chart(workload_df)
        .mark_bar()
        .encode(
            x=alt.X("afdeling:N", sort=dept_order, title="Afdeling"),
            y=alt.Y("totaal_werkload:Q", title="Totaal workload"),
            color=alt.Color(
                "afdeling:N",
                scale=alt.Scale(domain=dept_order, range=dept_colors),
                title="Afdeling",
            ),
            tooltip=["afdeling:N", "totaal_werkload:Q", "open_statussen:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(dept_chart, use_container_width=True)

    st.subheader("Grafiek statusverdeling (gefilterde dataset)")
    bar_df = status_dist.rename_axis("status").reset_index(name="aantal")
    status_order = bar_df["status"].tolist()
    color_scale = build_color_scale(status_order)
    bar_chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X("status:N", sort="-y", title="Status"),
            y=alt.Y("aantal:Q", title="Aantal"),
            color=alt.Color("status:N", scale=color_scale, legend=None),
            tooltip=["status:N", "aantal:Q"],
        )
        .properties(height=360)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("Cirkeldiagram statusverdeling")
    pie_df = bar_df.copy()
    pie_chart = (
        alt.Chart(pie_df)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="aantal", type="quantitative"),
            color=alt.Color(field="status", type="nominal", title="Status", scale=color_scale),
            tooltip=["status:N", "aantal:Q"],
        )
        .properties(height=420)
    )
    st.altair_chart(pie_chart, use_container_width=True)

    st.subheader("Stacked diagram status over tijd")
    if "CreatedAt" in chart_source.columns and chart_source["CreatedAt"].notna().any():
        stacked_df = chart_source[["CreatedAt", "status_label"]].dropna(subset=["CreatedAt"]).copy()
        if not include_na:
            stacked_df = stacked_df[~stacked_df["status_label"].isin(["<NA>", "nan", "None"])]
        stacked_df["periode"] = stacked_df["CreatedAt"].dt.to_period("M").dt.to_timestamp()
        stacked_monthly = (
            stacked_df.groupby(["periode", "status_label"])
            .size()
            .reset_index(name="aantal")
        )
        stacked_chart = (
            alt.Chart(stacked_monthly)
            .mark_area()
            .encode(
                x=alt.X("periode:T", title="Maand"),
                y=alt.Y("aantal:Q", stack="zero", title="Aantal per maand"),
                color=alt.Color("status_label:N", title="Status", scale=color_scale),
                tooltip=["periode:T", "status_label:N", "aantal:Q"],
            )
            .properties(height=420)
        )
        st.altair_chart(stacked_chart, use_container_width=True)

        st.subheader("Stacked diagram over tijd (cumulatief, area stijl)")
        cumulative_wide = (
            stacked_monthly.pivot(index="periode", columns="status_label", values="aantal")
            .fillna(0)
            .sort_index()
        )
        cumulative_wide = cumulative_wide.cumsum()
        cumulative = (
            cumulative_wide.reset_index()
            .melt(id_vars="periode", var_name="status_label", value_name="aantal_cumulatief")
        )
        cumulative_chart = (
            alt.Chart(cumulative)
            .mark_area(opacity=0.9)
            .encode(
                x=alt.X("periode:T", title="Maand", axis=alt.Axis(format="%b %Y", labelAngle=0)),
                y=alt.Y("aantal_cumulatief:Q", stack="zero", title="Cumulatief aantal records"),
                color=alt.Color(
                    "status_label:N",
                    title="Status",
                    scale=color_scale,
                    legend=alt.Legend(orient="right"),
                ),
                tooltip=["periode:T", "status_label:N", "aantal_cumulatief:Q"],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(cumulative_chart, use_container_width=True)
    else:
        st.info("Geen CreatedAt data beschikbaar voor stacked diagram over tijd.")

    st.subheader("Doorlooptijd en statusduur")
    now_ts = pd.Timestamp.now()
    analysis_df = filtered.copy()
    if not include_na:
        analysis_df = analysis_df[~analysis_df["status_label"].isin(["<NA>", "nan", "None"])]

    if "UpdatedAt" in analysis_df.columns and analysis_df["UpdatedAt"].notna().any():
        analysis_df["dagen_op_huidige_status"] = (
            (now_ts - analysis_df["UpdatedAt"]).dt.total_seconds() / 86400.0
        )
        active_status_df = analysis_df[
            ~analysis_df["status_label"].str.lower().eq("final")
        ].copy()
        status_age = (
            active_status_df.dropna(subset=["dagen_op_huidige_status"])
            .groupby("status_label")["dagen_op_huidige_status"]
            .agg(
                aantal="count",
                gemiddeld_dagen="mean",
                mediaan_dagen="median",
                p90_dagen=lambda x: x.quantile(0.90),
            )
            .reset_index()
            .sort_values("mediaan_dagen", ascending=False)
        )
        status_age["gemiddeld_dagen"] = status_age["gemiddeld_dagen"].round(1)
        status_age["mediaan_dagen"] = status_age["mediaan_dagen"].round(1)
        status_age["p90_dagen"] = status_age["p90_dagen"].round(1)

        st.markdown("**Tijd op huidige status (sinds laatste update, exclusief Final)**")
        st.dataframe(status_age, use_container_width=True, height=320)
    else:
        st.info("Geen UpdatedAt data beschikbaar voor statusduur.")

    if (
        "CreatedAt" in analysis_df.columns
        and "UpdatedAt" in analysis_df.columns
        and analysis_df["CreatedAt"].notna().any()
        and analysis_df["UpdatedAt"].notna().any()
    ):
        final_df = analysis_df[
            analysis_df["status_label"].str.lower().eq("final")
        ].copy()
        final_df["doorlooptijd_dagen"] = (
            (final_df["UpdatedAt"] - final_df["CreatedAt"]).dt.total_seconds() / 86400.0
        )
        final_df = final_df.dropna(subset=["doorlooptijd_dagen"])
        final_df = final_df[final_df["doorlooptijd_dagen"] >= 0]

        st.markdown("**Doorlooptijd Open -> Final (benadering: CreatedAt -> UpdatedAt voor Final records)**")
        if not final_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final records", int(len(final_df)))
            c2.metric("Gemiddeld (dagen)", f"{final_df['doorlooptijd_dagen'].mean():.1f}")
            c3.metric("Mediaan (dagen)", f"{final_df['doorlooptijd_dagen'].median():.1f}")
            c4.metric("P90 (dagen)", f"{final_df['doorlooptijd_dagen'].quantile(0.90):.1f}")

            hist_df = final_df["doorlooptijd_dagen"].round(1).to_frame(name="dagen")
            st.bar_chart(hist_df["dagen"].value_counts().sort_index())
        else:
            st.info("Geen bruikbare Final records gevonden voor doorlooptijd.")
    else:
        st.info("Onvoldoende CreatedAt/UpdatedAt data voor doorlooptijdanalyse.")


if __name__ == "__main__":
    main()
