from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import streamlit as st


MAIL_FOLDER = Path(__file__).resolve().parent / "support mails"

E_TITLES = {
    "E0": "No_Tilt_Sensor_Detected",
    "E1": "Jammed_Spindle_Detected",
    "E2": "Stairlift_No_Defined_Orientation",
    "E3": "Overspeed_Safety_Gear_Activated",
    "E4": "Safety_Pic_Watchdog_Triggered",
    "E5": "Critical_Low_Voltage_Drop",
    "E6": "Traction_Motor_Encoder_Failure",
    "E7": "Spindle_Motor_Encoder_Failure",
    "E8": "Traction_Motor_Overspeed_Detected",
    "E9": "Traction_Current_Supply_Fail",
    "E10": "Traction_Motor_Over_Current",
    "E11": "Spindle_Current_Supply_Fail",
    "E12": "Spindle_Motor_Over_Current",
    "E13": "Traction_Brake_Failed",
    "E14": "Encoder_Supply_Failed",
    "E15": "System_Over_Current",
    "E16": "Swivel_Motor_Encoder_Failure",
    "E17": "Safety_Line_Footrest_Failure",
    "E18": "Safety_Line_Guidance_Failure",
    "E19": "Safety_Line_Motor_Failure",
    "E20": "Swivel_Encoder_Wire_Failure",
    "E21": "Spindle_Encoder_Wire_Failure",
    "E22": "Traction_Encoder_Wire_Failure",
    "E23": "Chair_PCB_5_Degree_Violation",
    "E24": "Safety_Lines_Supply_Failure",
    "E25": "Swivel_Current_Supply_Fail",
    "E26": "Swivel_Motor_Over_Current",
    "E27": "Communication_Wrong_Message_Number",
    "E28": "SRAM_Test_Failed",
    "E29": "Spindle_No_Current_Detected",
    "E30": "SmartSwitchTest_Failed",
    "E31": "Footrest_Current_Supply_Fail",
}

W_TITLES = {
    "W0": "Tilt_Sensor_Too_Tilted",
    "W1": "Rail_Config_Incomplete",
    "W2": "Traction_Encoder_Out_Of_Bound",
    "W3": "Spindle_Encoder_Out_Of_Bound",
    "W4": "Not_Charging",
    "W5": "Safety_Line_Interrupted",
    "W6": "Low_Supply_Voltage",
    "W7": "Chair_Error_Too_High",
    "W8": "Low_Supply_Voltage_Since_Bootup",
    "W9": "Wrong_Logic_Voltage",
    "W10": "Angle_Comparator_Triggered",
    "W11": "Friction_Wheel_Slipping",
    "W12": "Communication_Timed_Out",
    "W13": "ArmRests_Or_Seatbelt_Not_In_Position",
    "W14": "Traction_Encoder_Illegal_Transitions",
    "W15": "Swivel_Encoder_Out_Of_Bound",
    "W16": "Emergency_State_Active",
    "W17": "Safety_Footrest_Left_Pressed",
    "W18": "Safety_Footrest_Right_Pressed",
    "W19": "Safety_Guidance_Up_Pressed",
    "W20": "Safety_Guidance_Down_Pressed",
    "W21": "Safety_Motor_Up_Pressed",
    "W22": "Safety_Motor_Down_Pressed",
    "W23": "KeySwitch_Active",
    "W24": "Temperature_Too_High",
    "W25": "Hingrail_Fold_Fail",
    "W26": "Rail_Checksum_Incorrect",
    "W27": "Safety_Pic_Intervened",
    "W28": "Rail_Improbable",
    "W29": "EEprom_CRC_Incorrect",
    "W30": "Chair_Not_Closed",
    "W31": "Faulty_Friction",
}


@dataclass
class MailRecord:
    file_name: str
    date: pd.Timestamp | pd.NaT
    up_number: int | None
    company: str
    errors: list[str]
    warnings: list[str]
    safeties: list[str]


def parse_header_line(text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_date(text: str) -> pd.Timestamp | pd.NaT:
    raw = parse_header_line(text, "Date")
    if not raw:
        return pd.NaT
    return pd.to_datetime(raw, errors="coerce")


def parse_up_number(text: str, file_name: str) -> int | None:
    pool = f"{file_name}\n{text[:3000]}"
    m = re.search(r"UP-(\d{1,7})", pool, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_codes(text: str, code_type: str, valid_codes: set[str] | None = None) -> list[str]:
    """
    Parse codes with priority on explicit identification lines, e.g.:
      Error Identification nr: 8-...
      Warning Identification nr: 4-...
    Fallback to compact |E| / |W| blocks if identification lines are absent.
    """
    prefix = code_type.upper()
    kind = "Error" if prefix == "E" else "Warning" if prefix == "W" else "Safety"

    out: list[str] = []
    seen: set[str] = set()

    # Preferred source: explicit identification lines
    ident_nums = re.findall(
        rf"{kind}\s+Identification\s+nr:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    for n in ident_nums:
        code = f"{prefix}{int(n)}"
        if valid_codes is not None and code not in valid_codes:
            continue
        if code not in seen:
            seen.add(code)
            out.append(code)

    if out:
        return out

    # Fallback source: compact block like |E| 0 0 1 0
    m = re.search(rf"\|{prefix}\|([^|\n\r]*)", text, flags=re.IGNORECASE)
    if not m:
        return []

    nums = re.findall(r"\d+", m.group(1))
    for n in nums:
        if n == "0":
            continue
        code = f"{prefix}{int(n)}"
        if valid_codes is not None and code not in valid_codes:
            continue
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


@st.cache_data(show_spinner=False)
def load_mail_records(folder: str) -> pd.DataFrame:
    root = Path(folder)
    rows: list[dict] = []
    valid_e_codes = set(E_TITLES.keys())
    valid_w_codes = set(W_TITLES.keys())
    for p in sorted(root.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        date = parse_date(text)
        up_number = parse_up_number(text, p.name)
        company = parse_header_line(text, "From") or "Unknown"
        errors = parse_codes(text, "E", valid_codes=valid_e_codes)
        warnings = parse_codes(text, "W", valid_codes=valid_w_codes)
        safeties = parse_codes(text, "S")

        rows.append(
            {
                "file_name": p.name,
                "date": date,
                "up_number": up_number,
                "company": company,
                "errors": errors,
                "warnings": warnings,
                "safeties": safeties,
            }
        )

    return pd.DataFrame(rows)


def counts_to_df(counter: Counter, title_map: dict[str, str]) -> pd.DataFrame:
    if not counter:
        return pd.DataFrame(columns=["code", "count", "title", "pct"])
    df = pd.DataFrame(counter.items(), columns=["code", "count"]).sort_values("count", ascending=False)
    total = float(df["count"].sum())
    df["title"] = df["code"].map(lambda c: title_map.get(c, "Unknown"))
    df["pct"] = (df["count"] / total * 100).round(2)
    return df


def render_bar(df: pd.DataFrame, subtitle: str) -> None:
    st.caption(subtitle)
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("code:N", sort="-y", title="Code"),
            y=alt.Y("count:Q", title="Aantal"),
            tooltip=["code:N", "count:Q", "title:N", "pct:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_pie(df: pd.DataFrame, subtitle: str) -> None:
    st.caption(subtitle)
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return
    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("code:N", title="Code"),
            tooltip=["code:N", "count:Q", "pct:Q", "title:N"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_bar(ax, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "Geen data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return
    plot_df = df.head(12).copy()
    ax.bar(plot_df["code"], plot_df["count"])
    ax.set_title(title)
    ax.set_ylabel("Aantal")
    ax.tick_params(axis="x", rotation=45)


def _plot_pie(ax, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "Geen data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return
    plot_df = df.head(10).copy()
    ax.pie(plot_df["count"], labels=plot_df["code"], autopct="%1.1f%%")
    ax.set_title(title)


def build_pdf_report(
    filt: pd.DataFrame,
    err_df: pd.DataFrame,
    warn_df: pd.DataFrame,
    safe_df: pd.DataFrame,
    from_date,
    to_date,
    from_up: int,
    to_up: int,
    company: str,
) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.suptitle("Issue distribution from Service Reports", fontsize=16, fontweight="bold")
        txt = (
            f"Periode: {from_date} t/m {to_date}\n"
            f"UP-range: {from_up} t/m {to_up}\n"
            f"Company: {company}\n"
            f"Aantal service reports: {len(filt)}\n"
            f"Rapporten met errors: {int((filt['errors'].map(len) > 0).sum())}\n"
            f"Rapporten met warnings: {int((filt['warnings'].map(len) > 0).sum())}"
        )
        fig.text(0.05, 0.72, txt, fontsize=11, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        _plot_bar(axs[0, 0], err_df, "Errors - top")
        _plot_pie(axs[0, 1], err_df, "Errors distribution")
        _plot_bar(axs[0, 2], warn_df, "Warnings - top")
        _plot_pie(axs[1, 0], warn_df, "Warnings distribution")
        _plot_bar(axs[1, 1], safe_df, "Safeties - top")
        _plot_pie(axs[1, 2], safe_df, "Safeties distribution")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        trend = filt.copy()
        trend = trend[trend["date"].notna()].copy()
        if not trend.empty:
            trend["period"] = trend["date"].dt.to_period("M").dt.to_timestamp()
            trend["has_error"] = trend["errors"].map(lambda x: 1 if len(x) > 0 else 0)
            trend["has_warning"] = trend["warnings"].map(lambda x: 1 if len(x) > 0 else 0)
            trend_df = trend.groupby("period")[["has_error", "has_warning"]].sum().reset_index()

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(trend_df["period"], trend_df["has_error"], marker="o", label="Errors")
            ax.plot(trend_df["period"], trend_df["has_warning"], marker="o", label="Warnings")
            ax.set_title("Trend over tijd: rapporten met errors en warnings")
            ax.set_xlabel("Maand")
            ax.set_ylabel("Aantal rapporten")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            err_rows = []
            warn_rows = []
            for _, r in trend.iterrows():
                for code in r["errors"]:
                    err_rows.append({"period": r["period"], "code": code})
                for code in r["warnings"]:
                    warn_rows.append({"period": r["period"], "code": code})

            if err_rows or warn_rows:
                fig, axs = plt.subplots(2, 1, figsize=(14, 10))

                if err_rows:
                    err_trend = pd.DataFrame(err_rows)
                    err_top = err_trend["code"].value_counts().head(8).index.tolist()
                    err_trend = err_trend[err_trend["code"].isin(err_top)]
                    err_trend = err_trend.groupby(["period", "code"]).size().reset_index(name="count")
                    err_pivot = (
                        err_trend.pivot(index="period", columns="code", values="count")
                        .fillna(0)
                        .sort_index()
                    )
                    axs[0].stackplot(err_pivot.index, err_pivot.T.values, labels=err_pivot.columns)
                    axs[0].set_title("Error trend per code (Top 8)")
                    axs[0].set_ylabel("Aantal")
                    axs[0].grid(alpha=0.2)
                    axs[0].legend(loc="upper left", ncols=4, fontsize=8)
                else:
                    axs[0].text(0.5, 0.5, "Geen error trend data", ha="center", va="center")
                    axs[0].set_title("Error trend per code (Top 8)")
                    axs[0].axis("off")

                if warn_rows:
                    warn_trend = pd.DataFrame(warn_rows)
                    warn_top = warn_trend["code"].value_counts().head(8).index.tolist()
                    warn_trend = warn_trend[warn_trend["code"].isin(warn_top)]
                    warn_trend = warn_trend.groupby(["period", "code"]).size().reset_index(name="count")
                    warn_pivot = (
                        warn_trend.pivot(index="period", columns="code", values="count")
                        .fillna(0)
                        .sort_index()
                    )
                    axs[1].stackplot(warn_pivot.index, warn_pivot.T.values, labels=warn_pivot.columns)
                    axs[1].set_title("Warning trend per code (Top 8)")
                    axs[1].set_ylabel("Aantal")
                    axs[1].set_xlabel("Maand")
                    axs[1].grid(alpha=0.2)
                    axs[1].legend(loc="upper left", ncols=4, fontsize=8)
                else:
                    axs[1].text(0.5, 0.5, "Geen warning trend data", ha="center", va="center")
                    axs[1].set_title("Warning trend per code (Top 8)")
                    axs[1].axis("off")

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Issue Distribution", layout="wide")
    st.title("Issue distribution from Service Reports")

    if not MAIL_FOLDER.exists():
        st.error(f"Map niet gevonden: {MAIL_FOLDER}")
        st.stop()

    data = load_mail_records(str(MAIL_FOLDER))
    if data.empty:
        st.warning("Geen leesbare .txt mails gevonden.")
        st.stop()

    date_series = data["date"].dropna()
    min_date = date_series.min().date() if not date_series.empty else pd.Timestamp("2024-01-01").date()
    max_date = date_series.max().date() if not date_series.empty else pd.Timestamp.today().date()

    up_vals = data["up_number"].dropna().astype(int)
    up_vals = up_vals[up_vals >= 1]
    min_up = int(up_vals.min()) if not up_vals.empty else 1
    max_up = int(up_vals.max()) if not up_vals.empty else 9999
    if max_up < min_up:
        max_up = min_up

    companies = sorted(data["company"].fillna("Unknown").astype(str).unique().tolist())

    st.markdown("### Filters")
    fcol, xcol = st.columns([3, 1])
    with fcol:
        with st.form("filters_form"):
            c1, c2 = st.columns(2)
            from_date = c1.date_input("From date", min_date)
            to_date = c2.date_input("To date", max_date)

            c3, c4 = st.columns(2)
            from_up = c3.number_input("From UP-number", min_value=1, value=max(1, int(min_up)), step=1)
            to_up = c4.number_input("To UP-number", min_value=1, value=max(1, int(max_up)), step=1)

            company = st.selectbox("Company", options=["All companies"] + companies)
            submitted = st.form_submit_button("Submit")

    filt = data.copy()
    filt = filt[filt["date"].notna()]
    filt = filt[(filt["date"] >= pd.Timestamp(from_date)) & (filt["date"] <= pd.Timestamp(to_date))]
    filt = filt[filt["up_number"].notna()]
    filt = filt[(filt["up_number"] >= int(from_up)) & (filt["up_number"] <= int(to_up))]
    if company != "All companies":
        filt = filt[filt["company"] == company]

    reports_with_errors = filt[filt["errors"].map(lambda x: len(x) > 0)].copy()
    reports_with_warnings = filt[filt["warnings"].map(lambda x: len(x) > 0)].copy()

    err_counter = Counter()
    warn_counter = Counter()
    safe_counter = Counter()
    for _, r in filt.iterrows():
        err_counter.update(r["errors"])
        warn_counter.update(r["warnings"])
        safe_counter.update(r["safeties"])

    err_df = counts_to_df(err_counter, E_TITLES)
    warn_df = counts_to_df(warn_counter, W_TITLES)
    safe_df = counts_to_df(safe_counter, W_TITLES)

    with xcol:
        st.markdown("### Export")
        pdf_bytes = build_pdf_report(
            filt=filt,
            err_df=err_df,
            warn_df=warn_df,
            safe_df=safe_df,
            from_date=from_date,
            to_date=to_date,
            from_up=int(from_up),
            to_up=int(to_up),
            company=company,
        )
        st.download_button(
            "Export pagina naar PDF",
            data=pdf_bytes,
            file_name="issue_distribution_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption("Bevat rapportsecties met grafieken, zonder filtermenu.")

    st.markdown("### Reports with errors and warnings")
    k1, k2 = st.columns(2)
    k1.metric("Rapporten met errors", len(reports_with_errors))
    k2.metric("Rapporten met warnings", len(reports_with_warnings))

    l1, l2 = st.columns(2)
    with l1:
        st.write("**In welke rapporten zitten errors**")
        if reports_with_errors.empty:
            st.info("Geen rapporten met errors in deze selectie.")
        else:
            err_list = reports_with_errors[
                ["file_name", "date", "up_number", "company", "errors"]
            ].copy()
            err_list["errors_nr"] = err_list["errors"].map(
                lambda codes: ", ".join(c.replace("E", "") for c in codes)
            )
            err_list["errors"] = err_list["errors"].map(lambda x: ", ".join(x))
            err_list = err_list.rename(columns={"errors_nr": "errors_nr (identification nr)"})
            st.dataframe(err_list, use_container_width=True, height=320)
    with l2:
        st.write("**In welke rapporten zitten warnings**")
        if reports_with_warnings.empty:
            st.info("Geen rapporten met warnings in deze selectie.")
        else:
            warn_list = reports_with_warnings[
                ["file_name", "date", "up_number", "company", "warnings"]
            ].copy()
            warn_list["warnings_nr"] = warn_list["warnings"].map(
                lambda codes: ", ".join(c.replace("W", "") for c in codes)
            )
            warn_list["warnings"] = warn_list["warnings"].map(lambda x: ", ".join(x))
            warn_list = warn_list.rename(columns={"warnings_nr": "warnings_nr (identification nr)"})
            st.dataframe(warn_list, use_container_width=True, height=320)

    st.markdown("### Trend over tijd: errors en warnings")
    trend_base = filt.copy()
    trend_base = trend_base[trend_base["date"].notna()].copy()
    if trend_base.empty:
        st.info("Geen datumdata beschikbaar voor trendgrafiek.")
    else:
        trend_base["period"] = trend_base["date"].dt.to_period("M").dt.to_timestamp()
        trend_base["has_error"] = trend_base["errors"].map(lambda x: 1 if len(x) > 0 else 0)
        trend_base["has_warning"] = trend_base["warnings"].map(lambda x: 1 if len(x) > 0 else 0)

        trend_df = (
            trend_base.groupby("period")[["has_error", "has_warning"]]
            .sum()
            .reset_index()
            .rename(columns={"has_error": "Errors", "has_warning": "Warnings"})
        )
        trend_long = trend_df.melt(
            id_vars="period",
            value_vars=["Errors", "Warnings"],
            var_name="type",
            value_name="count",
        )
        line_chart = (
            alt.Chart(trend_long)
            .mark_area(interpolate="monotone", opacity=0.85)
            .encode(
                x=alt.X("period:T", title="Maand"),
                y=alt.Y("count:Q", stack="zero", title="Aantal rapporten"),
                color=alt.Color("type:N", title="Type", scale=alt.Scale(domain=["Errors", "Warnings"], range=["#ff6b6b", "#66b3ff"])),
                tooltip=["period:T", "type:N", "count:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(line_chart, use_container_width=True)

        st.markdown("#### Error trend per code")
        err_mode = st.selectbox(
            "Error codes tonen",
            options=["Top 5", "Alle"],
            index=0,
            key="err_top_mode",
        )
        err_rows = []
        for _, r in trend_base.iterrows():
            for code in r["errors"]:
                err_rows.append({"period": r["period"], "code": code})
        if err_rows:
            err_trend = pd.DataFrame(err_rows)
            err_top_codes = (
                err_trend["code"].value_counts().head(5).index.tolist()
            )
            if err_mode == "Top 5":
                err_trend = err_trend[err_trend["code"].isin(err_top_codes)]
            err_trend = err_trend.groupby(["period", "code"]).size().reset_index(name="count")
            err_line = (
                alt.Chart(err_trend)
                .mark_area(interpolate="monotone", opacity=0.85)
                .encode(
                    x=alt.X("period:T", title="Maand"),
                    y=alt.Y("count:Q", stack="zero", title="Aantal error meldingen"),
                    color=alt.Color("code:N", title="Error code"),
                    tooltip=["period:T", "code:N", "count:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(err_line, use_container_width=True)
        else:
            st.info("Geen errors beschikbaar voor trend per code.")

        st.markdown("#### Warning trend per code")
        warn_mode = st.selectbox(
            "Warning codes tonen",
            options=["Top 5", "Alle"],
            index=0,
            key="warn_top_mode",
        )
        warn_rows = []
        for _, r in trend_base.iterrows():
            for code in r["warnings"]:
                warn_rows.append({"period": r["period"], "code": code})
        if warn_rows:
            warn_trend = pd.DataFrame(warn_rows)
            warn_top_codes = (
                warn_trend["code"].value_counts().head(5).index.tolist()
            )
            if warn_mode == "Top 5":
                warn_trend = warn_trend[warn_trend["code"].isin(warn_top_codes)]
            warn_trend = warn_trend.groupby(["period", "code"]).size().reset_index(name="count")
            warn_line = (
                alt.Chart(warn_trend)
                .mark_area(interpolate="monotone", opacity=0.85)
                .encode(
                    x=alt.X("period:T", title="Maand"),
                    y=alt.Y("count:Q", stack="zero", title="Aantal warning meldingen"),
                    color=alt.Color("code:N", title="Warning code"),
                    tooltip=["period:T", "code:N", "count:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(warn_line, use_container_width=True)
        else:
            st.info("Geen warnings beschikbaar voor trend per code.")

    st.markdown("### Amount of service reports")
    st.metric("Amount of service reports", len(filt))

    st.markdown("### Distributions")
    b1, p1 = st.columns(2)
    with b1:
        render_bar(err_df, f"Errors from {from_date} to {to_date}")
    with p1:
        render_pie(err_df, f"Error distribution from {from_date} to {to_date}")

    b2, p2 = st.columns(2)
    with b2:
        render_bar(warn_df, f"Warnings from {from_date} to {to_date}")
    with p2:
        render_pie(warn_df, f"Warning distribution from {from_date} to {to_date}")

    b3, p3 = st.columns(2)
    with b3:
        render_bar(safe_df, f"Safeties from {from_date} to {to_date}")
    with p3:
        render_pie(safe_df, f"Safety distribution from {from_date} to {to_date}")

    st.markdown("### Code Reference")
    t1, t2 = st.columns(2)
    with t1:
        st.write("**Warning codes**")
        w_ref = pd.DataFrame(sorted(W_TITLES.items()), columns=["Code", "Title"])
        w_ref["NR"] = w_ref["Code"].str.replace("W", "", regex=False).astype(int)
        w_ref = w_ref[["NR", "Code", "Title"]]
        st.dataframe(w_ref, use_container_width=True, height=520)
    with t2:
        st.write("**Error codes**")
        e_ref = pd.DataFrame(sorted(E_TITLES.items()), columns=["Code", "Title"])
        e_ref["NR"] = e_ref["Code"].str.replace("E", "", regex=False).astype(int)
        e_ref = e_ref[["NR", "Code", "Title"]]
        st.dataframe(e_ref, use_container_width=True, height=520)


if __name__ == "__main__":
    main()
