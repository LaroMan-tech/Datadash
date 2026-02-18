from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Service Rapport Dashboard"
DEFAULT_STATUS_MAP = {
    18: {"key": "OPEN", "label": "open"},
    20: {"key": "REPAIR", "label": "repair"},
    30: {"key": "CREDIT", "label": "credit"},
    31: {"key": "STATUS_31", "label": "status_31"},
    32: {"key": "STATUS_32", "label": "status_32"},
    33: {"key": "STATUS_33", "label": "status_33"},
    34: {"key": "STATUS_34", "label": "status_34"},
    35: {"key": "STATUS_35", "label": "status_35"},
    66: {"key": "STATUS_66", "label": "status_66"},
    67: {"key": "STATUS_67", "label": "status_67"},
    68: {"key": "STATUS_68", "label": "status_68"},
    69: {"key": "STATUS_69", "label": "status_69"},
    71: {"key": "STATUS_71", "label": "status_71"},
}


def read_table_auto(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path.name}")


def auto_parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_hints = ("date", "datum", "tijd", "time", "signed", "created", "updated")
    candidate_cols = [
        col for col in df.columns if any(hint in col.lower() for hint in date_hints)
    ]

    for col in candidate_cols:
        if not pd.api.types.is_object_dtype(df[col]):
            continue

        non_null = int(df[col].notna().sum())
        if non_null == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted_default = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
            converted_dayfirst = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        success_default = converted_default.notna().sum() / non_null
        success_dayfirst = converted_dayfirst.notna().sum() / non_null

        if max(success_default, success_dayfirst) >= 0.6:
            df[col] = converted_dayfirst if success_dayfirst > success_default else converted_default

    return df


@st.cache_data(show_spinner=False)
def load_datasets(folder: str) -> dict[str, pd.DataFrame]:
    root = Path(folder)
    files = sorted(list(root.glob("*.csv")) + list(root.glob("*.xlsx")) + list(root.glob("*.xls")))

    datasets: dict[str, pd.DataFrame] = {}
    for file_path in files:
        if file_path.name.startswith("~$"):
            continue
        if file_path.name.lower().startswith("status_mapping"):
            continue
        df = read_table_auto(file_path)
        df = auto_parse_date_columns(df)
        datasets[file_path.stem] = df

    return datasets


@st.cache_data(show_spinner=False)
def load_info_datasheets(folder: str) -> dict[str, pd.DataFrame]:
    info_dir = Path(folder) / "Info datasheets"
    if not info_dir.exists():
        return {}

    files = sorted(list(info_dir.glob("*.csv")) + list(info_dir.glob("*.xlsx")) + list(info_dir.glob("*.xls")))
    info_tables: dict[str, pd.DataFrame] = {}
    for file_path in files:
        if file_path.name.startswith("~$"):
            continue
        df = read_table_auto(file_path)
        df = auto_parse_date_columns(df)
        info_tables[file_path.stem] = df
    return info_tables


def get_datetime_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_status_code(value) -> str:
    if pd.isna(value):
        return "<NA>"
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
    except (TypeError, ValueError):
        pass
    return str(value)


def status_key(value, status_map: dict[int, dict[str, str]]) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    try:
        num = float(value)
        if num.is_integer():
            code = int(num)
            return status_map.get(code, {"key": f"STATUS_{code}"}).get("key", f"STATUS_{code}")
    except (TypeError, ValueError):
        pass
    return str(value)


def status_label(value, status_map: dict[int, dict[str, str]]) -> str:
    if pd.isna(value):
        return "<NA>"
    try:
        num = float(value)
        if num.is_integer():
            code = int(num)
            return status_map.get(code, {"label": f"status_{code}"}).get("label", f"status_{code}")
    except (TypeError, ValueError):
        pass
    return str(value)


def load_status_map(script_dir: Path) -> dict[int, dict[str, str]]:
    status_map = dict(DEFAULT_STATUS_MAP)
    mapping_path = script_dir / "status_mapping.csv"
    if not mapping_path.exists():
        return status_map

    try:
        mapping_df = pd.read_csv(mapping_path)
        if {"code", "key", "label"}.issubset(mapping_df.columns):
            for _, row in mapping_df[["code", "key", "label"]].dropna().iterrows():
                code = int(float(row["code"]))
                key = str(row["key"]).strip()
                label = str(row["label"]).strip()
                if key and label:
                    status_map[code] = {"key": key, "label": label}
    except Exception:
        return status_map

    return status_map


def normalize_lookup_key(value):
    if pd.isna(value):
        return None
    try:
        num = float(value)
        if num.is_integer():
            return int(num)
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_lookup_map(df: pd.DataFrame, key_col: str, value_col: str) -> dict:
    if key_col not in df.columns or value_col not in df.columns:
        return {}
    out: dict = {}
    for _, row in df[[key_col, value_col]].dropna().iterrows():
        key = normalize_lookup_key(row[key_col])
        value = str(row[value_col]).strip()
        if key is None or not value:
            continue
        out[key] = value
    return out


def map_series_with_lookup(series: pd.Series, lookup: dict) -> pd.Series:
    return series.map(lambda v: lookup.get(normalize_lookup_key(v)))


def enrich_with_info_tables(df: pd.DataFrame, info_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()

    location_tbl = info_tables.get("iso_location_202602161022")
    observation_tbl = info_tables.get("iso_observation_202602161023")
    product_status_tbl = info_tables.get("iso_product_status_202602161023")
    msg_log_type_tbl = info_tables.get("msg_log_type_202602161025")
    link_sr_log_type_tbl = info_tables.get("iso_link_sr_log_type_202602161025")
    link_sr_chairlift_tbl = info_tables.get("iso_link_sr_chairlift_202602161024")

    if location_tbl is not None and "Location_ID" in out.columns:
        location_map = build_lookup_map(location_tbl, "Location_ID", "Title")
        out["location_title"] = map_series_with_lookup(out["Location_ID"], location_map)

    if observation_tbl is not None:
        obs_map = build_lookup_map(observation_tbl, "Observation_ID", "Title")
        if "ProblemObservation_ID" in out.columns:
            out["problem_category_title"] = map_series_with_lookup(out["ProblemObservation_ID"], obs_map)
        if "InternalObservation_ID" in out.columns:
            out["internal_observation_title"] = map_series_with_lookup(out["InternalObservation_ID"], obs_map)
        if "ExternalObservation_ID" in out.columns:
            out["external_observation_title"] = map_series_with_lookup(out["ExternalObservation_ID"], obs_map)

    if product_status_tbl is not None and "ProductStatus_ID" in out.columns:
        product_status_map = build_lookup_map(product_status_tbl, "ProductStatus_ID", "Title")
        out["product_status_title"] = map_series_with_lookup(out["ProductStatus_ID"], product_status_map)

    if (
        msg_log_type_tbl is not None
        and link_sr_log_type_tbl is not None
        and "SR_ID" in out.columns
        and {"SR_ID", "LogType_ID"}.issubset(link_sr_log_type_tbl.columns)
    ):
        log_type_map = build_lookup_map(msg_log_type_tbl, "LogType_ID", "Title")
        link = link_sr_log_type_tbl[["SR_ID", "LogType_ID"]].dropna().copy()
        link["log_type_title"] = map_series_with_lookup(link["LogType_ID"], log_type_map)
        link = link.dropna(subset=["log_type_title"])
        if not link.empty:
            grouped = (
                link.groupby("SR_ID")["log_type_title"]
                .agg(lambda x: ", ".join(sorted(set(x.astype(str)))))
            )
            out["log_types"] = out["SR_ID"].map(grouped)

    if (
        link_sr_chairlift_tbl is not None
        and "SR_ID" in out.columns
        and {"SR_ID", "ChairLift_ID"}.issubset(link_sr_chairlift_tbl.columns)
    ):
        link_chair = link_sr_chairlift_tbl[["SR_ID", "ChairLift_ID"]].dropna().copy()
        if not link_chair.empty:
            grouped_chair = (
                link_chair.groupby("SR_ID")["ChairLift_ID"]
                .agg(lambda x: ", ".join(sorted(set(x.astype(str)))))
            )
            out["chairlift_ids"] = out["SR_ID"].map(grouped_chair)

    return out


def apply_filters(
    df: pd.DataFrame,
    date_col: str | None,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
    status_values: list,
    active_values: list,
    selected_categories: dict[str, list],
    selected_numeric_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = df.copy()

    if date_col and date_col in out.columns:
        if date_from is not None:
            out = out[out[date_col] >= pd.Timestamp(date_from)]
        if date_to is not None:
            out = out[out[date_col] <= pd.Timestamp(date_to)]

    if "StatusFlag_ID" in out.columns and status_values:
        out = out[out["StatusFlag_ID"].isin(status_values)]

    if "IsActive" in out.columns and active_values:
        out = out[out["IsActive"].isin(active_values)]

    for col, vals in selected_categories.items():
        if col in out.columns and vals:
            out = out[out[col].isin(vals)]

    for col, (vmin, vmax) in selected_numeric_ranges.items():
        if col in out.columns:
            out = out[out[col].between(vmin, vmax, inclusive="both")]

    return out


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    script_dir = Path(__file__).resolve().parent
    build_ts = Path(__file__).stat().st_mtime
    st.caption(f"Build: {pd.to_datetime(build_ts, unit='s')}")
    datasets = load_datasets(str(script_dir))
    info_tables = load_info_datasheets(str(script_dir))
    status_map = load_status_map(script_dir)

    if not datasets:
        st.error("Geen CSV/Excel-bestanden gevonden in dezelfde map als dit script.")
        st.stop()

    st.sidebar.header("Filters")
    st.sidebar.caption(f"Status map rows: {len(status_map)}")
    dataset_names = list(datasets.keys())
    selected_dataset = st.sidebar.selectbox("Dataset", dataset_names)

    df = enrich_with_info_tables(datasets[selected_dataset], info_tables)

    datetime_cols = get_datetime_cols(df)
    date_col = st.sidebar.selectbox(
        "Datumkolom",
        options=["(geen)"] + datetime_cols,
        index=0,
    )
    if date_col == "(geen)":
        date_col = None

    date_from = None
    date_to = None
    if date_col:
        min_dt = df[date_col].min()
        max_dt = df[date_col].max()
        if pd.notna(min_dt) and pd.notna(max_dt):
            date_from, date_to = st.sidebar.date_input(
                "Datumbereik",
                value=(min_dt.date(), max_dt.date()),
            )

    status_values: list = []
    active_values: list = []
    selected_categories: dict[str, list] = {}
    selected_numeric_ranges: dict[str, tuple[float, float]] = {}

    filtered = apply_filters(
        df=df,
        date_col=date_col,
        date_from=pd.Timestamp(date_from) if date_from is not None else None,
        date_to=pd.Timestamp(date_to) if date_to is not None else None,
        status_values=status_values,
        active_values=active_values,
        selected_categories=selected_categories,
        selected_numeric_ranges=selected_numeric_ranges,
    )

    st.sidebar.subheader("Gevraagde filters")
    col_map = {
        "id": first_existing_column(df, ["id", "SR_ID", "SignatureServiceReport_ID", "Signature_ID"]),
        "report_code": first_existing_column(df, ["report_code", "SRCode"]),
        "stairlift_id": first_existing_column(df, ["stairlift_id", "location_title", "Location_ID", "chairlift_ids"]),
        "status": first_existing_column(df, ["status", "StatusFlag_ID", "ProductStatus_ID"]),
        "problem_category": first_existing_column(df, ["problem_category", "problem_category_title", "ProblemObservation_ID"]),
        "internal_observation": first_existing_column(df, ["internal_observation", "internal_observation_title", "InternalObservation_ID"]),
        "malfunction_description": first_existing_column(df, ["malfunction_description", "DescriptionMalfunction"]),
        "malfunction_cause": first_existing_column(df, ["malfunction_cause", "CauseMalfunction"]),
        "solution": first_existing_column(df, ["solution", "Solution"]),
        "track_trace_number": first_existing_column(df, ["track_trace_number", "TrackTraceNumber"]),
        "repair_time_minutes": first_existing_column(df, ["repair_time_minutes", "RepairTime"]),
        "finance_comment": first_existing_column(df, ["finance_comment", "FinanceComment"]),
        "follow_up_reason": first_existing_column(df, ["follow_up_reason", "ReasonNoFollowUpAction"]),
        "external_observation": first_existing_column(df, ["external_observation", "external_observation_title", "ExternalObservation_ID"]),
        "product_status": first_existing_column(df, ["product_status", "product_status_title", "ProductStatus_ID"]),
        "log_types": first_existing_column(df, ["log_types"]),
        "created_at": first_existing_column(df, ["created_at", "CreatedAt"]),
        "updated_at": first_existing_column(df, ["updated_at", "UpdatedAt"]),
        "is_active": first_existing_column(df, ["is_active", "IsActive"]),
    }
    display_name_map = {
        real_col: friendly_name for friendly_name, real_col in col_map.items() if real_col
    }

    text_filters = {
        "id": st.sidebar.text_input("id bevat"),
        "report_code": st.sidebar.text_input("report_code bevat"),
        "track_trace_number": st.sidebar.text_input("track_trace_number bevat"),
        "malfunction_description": st.sidebar.text_input("malfunction_description bevat"),
        "malfunction_cause": st.sidebar.text_input("malfunction_cause bevat"),
        "solution": st.sidebar.text_input("solution bevat"),
    }

    category_filter_keys = [
        "stairlift_id",
        "status",
        "problem_category",
        "internal_observation",
        "external_observation",
        "product_status",
        "log_types",
        "finance_comment",
        "follow_up_reason",
        "is_active",
    ]
    selected_requested_categories: dict[str, list] = {}
    selected_status_labels: list[str] = []
    for key in category_filter_keys:
        real_col = col_map[key]
        if not real_col:
            continue
        if key == "status":
            status_options = sorted(
                df[real_col].dropna().map(lambda v: status_label(v, status_map)).unique().tolist()
            )
            selected_status_labels = st.sidebar.multiselect(key, options=status_options)
            continue
        options = sorted(df[real_col].dropna().astype(str).unique().tolist())
        chosen = st.sidebar.multiselect(key, options=options)
        if chosen:
            selected_requested_categories[real_col] = chosen

    repair_time_range = None
    repair_col = col_map["repair_time_minutes"]
    if repair_col and pd.api.types.is_numeric_dtype(df[repair_col]):
        s = df[repair_col].dropna()
        if not s.empty and float(s.min()) != float(s.max()):
            rmin, rmax = float(s.min()), float(s.max())
            repair_time_range = st.sidebar.slider(
                "repair_time_minutes",
                min_value=rmin,
                max_value=rmax,
                value=(rmin, rmax),
            )

    created_range = None
    created_col = col_map["created_at"]
    if created_col and pd.api.types.is_datetime64_any_dtype(df[created_col]):
        cmin = df[created_col].min()
        cmax = df[created_col].max()
        if pd.notna(cmin) and pd.notna(cmax):
            created_range = st.sidebar.date_input(
                "created_at",
                value=(cmin.date(), cmax.date()),
                key="created_at_range",
            )

    updated_range = None
    updated_col = col_map["updated_at"]
    if updated_col and pd.api.types.is_datetime64_any_dtype(df[updated_col]):
        umin = df[updated_col].min()
        umax = df[updated_col].max()
        if pd.notna(umin) and pd.notna(umax):
            updated_range = st.sidebar.date_input(
                "updated_at",
                value=(umin.date(), umax.date()),
                key="updated_at_range",
            )

    for key, query in text_filters.items():
        real_col = col_map[key]
        if real_col and query.strip():
            filtered = filtered[
                filtered[real_col]
                .astype(str)
                .str.contains(query.strip(), case=False, na=False)
            ]

    for real_col, chosen in selected_requested_categories.items():
        filtered = filtered[filtered[real_col].astype(str).isin(chosen)]

    status_col = col_map.get("status")
    if status_col and selected_status_labels:
        filtered = filtered[
            filtered[status_col].map(lambda v: status_label(v, status_map)).isin(selected_status_labels)
        ]

    if repair_col and repair_time_range:
        filtered = filtered[
            filtered[repair_col].between(repair_time_range[0], repair_time_range[1], inclusive="both")
        ]

    if created_col and created_range:
        c_from, c_to = created_range
        filtered = filtered[
            (filtered[created_col] >= pd.Timestamp(c_from))
            & (filtered[created_col] <= pd.Timestamp(c_to))
        ]

    if updated_col and updated_range:
        u_from, u_to = updated_range
        filtered = filtered[
            (filtered[updated_col] >= pd.Timestamp(u_from))
            & (filtered[updated_col] <= pd.Timestamp(u_to))
        ]

    tab1, tab2, tab3, tab4 = st.tabs(["Overzicht", "Analyse", "Data", "Duidingen"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rijen (gefilterd)", f"{len(filtered):,}".replace(",", "."))
        c2.metric("Kolommen", len(filtered.columns))
        c3.metric("Duplicaten", int(filtered.duplicated().sum()))

        total_cells = int(filtered.shape[0] * filtered.shape[1]) if not filtered.empty else 0
        missing_cells = int(filtered.isna().sum().sum()) if not filtered.empty else 0
        missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0
        c4.metric("Missend %", f"{missing_pct:.1f}%")

        st.subheader("Missende waarden per kolom")
        miss = pd.DataFrame({
            "missing_count": filtered.isna().sum(),
            "missing_pct": filtered.isna().mean() * 100,
        }).sort_values(["missing_pct", "missing_count"], ascending=False)
        miss = miss.rename(index=display_name_map)
        st.dataframe(miss, use_container_width=True, height=320)

        status_col = col_map.get("status")
        if status_col and status_col in df.columns:
            st.subheader("Beschikbare statussen")
            status_counts = df[status_col].value_counts(dropna=False).rename_axis("raw").reset_index(name="aantal")
            status_counts["status"] = status_counts["raw"].map(lambda v: status_label(v, status_map))
            status_counts["status_key"] = status_counts["raw"].map(lambda v: status_key(v, status_map))
            status_counts["code"] = status_counts["raw"].map(normalize_status_code)
            status_counts = status_counts[["status", "status_key", "code", "aantal"]]
            st.dataframe(status_counts, use_container_width=True, height=240)

    with tab2:
        st.subheader("Verdeling op categorische kolom")
        cat_candidates = [
            c for c in filtered.columns
            if (pd.api.types.is_object_dtype(filtered[c]) or pd.api.types.is_categorical_dtype(filtered[c]))
            and filtered[c].nunique(dropna=True) <= 50
        ]
        if cat_candidates:
            chosen_cat = st.selectbox(
                "Kies kolom",
                cat_candidates,
                format_func=lambda c: display_name_map.get(c, c),
            )
            vc = filtered[chosen_cat].fillna("<NA>").astype(str).value_counts().head(20)
            st.bar_chart(vc)
        else:
            st.info("Geen geschikte categorische kolommen met lage cardinaliteit.")

        status_col = col_map.get("status")
        if status_col and status_col in filtered.columns:
            st.subheader("Verdeling op statussoort")
            status_dist_filtered = (
                filtered[status_col]
                .map(lambda v: status_label(v, status_map))
                .value_counts(dropna=False)
            )
            status_dist_full = (
                df[status_col]
                .map(lambda v: status_label(v, status_map))
                .value_counts(dropna=False)
            )

            c_full, c_filtered = st.columns(2)
            c_full.markdown("**Volledige dataset**")
            c_full.dataframe(status_dist_full.rename("count").reset_index(), use_container_width=True, height=260)
            c_filtered.markdown("**Gefilterde dataset**")
            c_filtered.dataframe(status_dist_filtered.rename("count").reset_index(), use_container_width=True, height=260)
            st.markdown("**Grafiek statusverdeling (volledige dataset)**")
            st.bar_chart(status_dist_full)
            st.markdown("**Grafiek statusverdeling (gefilterde dataset)**")
            st.bar_chart(status_dist_filtered)

        st.subheader("Verdeling op numerieke kolom")
        num_candidates = [
            c for c in filtered.columns
            if pd.api.types.is_numeric_dtype(filtered[c]) and c != status_col
        ]
        if num_candidates:
            chosen_num = st.selectbox(
                "Kies numerieke kolom",
                num_candidates,
                format_func=lambda c: display_name_map.get(c, c),
            )
            st.line_chart(filtered[chosen_num].dropna().reset_index(drop=True))
            st.dataframe(filtered[chosen_num].describe().to_frame("stats"), use_container_width=True)
        else:
            st.info("Geen numerieke kolommen beschikbaar.")

    with tab3:
        st.subheader("Datapreview")
        display_df = filtered.rename(columns=display_name_map)
        st.dataframe(display_df, use_container_width=True, height=480)
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download gefilterde data (CSV)",
            data=csv_bytes,
            file_name=f"{selected_dataset}_filtered.csv",
            mime="text/csv",
        )

    with tab4:
        if info_tables:
            st.subheader("Info datasheets geladen")
            info_rows = [
                {"bestand": name, "rows": len(tbl), "kolommen": len(tbl.columns)}
                for name, tbl in info_tables.items()
            ]
            st.dataframe(pd.DataFrame(info_rows), use_container_width=True, height=220)

        st.write("PDF met duidingen gevonden:")
        pdf_files = sorted(script_dir.glob("*.pdf"))
        if pdf_files:
            for pdf in pdf_files:
                st.code(str(pdf))
            st.info(
                "Deze app kan de PDF in deze omgeving niet automatisch parsen. "
                "Als je de mapping als CSV/XLSX toevoegt (bijv. kolommen: veld, duiding), "
                "kan ik die direct in deze tab opnemen."
            )
        else:
            st.info("Geen PDF gevonden in de map.")

        mapping_files = sorted(script_dir.glob("*mapping*.csv")) + sorted(script_dir.glob("*mapping*.xlsx"))
        if mapping_files:
            st.subheader("Beschikbare mapping bestanden")
            for mf in mapping_files:
                st.write(f"- {mf.name}")

    st.caption('Run lokaal met: streamlit run "Startanalyse 2 backend.py"')


if __name__ == "__main__":
    main()
