from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv_auto(csv_file: Path) -> pd.DataFrame:
    """Read CSV with delimiter auto-detection."""
    return pd.read_csv(csv_file, sep=None, engine="python")


def auto_parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse likely date/time columns in-place.
    Uses column name hints and conversion success ratio.
    """
    date_hints = ("date", "datum", "tijd", "time")
    candidate_cols = [
        col for col in df.columns if any(hint in col.lower() for hint in date_hints)
    ]

    for col in candidate_cols:
        if not pd.api.types.is_object_dtype(df[col]):
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted_default = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
            converted_dayfirst = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        success_default = converted_default.notna().sum() / non_null
        success_dayfirst = converted_dayfirst.notna().sum() / non_null

        best_converted = converted_default
        best_success = success_default
        if success_dayfirst > success_default:
            best_converted = converted_dayfirst
            best_success = success_dayfirst

        if best_success >= 0.6:
            df[col] = best_converted

    return df


def combine_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all loaded datasets into one DataFrame with source info."""
    frames: list[pd.DataFrame] = []
    for name, df in datasets.items():
        frame = df.copy()
        frame["bron_bestand"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_column_type_counts(df: pd.DataFrame, output_path: Path) -> pd.Series:
    """Create and save a bar chart with counts per column dtype."""
    dtype_counts = df.dtypes.astype(str).value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(dtype_counts.index, dtype_counts.values)
    plt.title("Aantal kolommen per kolomsoort (dtype)")
    plt.xlabel("Kolomsoort")
    plt.ylabel("Aantal kolommen")
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = int(bar.get_height())
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return dtype_counts


def summarize_dataset(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    """Build a compact quality summary row for one dataset."""
    row_count, col_count = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(row_count * col_count) if row_count and col_count else 0
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0
    numeric_cols = int(len(df.select_dtypes(include=[np.number]).columns))
    date_cols = int(len(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns))
    object_cols = int(len(df.select_dtypes(include=["object"]).columns))

    return {
        "dataset": name,
        "rows": row_count,
        "columns": col_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": (duplicate_rows / row_count * 100) if row_count else 0.0,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "numeric_columns": numeric_cols,
        "datetime_columns": date_cols,
        "object_columns": object_cols,
    }


def create_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness table per column."""
    missing = df.isna().sum().rename("missing_count")
    pct = (df.isna().mean() * 100).rename("missing_pct")
    out = pd.concat([missing, pct], axis=1).sort_values(
        by=["missing_pct", "missing_count"], ascending=False
    )
    return out


def create_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric describe table with useful percentiles."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def create_categorical_top_values(
    df: pd.DataFrame, max_cardinality: int = 20, top_n: int = 5
) -> pd.DataFrame:
    """
    Create top value counts for low-cardinality categorical columns.
    Keeps output compact and directly useful for exploration.
    """
    rows: list[dict[str, str | int | float]] = []
    object_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
    for col in object_cols:
        unique_values = df[col].nunique(dropna=True)
        if unique_values == 0 or unique_values > max_cardinality:
            continue

        counts = df[col].value_counts(dropna=False).head(top_n)
        for value, count in counts.items():
            if pd.isna(value):
                value_str = "<NA>"
            else:
                value_str = str(value)
            rows.append(
                {
                    "column": col,
                    "unique_values": int(unique_values),
                    "value": value_str,
                    "count": int(count),
                    "pct": float(count / len(df) * 100) if len(df) else 0.0,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["column", "count"], ascending=[True, False])


def plot_missingness_top(
    missingness: pd.DataFrame, output_path: Path, top_n: int = 12
) -> None:
    """Plot top columns with missing values."""
    plot_df = missingness[missingness["missing_count"] > 0].head(top_n)
    if plot_df.empty:
        return

    plt.figure(figsize=(11, 5))
    bars = plt.bar(plot_df.index.astype(str), plot_df["missing_pct"].values)
    plt.title("Top kolommen met missende waarden (%)")
    plt.xlabel("Kolom")
    plt.ylabel("Missend (%)")
    plt.xticks(rotation=45, ha="right")

    for bar, value in zip(bars, plot_df["missing_pct"].values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_monthly_volume(df: pd.DataFrame, dataset_name: str, output_folder: Path) -> None:
    """Plot monthly record counts for each datetime column."""
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in dt_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        monthly = series.dt.to_period("M").value_counts().sort_index()
        if monthly.empty:
            continue

        x_values = monthly.index.astype(str)
        y_values = monthly.values

        plt.figure(figsize=(12, 4))
        plt.plot(x_values, y_values, marker="o")
        plt.title(f"Maandvolume op {col} ({dataset_name})")
        plt.xlabel("Maand")
        plt.ylabel("Aantal records")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = output_folder / f"eda_{dataset_name}_{col}_monthly_volume.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def load_csvs_from_same_folder() -> dict[str, pd.DataFrame]:
    """Load all CSV files from this script's folder into a dict."""
    folder = Path(__file__).resolve().parent
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    datasets: dict[str, pd.DataFrame] = {}
    for csv_file in csv_files:
        df = read_csv_auto(csv_file)
        df = auto_parse_date_columns(df)
        datasets[csv_file.stem] = df
        print(f"Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")

    return datasets


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    output_folder = folder / "eda_output"
    output_folder.mkdir(exist_ok=True)

    data = load_csvs_from_same_folder()
    print(f"\nLoaded {len(data)} datasets in total.")
    combined_data = combine_datasets(data)
    print(
        f"Combined dataset: {len(combined_data)} rows, "
        f"{len(combined_data.columns)} columns"
    )

    plot_path = folder / "kolomsoorten_overzicht.png"
    dtype_counts = plot_column_type_counts(combined_data, plot_path)
    print("\nKolomsoorten en aantallen:")
    print(dtype_counts.to_string())
    print(f"\nVisualisatie opgeslagen als: {plot_path}")

    summary_rows: list[dict[str, float | int | str]] = []
    for dataset_name, df in data.items():
        summary_rows.append(summarize_dataset(dataset_name, df))

        missingness = create_missingness_table(df)
        missingness.to_csv(
            output_folder / f"eda_{dataset_name}_missingness.csv", index=True
        )
        plot_missingness_top(
            missingness, output_folder / f"eda_{dataset_name}_missingness_top.png"
        )

        numeric_summary = create_numeric_summary(df)
        if not numeric_summary.empty:
            numeric_summary.to_csv(
                output_folder / f"eda_{dataset_name}_numeric_summary.csv", index=True
            )

        categorical_summary = create_categorical_top_values(df)
        if not categorical_summary.empty:
            categorical_summary.to_csv(
                output_folder / f"eda_{dataset_name}_categorical_top_values.csv",
                index=False,
            )

        plot_monthly_volume(df, dataset_name, output_folder)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="rows", ascending=False)
    summary_path = output_folder / "eda_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    combined_missingness = create_missingness_table(combined_data)
    combined_missingness.to_csv(
        output_folder / "eda_combined_missingness.csv", index=True
    )
    plot_missingness_top(
        combined_missingness, output_folder / "eda_combined_missingness_top.png"
    )

    print("\nEDA-output opgeslagen in:")
    print(output_folder)
    print("\nDataset-samenvatting:")
    print(summary_df.to_string(index=False))
