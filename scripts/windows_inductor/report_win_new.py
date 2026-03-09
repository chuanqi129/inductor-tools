#!/usr/bin/env python3
"""Parse inductor_log CSV results and emit summaries.

Usage:
  python scripts/parse_inductor_log.py --root inductor_log
  python scripts/parse_inductor_log.py --details
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


ACCURACY_SUFFIX = "_accuracy.csv"
PERF_SUFFIX = "_performance.csv"
PASS_VALUES_DEFAULT = {"pass", "pass_due_to_skip"}


def find_log_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(ACCURACY_SUFFIX) or filename.endswith(PERF_SUFFIX):
                yield os.path.join(dirpath, filename)


def parse_accuracy_file(path: str, pass_values: set) -> Dict[str, Any]:
    total = 0
    passed = 0
    status_counts: Dict[str, int] = {}
    rows: List[Dict[str, str]] = []

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            status = (row.get("accuracy") or "").strip().lower()
            total += 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in pass_values:
                passed += 1

    pass_rate = (passed / total) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "status_counts": status_counts,
        "rows": rows,
    }


def parse_performance_file(path: str) -> Dict[str, Any]:
    speedups: List[float] = []
    rows: List[Dict[str, str]] = []
    excluded = 0

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            value = (row.get("speedup") or "").strip()
            if not value:
                excluded += 1
                continue
            try:
                speedup = float(value)
            except ValueError:
                excluded += 1
                continue
            if speedup <= 0:
                excluded += 1
                continue
            speedups.append(speedup)

    geo_mean = geometric_mean(speedups)
    avg = (sum(speedups) / len(speedups)) if speedups else 0.0
    return {
        "count": len(speedups),
        "excluded": excluded,
        "geo_mean": geo_mean,
        "avg": avg,
        "speedups": speedups,
        "rows": rows,
    }


def geometric_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def suite_precision_from_path(root: str, path: str) -> Tuple[str, str]:
    rel_path = os.path.relpath(path, root)
    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        return "unknown", "unknown"
    return parts[0], parts[1]


def format_accuracy_summary(data: Dict[str, Any]) -> str:
    total = data["total"]
    passed = data["passed"]
    rate = data["pass_rate"] * 100
    return f"pass rate {rate:.1f}% ({passed}/{total})"


def format_performance_summary(data: Dict[str, Any]) -> str:
    return f"{data['geo_mean']:.2f}x"


def build_report(root: str, pass_values: set) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    for path in sorted(find_log_files(root)):
        suite, precision = suite_precision_from_path(root, path)
        report.setdefault(suite, {}).setdefault(
            precision,
            {
                "accuracy": [],
                "performance": [],
            },
        )

        if path.endswith(ACCURACY_SUFFIX):
            report[suite][precision]["accuracy"].append(
                {
                    "path": path,
                    "summary": parse_accuracy_file(path, pass_values),
                }
            )
        elif path.endswith(PERF_SUFFIX):
            report[suite][precision]["performance"].append(
                {
                    "path": path,
                    "summary": parse_performance_file(path),
                }
            )

    return report


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for suite, suite_data in report.items():
        summary[suite] = {}
        for precision, data in suite_data.items():
            acc_entries = data.get("accuracy", [])
            perf_entries = data.get("performance", [])

            acc_total = 0
            acc_passed = 0
            acc_status_counts: Dict[str, int] = {}
            for entry in acc_entries:
                acc = entry["summary"]
                acc_total += acc["total"]
                acc_passed += acc["passed"]
                for status, count in acc["status_counts"].items():
                    acc_status_counts[status] = acc_status_counts.get(status, 0) + count

            acc_rate = (acc_passed / acc_total) if acc_total else 0.0

            perf_speedups: List[float] = []
            perf_excluded = 0
            for entry in perf_entries:
                perf = entry["summary"]
                perf_speedups.extend(perf["speedups"])
                perf_excluded += perf["excluded"]

            perf_geo_mean = geometric_mean(perf_speedups)

            summary[suite][precision] = {
                "accuracy": {
                    "total": acc_total,
                    "passed": acc_passed,
                    "pass_rate": acc_rate,
                    "status_counts": acc_status_counts,
                },
                "performance": {
                    "count": len(perf_speedups),
                    "excluded": perf_excluded,
                    "geo_mean": perf_geo_mean,
                },
            }

    return summary


def print_report(
    summary: Dict[str, Any],
    report: Dict[str, Any],
    details: bool,
    rows: bool,
) -> None:
    for suite, suite_data in summary.items():
        print(f"Suite: {suite}")
        for precision, data in suite_data.items():
            acc_summary = format_accuracy_summary(data["accuracy"])
            perf_summary = format_performance_summary(data["performance"])
            print(f"  Precision: {precision}")
            print(f"    Accuracy: {acc_summary}")
            print(f"    Performance: {perf_summary}")

            if details:
                acc_entries = report[suite][precision].get("accuracy", [])
                perf_entries = report[suite][precision].get("performance", [])
                for entry in acc_entries:
                    acc_detail = format_accuracy_summary(entry["summary"])
                    print(f"      {os.path.basename(entry['path'])}: {acc_detail}")
                for entry in perf_entries:
                    perf_detail = format_performance_summary(entry["summary"])
                    print(f"      {os.path.basename(entry['path'])}: {perf_detail}")

            if rows:
                acc_entries = report[suite][precision].get("accuracy", [])
                perf_entries = report[suite][precision].get("performance", [])
                for entry in acc_entries:
                    print(f"      Accuracy rows: {os.path.basename(entry['path'])}")
                    for row in entry["summary"]["rows"]:
                        name = row.get("name", "")
                        status = row.get("accuracy", "")
                        print(f"        {name}: {status}")
                for entry in perf_entries:
                    print(f"      Performance rows: {os.path.basename(entry['path'])}")
                    for row in entry["summary"]["rows"]:
                        name = row.get("name", "")
                        speedup = row.get("speedup", "")
                        abs_latency = row.get("abs_latency", "")
                        print(
                            f"        {name}: abs_latency={abs_latency}, speedup={speedup}"
                        )
        print("")


def build_excel_frames(
    summary: Dict[str, Any],
    report: Dict[str, Any],
    source_label: str = "",
) -> Tuple[Any, Any, Any, Any]:
    acc_summary_rows: List[Dict[str, Any]] = []
    perf_summary_rows: List[Dict[str, Any]] = []
    acc_detail_rows: List[Dict[str, Any]] = []
    perf_detail_rows: List[Dict[str, Any]] = []

    for suite, suite_data in summary.items():
        for precision, data in suite_data.items():
            acc = data["accuracy"]
            perf = data["performance"]
            acc_summary_rows.append(
                {
                    "source": source_label,
                    "suite": suite,
                    "precision": precision,
                    "pass_rate": acc["pass_rate"],
                    "passed": acc["passed"],
                    "total": acc["total"],
                    "status_counts": json.dumps(acc["status_counts"], sort_keys=True),
                }
            )
            perf_summary_rows.append(
                {
                    "source": source_label,
                    "suite": suite,
                    "precision": precision,
                    "geo_mean": perf["geo_mean"],
                    "count": perf["count"],
                    "excluded": perf["excluded"],
                }
            )

            acc_entries = report[suite][precision].get("accuracy", [])
            perf_entries = report[suite][precision].get("performance", [])

            for entry in acc_entries:
                filename = os.path.basename(entry["path"])
                for row in entry["summary"]["rows"]:
                    acc_detail_rows.append(
                        {
                            "source": source_label,
                            "suite": suite,
                            "precision": precision,
                            "dev": row.get("dev", ""),
                            "name": row.get("name", ""),
                            "batch_size": row.get("batch_size", ""),
                            "accuracy": row.get("accuracy", ""),
                        }
                    )

            for entry in perf_entries:
                filename = os.path.basename(entry["path"])
                for row in entry["summary"]["rows"]:
                    perf_detail_rows.append(
                        {
                            "source": source_label,
                            "suite": suite,
                            "precision": precision,
                            "dev": row.get("dev", ""),
                            "name": row.get("name", ""),
                            "batch_size": row.get("batch_size", ""),
                            "abs_latency": row.get("abs_latency", ""),
                            "speedup": row.get("speedup", ""),
                            "compilation_latency": row.get("compilation_latency", ""),
                        }
                    )

    # Create DataFrames
    acc_summary_df = pd.DataFrame(acc_summary_rows)
    perf_summary_df = pd.DataFrame(perf_summary_rows)
    acc_detail_df = pd.DataFrame(acc_detail_rows)
    perf_detail_df = pd.DataFrame(perf_detail_rows)
    
    # Convert numeric columns to proper types to avoid "numbers stored as text" warning
    if not acc_detail_df.empty and "batch_size" in acc_detail_df.columns:
        acc_detail_df["batch_size"] = pd.to_numeric(acc_detail_df["batch_size"], errors="coerce")
    
    if not perf_detail_df.empty:
        numeric_cols = ["batch_size", "abs_latency", "speedup", "compilation_latency"]
        for col in numeric_cols:
            if col in perf_detail_df.columns:
                perf_detail_df[col] = pd.to_numeric(perf_detail_df[col], errors="coerce")
    
    return (
        acc_summary_df,
        perf_summary_df,
        acc_detail_df,
        perf_detail_df,
    )


def auto_adjust_column_widths(worksheet: Any, dataframe: Any, center_format: Any) -> None:
    """Auto-adjust column widths based on content length and apply center alignment."""
    for idx, col in enumerate(dataframe.columns):
        # Get the maximum length of the column name and column values
        max_len = len(str(col))
        for value in dataframe[col].astype(str):
            max_len = max(max_len, len(value))
        # Set column width with some padding (max 50 to avoid extremely wide columns)
        adjusted_width = min(max_len + 2, 50)
        worksheet.set_column(idx, idx, adjusted_width, center_format)


def write_excel(
    path: str,
    summary: Dict[str, Any],
    report: Dict[str, Any],
    summary_ref: Dict[str, Any] = None,
    report_ref: Dict[str, Any] = None,
) -> None:
    if pd is None:
        raise RuntimeError("pandas is required for --excel output")

    acc_summary_df, perf_summary_df, acc_detail_df, perf_detail_df = build_excel_frames(
        summary,
        report,
        source_label="inductor_log",
    )

    if summary_ref and report_ref:
        (
            acc_summary_ref_df,
            perf_summary_ref_df,
            acc_detail_ref_df,
            perf_detail_ref_df,
        ) = build_excel_frames(
            summary_ref,
            report_ref,
            source_label="inductor_log_ref",
        )
        acc_summary_df = pd.concat(
            [acc_summary_df, acc_summary_ref_df], ignore_index=True
        )
        perf_summary_df = pd.concat(
            [perf_summary_df, perf_summary_ref_df], ignore_index=True
        )
        acc_detail_df = pd.concat([acc_detail_df, acc_detail_ref_df], ignore_index=True)
        perf_detail_df = pd.concat(
            [perf_detail_df, perf_detail_ref_df], ignore_index=True
        )

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        # Get the xlsxwriter workbook and create a format for center alignment
        workbook = writer.book
        center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        header_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'bold': True,
        })
        
        # Write data to sheets
        acc_summary_df.to_excel(writer, sheet_name="acc_summary", index=False)
        perf_summary_df.to_excel(writer, sheet_name="perf_summary", index=False)
        acc_detail_df.to_excel(writer, sheet_name="acc_details", index=False)
        perf_detail_df.to_excel(writer, sheet_name="perf_details", index=False)
        
        # Apply formatting and auto-adjust column widths for each sheet
        for sheet_name, df in [
            ("acc_summary", acc_summary_df),
            ("perf_summary", perf_summary_df),
            ("acc_details", acc_detail_df),
            ("perf_details", perf_detail_df),
        ]:
            worksheet = writer.sheets[sheet_name]
            
            # Format header row
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths with center alignment
            auto_adjust_column_widths(worksheet, df, center_format)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse inductor_log CSV results")
    parser.add_argument(
        "--root",
        type=str,
        default="inductor_log",
        help="Root directory containing benchmark suites",
    )
    parser.add_argument(
        "--root_ref",
        type=str,
        default="",
        help="Reference root directory for comparison (e.g., inductor_log_ref)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-file summaries in text output",
    )
    parser.add_argument(
        "--rows",
        action="store_true",
        help="Include per-model rows in text output",
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="",
        help="Write results to an Excel file (requires pandas)",
    )
    args = parser.parse_args()

    pass_values = PASS_VALUES_DEFAULT

    report = build_report(args.root, pass_values)
    summary = summarize_report(report)

    report_ref = None
    summary_ref = None
    if args.root_ref:
        report_ref = build_report(args.root_ref, pass_values)
        summary_ref = summarize_report(report_ref)

    if args.excel:
        write_excel(args.excel, summary, report, summary_ref, report_ref)

    print_report(summary, report, args.details, args.rows)

    if args.root_ref:
        print("\n" + "=" * 50)
        print(f"Reference Data from: {args.root_ref}")
        print("=" * 50 + "\n")
        print_report(summary_ref, report_ref, args.details, args.rows)


if __name__ == "__main__":
    main()
