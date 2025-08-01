#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from datetime import datetime, date, time as dtime, timedelta
import glob
import re
import csv

# 配置
RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
WINDOW_MINUTES = 30
TOLERANCE_MISSING = 5  # 每个 window 允许少量缺失秒数
EXPECTED_TOTAL_SECONDS = int((datetime.combine(date.today(), RTH_END) - datetime.combine(date.today(), RTH_START)).total_seconds())  # 23400

def get_windows_for_day(d: date):
    start = datetime.combine(d, RTH_START)
    end = datetime.combine(d, RTH_END)
    windows = []
    cursor = start
    while cursor < end:
        next_end = cursor + timedelta(minutes=WINDOW_MINUTES)
        windows.append((cursor, min(next_end, end)))
        cursor = next_end
    return windows

def window_is_complete(df_times, window_start, window_end, tol=TOLERANCE_MISSING):
    mask = (df_times >= window_start) & (df_times < window_end)
    sub = df_times[mask]
    if sub.empty:
        return False
    expected = int((window_end - window_start).total_seconds())
    if len(sub) + tol < expected:
        return False
    if sub.min() - window_start > timedelta(seconds=5):
        return False
    if window_end - sub.max() > timedelta(seconds=5):
        return False
    return True

def analyze_folder(data_dir: str):
    pattern = os.path.join(data_dir, "SPX_1s_*.csv")
    files = sorted(glob.glob(pattern))
    report = []
    for fp in files:
        m = re.search(r"SPX_1s_(\d{4}-\d{2}-\d{2})\.csv", fp)
        if not m:
            continue
        day_str = m.group(1)
        try:
            d = datetime.fromisoformat(day_str).date()
        except Exception:
            continue
        row = {"date": d, "file": os.path.basename(fp)}
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            row.update({
                "status": "read_error",
                "error": str(e),
            })
            report.append(row)
            continue

        if "date" not in df.columns:
            row.update({
                "status": "no_date_column",
                "rows": len(df),
            })
            report.append(row)
            continue

        df["date"] = pd.to_datetime(df["date"])
        # 统一时区：如果有 tz 转成 UTC 再去掉
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        df_times = df["date"]

        actual_rows = len(df)
        expected_rows = int((datetime.combine(d, RTH_END) - datetime.combine(d, RTH_START)).total_seconds())
        missing_windows = []
        for ws, we in get_windows_for_day(d):
            if not window_is_complete(df_times, ws, we):
                missing_windows.append(f"{ws.time().strftime('%H:%M:%S')}-{we.time().strftime('%H:%M:%S')}")

        row.update({
            "status": "ok",
            "rows": actual_rows,
            "expected_rows": expected_rows,
            "missing_window_count": len(missing_windows),
            "missing_windows": ";".join(missing_windows),
        })
        report.append(row)

    return sorted(report, key=lambda x: x["date"])

def main():
    p = argparse.ArgumentParser(description="SPX 1s 数据 completeness 报告")
    p.add_argument("--dir", "-d", required=True, help="存放 SPX_1s_YYYY-MM-DD.csv 的目录")
    p.add_argument("--out", "-o", default="completeness_summary.csv", help="输出报告 CSV 文件名")
    args = p.parse_args()

    report = analyze_folder(args.dir)
    if not report:
        print("没有发现符合的文件，检查目录和命名。")
        return

    # 打印简要
    print(f"{'date':10}  {'rows':>7}  {'expect':>7}  {'missing_wins':>14}")
    for r in report:
        dt = r["date"].isoformat()
        if r.get("status") != "ok":
            print(f"{dt}  ERROR {r.get('status')} {r.get('error','')}")
        else:
            print(f"{dt}  {r['rows']:7,}  {r['expected_rows']:7,}  {r['missing_window_count']:14} {'(some missing)' if r['missing_window_count'] else '(complete)'}")

    # 写 CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "file", "status", "rows", "expected_rows", "missing_window_count", "missing_windows", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report:
            writer.writerow({
                "date": r.get("date"),
                "file": r.get("file"),
                "status": r.get("status"),
                "rows": r.get("rows", ""),
                "expected_rows": r.get("expected_rows", ""),
                "missing_window_count": r.get("missing_window_count", ""),
                "missing_windows": r.get("missing_windows", ""),
                "error": r.get("error", ""),
            })
    print(f"\n保存到报告: {args.out}")

if __name__ == "__main__":
    main()
