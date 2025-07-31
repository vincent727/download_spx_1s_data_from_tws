#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_spx_1s_30d_paced_upgraded.py

升级版 SPX 1s 历史数据下载（含稳健 pacing 控制 + 162 退避 + debug logging）。

特性：
  * Pacer 类：统一控制“滑动窗口 + 最小间隔”请求节奏，避免 burst 触发 pacing violation。
  * 捕捉 IB Error 162（pacing violation），退避重试当前 window（指数退避但有限制）。
  * 30 分钟固定窗口下载 + 合并去重。保证丢失段可重试，不丢整个日。
  * 跳过周末与美股节假日（若安装 pandas_market_calendars）。
  * 文件存在时跳过（可根据 size threshold 另加判断逻辑手动扩展）。
  * 日志写入到 console 和 download.log，debug 模式下打印 recent request queue。
  * 参数可调：天数、IB 链接、节奏、安全边界、日志等级等。
"""
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta, date, time as dtime
from collections import deque

try:
    from ib_insync import IB, Contract, util
except ImportError:
    print("需要安装 ib_insync: pip install ib_insync", file=sys.stderr)
    raise

import pandas as pd

# 交易时间（美东）
RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)

BAR_SIZE = "1 secs"
WHAT_TO_SHOW = "TRADES"
USE_RTH = True
FORMAT_DATE = 1

# 默认输出文件夹
DEFAULT_OUT_DIR = "./spx_1s_data"

# 处理节假日
USE_MARKET_CAL = True
try:
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
except ImportError:
    USE_MARKET_CAL = False
    print("警告：未安装 pandas_market_calendars，节假日只跳过周末。安装以增强准确性: pip install pandas_market_calendars", file=sys.stderr)


class Pacer:
    def __init__(self, max_per_window: int = 45, window_sec: int = 600, min_interval: float | None = None):
        """
        控制历史数据请求节奏。
        - max_per_window: 滑动窗口内允许的最大请求数（保守设小于 IB 官方 60）
        - window_sec: 滑动窗口大小（秒），通常 600 秒（10 分钟）
        - min_interval: 两次请求之间的最小间隔（秒），若 None 则用 window_sec / max_per_window
        """
        self.max = max_per_window
        self.window = window_sec
        self.req_times = deque()
        self.min_interval = min_interval if min_interval is not None else (window_sec / max_per_window)
        self.last_req_time = 0.0

    def wait_for_slot(self, logger: logging.Logger, debug: bool = False):
        now = time.time()
        # 清理过期
        while self.req_times and now - self.req_times[0] > self.window:
            self.req_times.popleft()

        # 如果窗口满了，等最老的出窗
        if len(self.req_times) >= self.max:
            to_wait = self.window - (now - self.req_times[0]) + 0.05
            logger.debug(f"Pacer window full ({len(self.req_times)}/{self.max}), sleeping {to_wait:.2f}s")
            time.sleep(to_wait)
            now = time.time()
            while self.req_times and now - self.req_times[0] > self.window:
                self.req_times.popleft()

        # 强制最小间隔
        delta = now - self.last_req_time
        if delta < self.min_interval:
            extra = self.min_interval - delta
            logger.debug(f"Pacer enforcing min_interval, sleeping {extra:.2f}s")
            time.sleep(extra)
            now = time.time()

        # 记录此次请求
        self.last_req_time = now
        self.req_times.append(now)
        if debug:
            short_times = [datetime.fromtimestamp(t).strftime("%H:%M:%S") for t in list(self.req_times)]
            logger.debug(f"[DEBUG] Request issued at {datetime.fromtimestamp(now).strftime('%H:%M:%S')} | recent ({len(self.req_times)}): {short_times}")


def get_past_trading_days(n: int) -> list[date]:
    today = datetime.now().date()
    days = []
    lookback = 0
    while len(days) < n:
        cand = today - timedelta(days=lookback)
        lookback += 1
        if cand.weekday() >= 5:
            continue
        if USE_MARKET_CAL:
            try:
                schedule = nyse.schedule(start_date=cand, end_date=cand)
                if schedule.empty:
                    continue
            except Exception:
                # 如果 calendar 出错，fallback 只跳周末
                pass
        days.append(cand)
    return sorted(days)


def get_windows_for_day(d: date) -> list[tuple[datetime, datetime]]:
    start = datetime.combine(d, RTH_START)
    end = datetime.combine(d, RTH_END)
    windows = []
    cursor = start
    while cursor < end:
        next_end = cursor + timedelta(minutes=30)
        windows.append((cursor, min(next_end, end)))
        cursor = next_end
    return windows


def window_is_complete(df: pd.DataFrame, window_start: datetime, window_end: datetime, tolerance_missing: int = 5) -> bool:
    """
    判断某 30 分钟 window 在已有 df 里是否完整。
    自动处理 tz-aware / naive 的对齐。
    """
    if df.empty:
        return False

    df_times = pd.to_datetime(df["date"])
    # 构造与 df_times 相同 tz 语义的 window 边界
    if df_times.dt.tz is not None:
        tz = df_times.dt.tz
        ws = pd.Timestamp(window_start).tz_localize(tz)
        we = pd.Timestamp(window_end).tz_localize(tz)
    else:
        ws = pd.Timestamp(window_start)
        we = pd.Timestamp(window_end)

    mask = (df_times >= ws) & (df_times < we)
    sub = df_times[mask]
    if sub.empty:
        return False

    count = len(sub)
    expected = int((we - ws).total_seconds())
    if count + tolerance_missing < expected:
        return False

    # 边界覆盖：最早/最晚不能偏离太多
    if sub.min() - ws > timedelta(seconds=5):
        return False
    if we - sub.max() > timedelta(seconds=5):
        return False

    return True


def find_missing_windows_for_day(csv_path: str, d: date) -> list[tuple[datetime, datetime]]:
    windows = get_windows_for_day(d)
    if not os.path.exists(csv_path):
        return windows
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return windows
    if "date" not in df.columns:
        return windows
    df["date"] = pd.to_datetime(df["date"])
    missing = []
    for ws, we in windows:
        if not window_is_complete(df, ws, we):
            missing.append((ws, we))
    return missing


def build_spx_contract() -> Contract:
    c = Contract()
    c.secType = "IND"
    c.symbol = "SPX"
    c.exchange = "CBOE"
    c.currency = "USD"
    return c


def download_single_window(
    ib: IB,
    contract: Contract,
    window_start: datetime,
    window_end: datetime,
    pacer: Pacer,
    req_logger: logging.Logger,
    req_timestamps: deque,
    debug: bool,
    max_violation_backoff: int,
) -> pd.DataFrame | None:
    """下载指定 30 分钟 window（带 pacing + 162 退避）。"""
    end_str = window_end.strftime("%Y%m%d %H:%M:%S")
    duration_sec = int((window_end - window_start).total_seconds())
    pacing_violation_flag = {"hit": False}

    def error_handler(reqId, errorCode, errorString, *args):
        if errorCode == 162 and "pacing violation" in errorString.lower():
            pacing_violation_flag["hit"] = True
            req_logger.warning(f"Detected pacing violation (162) for window {window_start.time()}-{window_end.time()}: {errorString}")

    ib.errorEvent += error_handler
    attempt = 0
    backoff = max_violation_backoff
    last_req_time_local = None
    try:
        while True:
            # 节奏等待
            pacer.wait_for_slot(req_logger, debug=debug)
            now = datetime.now()
            # 维护外部滑动队列（供外部观察）
            req_timestamps.append(now)
            while req_timestamps and (now - req_timestamps[0]).total_seconds() > pacer.window:
                req_timestamps.popleft()

            # 发送请求
            pacing_violation_flag["hit"] = False
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr=f"{duration_sec} S",
                    barSizeSetting=BAR_SIZE,
                    whatToShow=WHAT_TO_SHOW,
                    useRTH=USE_RTH,
                    formatDate=FORMAT_DATE,
                    keepUpToDate=False,
                )
            except Exception as e:
                attempt += 1
                req_logger.error(f"Exception requesting window {window_start} (attempt {attempt}): {e}")
                if attempt > 5:
                    req_logger.error(f"Giving up window {window_start} after repeated exceptions.")
                    return None
                time.sleep(2 * attempt)
                continue

            last_req_time_local = datetime.now()

            if pacing_violation_flag["hit"]:
                # pacing violation: 不推进，指数退避
                req_logger.warning(f"Pacing violation on window {window_start}, backing off {backoff}s before retry.")
                time.sleep(backoff)
                backoff = min(int(backoff * 1.5), 1800)  # 上限 30 分钟
                attempt += 1
                if attempt > 8:
                    req_logger.error(f"Too many pacing violation retries for window {window_start}, giving up.")
                    return None
                continue

            if not bars:
                attempt += 1
                req_logger.warning(f"Empty data for window {window_start} (attempt {attempt}), retrying shortly.")
                if attempt > 3:
                    req_logger.error(f"Giving up empty window {window_start} after {attempt} attempts.")
                    return None
                time.sleep(1)
                continue

            df = util.df(bars)
            return df
    finally:
        ib.errorEvent -= error_handler


def process_day(
    ib: IB,
    contract: Contract,
    d: date,
    pacer: Pacer,
    req_timestamps: deque,
    out_dir: str,
    logger: logging.Logger,
    debug: bool,
    violation_backoff: int,
):
    csv_path = f"{out_dir}/SPX_1s_{d}.csv"
    missing_windows = find_missing_windows_for_day(csv_path, d)
    if not missing_windows:
        logger.info(f"{d} all windows complete, skipping.")
        return

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if "date" in existing_df.columns:
            existing_df["date"] = pd.to_datetime(existing_df["date"])
        else:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    collected_frames = []
    for ws, we in missing_windows:
        logger.info(f"{d} downloading window {ws.time()} - {we.time()}")
        df_win = download_single_window(
            ib,
            contract,
            ws,
            we,
            pacer,
            logger,
            req_timestamps,
            debug,
            violation_backoff,
        )
        if df_win is not None and not df_win.empty:
            collected_frames.append(df_win)
        else:
            logger.warning(f"Failed to fill window {ws} for day {d}")

    if not collected_frames and existing_df.empty:
        logger.warning(f"No data obtained for {d}, nothing to write.")
        return

    if existing_df.empty:
        combined = pd.concat(collected_frames, ignore_index=True)
    else:
        combined = pd.concat([existing_df] + collected_frames, ignore_index=True)

    combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    tmp_fn = f"{out_dir}/.tmp_SPX_1s_{d}.csv"
    final_fn = f"{out_dir}/SPX_1s_{d}.csv"
    combined.to_csv(tmp_fn, index=False)
    os.replace(tmp_fn, final_fn)
    logger.info(f"{d} saved {len(combined):,} rows.")


def parse_args():
    p = argparse.ArgumentParser(description="升级版下载 SPX 1s 数据，带 Pacer 节奏控制和 162 退避")
    p.add_argument("--days", "-n", type=int, default=30, help="回溯交易日数量（默认 30）")
    p.add_argument("--host", type=str, default="127.0.0.1", help="IB host")
    p.add_argument("--port", type=int, default=7497, help="IB port")
    p.add_argument("--clientId", type=int, default=1, help="IB clientId")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="输出目录")
    p.add_argument("--max-per-window", type=int, default=45, help="滑动窗口内最大请求数（保守于官方限额）")
    p.add_argument("--min-interval", type=float, default=None, help="每次请求最小间隔（秒），默认 window/max")
    p.add_argument("--violation-backoff", type=int, default=610, help="碰到 pacing violation 时退避秒数起始值")
    p.add_argument("--limit", type=int, default=None, help="最多处理多少天（用于调试）")
    p.add_argument("--log", type=str, default="download.log", help="日志文件路径")
    p.add_argument("--debug", action="store_true", help="启用 debug 详细输出")
    return p.parse_args()


def setup_logger(log_path: str, debug: bool) -> logging.Logger:
    logger = logging.getLogger("spx_downloader")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    logger = setup_logger(args.log, args.debug)
    util.patchAsyncio()

    ib = IB()
    logger.info(f"Connecting to IB at {args.host}:{args.port} clientId={args.clientId} …")
    ib.connect(args.host, args.port, clientId=args.clientId, readonly=False)

    spx = build_spx_contract()
    days = get_past_trading_days(args.days)
    if args.limit:
        days = days[-args.limit :]

    pacer = Pacer(max_per_window=args.max_per_window, window_sec=600, min_interval=args.min_interval)
    req_timestamps = deque()

    for d in days:
        try:
            process_day(
                ib,
                spx,
                d,
                pacer,
                req_timestamps,
                args.out,
                logger,
                args.debug,
                args.violation_backoff,
            )
        except Exception as e:
            logger.exception(f"Processing {d} failed: {e}")

    ib.disconnect()
    logger.info("All done.")


if __name__ == "__main__":
    main()
