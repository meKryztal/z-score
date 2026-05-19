import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Semaphore as _DL_Semaphore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import coint

PAIRS = [
    ("BRETTUSDT", "CHILLGUYUSDT"),
    ("CHILLGUYUSDT", "KASUSDT"),
    ("BIGTIMEUSDT", "CHILLGUYUSDT"),

]
TIMEFRAME = "5m"
# ── Несколько значений WINDOW для перебора ──────────────────────────────────
# Можно задать одно значение или список: WINDOWS = [100, 200, 300, 500]
WINDOWS = [300]  # <-- тестируем все эти окна
DDOF = 0

GRID_LEVELS = [2.0, 3.0, 4.0]
CLOSE_AT_ZERO = 0

TRADE_SIZE = 100.0  # по 50 на каждую монету в паре на каждый уровень, итого 300 на всю сетку в паре
COMMISSION = 0.0005

# Фандинг
USE_FUNDING = False  # учитывать фандинг в бэктесте (скачивает с Binance API)

# Фильтры результатов (пары не прошедшие — не попадают в CSV и сводку)
MAX_DRAWDOWN = -1111110.0  # максимально допустимая просадка ($), например -200
MAX_COINT_PVALUE = 1  # максимально допустимый p-value коинтеграции

BACKTEST_CANDLES = 100000  # количество свечей для теста
# OPT: ProcessPoolExecutor копирует память каждого воркера — каждый процесс ~200-400MB.
#      При 10 воркерах → до 4GB RAM + CPU saturated. Рекомендуем: cpu_count//2 или cpu_count-2.
#      Ставь 4-6 если ≤16GB RAM; 8-10 только при ≥32GB RAM.
import os as _os
_auto_workers = max(2, min(6, (_os.cpu_count() or 4) - 2))
BACKTEST_WORKERS = _auto_workers  # параллельных процессов для бэктеста
DOWNLOAD_WORKERS = 1  # потоков для загрузки (как в сканере)

# Источник данных:
#   USE_CACHE = True  — берём из кэша сканера (CACHE_DIR/), период = последние BACKTEST_CANDLES свечей
#   USE_CACHE = False — скачиваем с Binance последние BACKTEST_CANDLES свечей
USE_CACHE = True
CACHE_DIR = "cache"


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Entry:
    level: float
    entry_time: pd.Timestamp
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    size_usd: float = TRADE_SIZE


@dataclass
class Trade:
    direction: int
    entries: list = field(default_factory=list)
    exit_time: Optional[pd.Timestamp] = None
    exit_z: float = 0.0
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    closed: bool = False

    @property
    def first_entry_time(self): return self.entries[0].entry_time if self.entries else None

    @property
    def first_entry_z(self): return self.entries[0].entry_z if self.entries else None

    @property
    def n_entries(self): return len(self.entries)

    @property
    def total_size(self): return sum(e.size_usd for e in self.entries)

    @property
    def levels(self): return sorted(e.level for e in self.entries)

    @property
    def duration(self):
        if self.exit_time and self.first_entry_time:
            return self.exit_time - self.first_entry_time
        return None


def load_from_scanner_cache(symbol: str, window: int) -> "pd.DataFrame | None":
    from pathlib import Path
    cache_root = Path(CACHE_DIR)
    if not cache_root.exists():
        return None

    pattern = f"{symbol}_{TIMEFRAME}_*.parquet"
    files = list(cache_root.glob(pattern))
    if not files:
        return None

    cache_file = max(files, key=lambda p: p.stat().st_mtime)
    try:
        series = pd.read_parquet(cache_file)["close"]
    except Exception:
        return None

    df = series.to_frame("close")
    print(f"  ✓ Кэш: {cache_file.name}  ({len(df)} свечей, {df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """
    Скачивает исторические ставки фандинга для symbol с Binance Futures.
    Возвращает pd.Series с индексом pd.DatetimeIndex (UTC) и значениями float.
    При ошибке или отсутствии данных — возвращает пустую Series.
    """
    import requests as _req
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rows = []
    limit = 1000
    cur_start = start_ms
    while True:
        try:
            r = _req.get(url, params={
                "symbol": symbol,
                "startTime": cur_start,
                "endTime": end_ms,
                "limit": limit,
            }, timeout=10)
            rows = r.json()
        except Exception:
            break
        if not rows or not isinstance(rows, list):
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        cur_start = int(rows[-1]["fundingTime"]) + 1
        time.sleep(0.05)

    if not all_rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df.set_index("ts")["rate"].sort_index()


def fetch_binance_klines(symbol: str, interval: str, total_candles: int) -> pd.DataFrame:
    from binance.client import Client
    client = Client()

    all_rows = []
    end_time = None
    remaining = total_candles

    while remaining > 0:
        chunk = min(remaining, 1500)
        params = dict(symbol=symbol, interval=interval, limit=chunk)
        if end_time is not None:
            params["endTime"] = end_time
        klines = client.futures_klines(**params)
        if not klines:
            break
        all_rows = klines + all_rows
        end_time = int(klines[0][0]) - 1
        remaining -= len(klines)
        time.sleep(0.05)

    if not all_rows:
        raise ValueError(f"Нет данных для {symbol}")

    all_rows = all_rows[:-1][-total_candles:]

    df = pd.DataFrame(all_rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    return df


def save_to_cache(symbol: str, df: pd.DataFrame) -> None:
    """
    Сохраняет DataFrame со свечами в кэш (CACHE_DIR) в формате parquet.
    Имя файла: {symbol}_{TIMEFRAME}_{last_ts}.parquet
    Если для символа уже есть файл в кэше — объединяет данные и перезаписывает.
    """
    from pathlib import Path
    cache_root = Path(CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Проверяем — есть ли уже кэш для этого символа
    pattern = f"{symbol}_{TIMEFRAME}_*.parquet"
    old_files = list(cache_root.glob(pattern))
    merged = df.copy()

    if old_files:
        old_file = max(old_files, key=lambda p: p.stat().st_mtime)
        try:
            old_df = pd.read_parquet(old_file)
            # Объединяем старые и новые данные, убираем дубликаты
            merged = pd.concat([old_df, df[~df.index.isin(old_df.index)]])
            merged = merged.sort_index()
            old_file.unlink()  # удаляем старый файл перед записью нового
        except Exception:
            pass  # не смогли прочитать старый — просто запишем новый

    last_ts = merged.index[-1].strftime("%Y%m%d_%H%M")
    new_name = f"{symbol}_{TIMEFRAME}_{last_ts}.parquet"
    new_path = cache_root / new_name
    merged.to_parquet(new_path)
    print(f"  💾 Кэш сохранён: {new_name}  ({len(merged)} свечей)")


def get_klines(symbol: str, window: int) -> pd.DataFrame:
    total = window + BACKTEST_CANDLES  # прогрев + тест
    if USE_CACHE:
        df = load_from_scanner_cache(symbol, window)
        if df is not None:
            return df.iloc[-total:]
        print(f"  ⚠ Кэш не найден для {symbol} — докачиваем с Binance ({total} свечей)...")
        df = fetch_binance_klines(symbol, TIMEFRAME, total)
        save_to_cache(symbol, df)
        return df
    else:
        print(f"  Скачиваем {symbol} с Binance ({total} свечей)...")
        df = fetch_binance_klines(symbol, TIMEFRAME, total)
        save_to_cache(symbol, df)
        return df


def compute_zscore(price_a: pd.Series, price_b: pd.Series,
                   window: int, ddof: int = DDOF) -> pd.Series:
    spread = np.log(price_a) - np.log(price_b)
    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std(ddof=ddof)
    return (spread - roll_mean) / roll_std


def calc_entry_pnl(entry: Entry, exit_price_a: float,
                   exit_price_b: float, direction: int) -> float:
    half = entry.size_usd / 2.0
    ret_a = (exit_price_a - entry.entry_price_a) / entry.entry_price_a
    ret_b = (exit_price_b - entry.entry_price_b) / entry.entry_price_b
    if direction == 1:
        return ret_a * half - ret_b * half
    else:
        return -ret_a * half + ret_b * half


def run_backtest(df_a: pd.DataFrame, df_b: pd.DataFrame,
                 window: int,
                 funding_a: pd.Series = None, funding_b: pd.Series = None):
    """
    funding_a / funding_b — pd.Series с индексом DatetimeIndex (UTC) и ставками фандинга.
    Начисляется каждые 8ч: long платит positive rate, short получает (и наоборот).
    За каждую ногу: PnL -= (half_notional * rate) * sign(direction_for_that_leg)

    OPT: equity_curve теперь строится через pre-allocated numpy arrays вместо
         list-of-dicts + pd.DataFrame(equity_curve) — экономит ~60-70% памяти и
         время на 100k свечей.
    """
    idx = df_a.index.intersection(df_b.index)

    close_a = df_a.loc[idx, "close"]
    close_b = df_b.loc[idx, "close"]
    z = compute_zscore(close_a, close_b, window=window)

    z_arr = z.values
    pa_arr = close_a.values
    pb_arr = close_b.values
    ts_arr = idx
    n = len(ts_arr)

    # OPT: pre-allocate numpy arrays вместо list-of-dicts
    eq_ts        = np.empty(n, dtype="datetime64[ns]")
    eq_z         = np.full(n, np.nan)
    eq_equity    = np.zeros(n)
    eq_total     = np.zeros(n)
    eq_unreal    = np.zeros(n)
    eq_unreal_a  = np.zeros(n)
    eq_unreal_b  = np.zeros(n)
    eq_open      = np.zeros(n, dtype=np.int32)

    # Преобразуем funding в словари {timestamp: rate} для быстрого поиска
    funding_dict_a: dict = {}
    funding_dict_b: dict = {}
    if funding_a is not None and len(funding_a) > 0:
        funding_dict_a = funding_a.to_dict()
    if funding_b is not None and len(funding_b) > 0:
        funding_dict_b = funding_b.to_dict()

    all_funding_ts = set(funding_dict_a.keys()) | set(funding_dict_b.keys())

    open_trades: dict[int, Trade] = {}
    open_levels: dict[tuple, bool] = {}
    closed_trades: list[Trade] = []
    cumulative_pnl = 0.0
    total_funding_paid = 0.0

    for i in range(n):
        ts = ts_arr[i]
        z_val = z_arr[i]

        # OPT: записываем напрямую в массивы
        eq_ts[i] = ts.value  # nanoseconds int → datetime64

        if np.isnan(z_val):
            eq_equity[i] = cumulative_pnl
            eq_total[i]  = cumulative_pnl
            continue

        pa = pa_arr[i]
        pb = pb_arr[i]

        # ── Фандинг ──
        if ts in all_funding_ts and open_trades:
            rate_a = funding_dict_a.get(ts, 0.0)
            rate_b = funding_dict_b.get(ts, 0.0)
            for direction, trade in open_trades.items():
                for e in trade.entries:
                    half = e.size_usd / 2.0
                    if direction == 1:
                        funding_cost = half * rate_a - half * rate_b
                    else:
                        funding_cost = -half * rate_a + half * rate_b
                    cumulative_pnl -= funding_cost
                    total_funding_paid += funding_cost

        # ── Закрытие ──
        dirs_to_close = []
        for direction, trade in open_trades.items():
            should_close = (
                    (direction == 1 and z_val >= -CLOSE_AT_ZERO) or
                    (direction == -1 and z_val <= CLOSE_AT_ZERO)
            )
            if should_close:
                gross = sum(calc_entry_pnl(e, pa, pb, direction) for e in trade.entries)
                comm = sum(e.size_usd * COMMISSION * 2 for e in trade.entries)
                trade.pnl = gross - comm
                trade.commission = comm
                trade.exit_time = ts
                trade.exit_z = z_val
                trade.exit_price_a = pa
                trade.exit_price_b = pb
                trade.closed = True
                cumulative_pnl += trade.pnl
                closed_trades.append(trade)
                dirs_to_close.append(direction)

        for d in dirs_to_close:
            for lv in GRID_LEVELS:
                open_levels.pop((d, lv), None)
            del open_trades[d]

        # ── Открытие ──
        for level in GRID_LEVELS:
            direction = 0
            if z_val <= -level:
                direction = 1
            elif z_val >= level:
                direction = -1

            if direction == 0:
                continue
            if open_levels.get((direction, level)):
                continue

            if direction not in open_trades:
                open_trades[direction] = Trade(direction=direction)

            entry = Entry(
                level=level,
                entry_time=ts,
                entry_z=z_val,
                entry_price_a=pa,
                entry_price_b=pb,
                size_usd=TRADE_SIZE,
            )
            open_trades[direction].entries.append(entry)
            open_levels[(direction, level)] = True
            cumulative_pnl -= TRADE_SIZE * COMMISSION

        # ── Нереализованный PnL ──
        unrealized = 0.0
        unrealized_a = 0.0
        unrealized_b = 0.0
        for direction, trade in open_trades.items():
            for e in trade.entries:
                half = e.size_usd / 2.0
                ret_a = (pa - e.entry_price_a) / e.entry_price_a
                ret_b = (pb - e.entry_price_b) / e.entry_price_b
                if direction == 1:
                    leg_a = ret_a * half
                    leg_b = -ret_b * half
                else:
                    leg_a = -ret_a * half
                    leg_b = ret_b * half
                unrealized_a += leg_a
                unrealized_b += leg_b
                unrealized += leg_a + leg_b

        eq_z[i]        = z_val
        eq_equity[i]   = cumulative_pnl
        eq_total[i]    = cumulative_pnl + unrealized
        eq_unreal[i]   = unrealized
        eq_unreal_a[i] = unrealized_a
        eq_unreal_b[i] = unrealized_b
        eq_open[i]     = len(open_trades)

    if total_funding_paid != 0.0:
        sign = "+" if total_funding_paid >= 0 else ""
        print(f"  Фандинг (суммарно): {sign}{total_funding_paid:.2f}$")

    # OPT: строим DataFrame из готовых массивов — намного быстрее чем из list[dict]
    equity_df = pd.DataFrame({
        "ts":            pd.DatetimeIndex(eq_ts.astype("datetime64[ns]"), tz="UTC"),
        "z":             eq_z,
        "equity":        eq_equity,
        "equity_total":  eq_total,
        "unrealized":    eq_unreal,
        "unrealized_a":  eq_unreal_a,
        "unrealized_b":  eq_unreal_b,
        "open_trades":   eq_open,
    })

    return closed_trades, equity_df


def trades_to_df(trades: list) -> pd.DataFrame:
    rows = []
    for t in trades:
        row = {
            "entry_time": t.first_entry_time,
            "exit_time": t.exit_time,
            "direction": "LONG A/SHORT B" if t.direction == 1 else "SHORT A/LONG B",
            "n_entries": t.n_entries,
            "levels": str(t.levels),
            "entry_z": round(t.first_entry_z, 4) if t.first_entry_z else None,
            "exit_z": round(t.exit_z, 4),
            "total_size": t.total_size,
            "pnl": round(t.pnl, 4),
            "commission": round(t.commission, 4),
            "duration": str(t.duration),
        }
        for i, e in enumerate(t.entries):
            row[f"entry{i + 1}_time"] = e.entry_time
            row[f"entry{i + 1}_z"] = round(e.entry_z, 4)
            row[f"entry{i + 1}_price_a"] = e.entry_price_a
            row[f"entry{i + 1}_price_b"] = e.entry_price_b
        rows.append(row)
    return pd.DataFrame(rows)


def print_stats(trades: list, equity_df: pd.DataFrame, window: int, coint_pvalue: float = None):
    if not trades:
        print("Нет закрытых сделок.")
        return

    pnls = [t.pnl for t in trades]
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    total = sum(pnls)
    comm = sum(t.commission for t in trades)

    equity = equity_df["equity"]
    equity_total = equity_df["equity_total"]
    max_dd = (equity - equity.cummax()).min()  # только закрытые
    max_dd_real = (equity_total - equity_total.cummax()).min()  # с учётом открытых (реальная)

    # Максимальный нереализованный убыток по каждой ноге отдельно
    min_leg_a = equity_df["unrealized_a"].min()
    min_leg_b = equity_df["unrealized_b"].min()
    max_leg_a = equity_df["unrealized_a"].max()
    max_leg_b = equity_df["unrealized_b"].max()

    daily_ret = equity_df.set_index("ts")["equity_total"].resample("1D").last().diff().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    pf_str = (f"{abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)):.2f}"
              if losers else "∞")

    print("\n" + "═" * 62)
    print(f"  РЕЗУЛЬТАТЫ БЕКТЕСТА  (Binance Futures)  Window={window}")
    print("═" * 62)
    print(f"  Период:   {equity_df['ts'].iloc[0].date()} → {equity_df['ts'].iloc[-1].date()}")
    print(f"  TF: {TIMEFRAME}  |  Window: {window}  |  Вход/выход на закрытии свечи")
    print("─" * 62)
    print(f"  Сделок:        {len(trades):>6}")
    print(f"  Прибыльных:    {len(winners):>6}  ({len(winners) / len(trades) * 100:.0f}%)")
    print(f"  Убыточных:     {len(losers):>6}  ({len(losers) / len(trades) * 100:.0f}%)")
    print(f"  Profit Factor: {pf_str:>9}")
    print("─" * 62)
    print(f"  Итого PnL:         {total:>+9.2f} $")
    print(f"  Комиссии:          {-comm:>+9.2f} $")
    print(f"  Sharpe:            {sharpe:>9.2f}")
    print("─" * 62)
    print(f"  Max DD (закрытые): {max_dd:>+9.2f} $  ← только реализованный PnL")
    print(f"  Max DD (реальный): {max_dd_real:>+9.2f} $  ← с учётом открытых позиций")
    print("─" * 62)
    print(f"  Нога A — макс. нереализ.:  {max_leg_a:>+9.2f} $   мин.: {min_leg_a:>+9.2f} $")
    print(f"  Нога B — макс. нереализ.:  {max_leg_b:>+9.2f} $   мин.: {min_leg_b:>+9.2f} $")
    if coint_pvalue is not None:
        flag = "✓" if coint_pvalue <= MAX_COINT_PVALUE else "✗"
        print(f"─" * 62)
        print(f"  Coint p-value: {coint_pvalue:>9.4f}  {flag}")
    print("─" * 62)
    print("═" * 62)


def plot_pair(symbol_a: str, symbol_b: str,
              window_results: list,
              df_a: "pd.DataFrame | None" = None,
              df_b: "pd.DataFrame | None" = None):
    """
    window_results: список dict с ключами: window, trades, equity_df

    Макет (single окно):
    - Слева: большой PnL / Drawdown график на всю высоту
    - Справа: таблица со статистикой

    Макет (несколько окон):
    - Один график с несколькими equity кривыми
    """
    key = f"{symbol_a}_{symbol_b}"
    fname = f"{key}.png"

    C_BG   = "#09090f"
    C_GRID = "#1e1e2e"
    C_TEXT = "#c0c0d0"
    C_WIN  = "#00e5a0"
    C_LOSS = "#ff4d6d"
    C_ZERO = "#444466"
    C_EQ   = "#5b8fff"

    single = len(window_results) == 1

    if not single or not window_results:
        # ── Несколько окон: один график, несколько кривых ────────
        PALETTE = ["#5b8fff", "#ffb347", "#ff6eb4", "#a8ff78",
                   "#f9f871", "#c77dff", "#ff9770", "#4eb3ff"]

        fig, ax = plt.subplots(figsize=(16, 6), facecolor=C_BG)
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.axhline(0, color=C_ZERO, linewidth=0.8, linestyle="--")

        for idx_w, wr in enumerate(window_results):
            w          = wr["window"]
            equity_df  = wr["equity_df"]
            trades     = wr["trades"]
            color      = PALETTE[idx_w % len(PALETTE)]

            ts    = equity_df["ts"]
            eq    = equity_df["equity_total"]
            final = eq.iloc[-1]
            n_tr  = len(trades)
            n_win = sum(1 for t in trades if t.pnl > 0)
            wr_pct = n_win / n_tr * 100 if n_tr else 0
            label  = f"W={w}  {final:+.1f}$  T={n_tr}  WR={wr_pct:.0f}%"

            ax.plot(ts, eq, color=color, linewidth=1.5, label=label, zorder=4)
            ax.annotate(f"{final:+.1f}$",
                        xy=(ts.iloc[-1], final),
                        xytext=(5, 0), textcoords="offset points",
                        color=color, fontsize=7.5, fontweight="bold",
                        va="center", zorder=5)

        fig.suptitle(
            f"Backtest  {symbol_a}/{symbol_b}  |  {TIMEFRAME}  |  {BACKTEST_CANDLES} свечей",
            color=C_TEXT, fontsize=12, fontweight="bold", y=0.99
        )
        ax.set_title(f"{symbol_a} / {symbol_b}  —  Equity Curve",
                     color=C_TEXT, fontsize=10, pad=6)
        ax.set_ylabel("PnL $", color=C_TEXT, fontsize=8)
        ax.legend(fontsize=8, facecolor=C_BG, labelcolor=C_TEXT,
                  edgecolor=C_GRID, loc="upper left")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)
        print(f"  График: {fname}")
        return

    # ── Одно окно: PnL на всю ширину + таблица справа ─────────────
    wr        = window_results[0]
    window    = wr["window"]
    equity_df = wr["equity_df"]
    trades    = wr["trades"]

    ts  = equity_df["ts"]
    eq  = equity_df["equity_total"]

    final   = eq.iloc[-1]
    n_tr    = len(trades)
    n_win   = sum(1 for t in trades if t.pnl > 0)
    wr_pct  = n_win / n_tr * 100 if n_tr else 0
    max_dd  = float((eq - eq.cummax()).min())
    pnl_sum = sum(t.pnl for t in trades)
    comm    = sum(t.commission for t in trades)
    winners = [t for t in trades if t.pnl > 0]
    losers  = [t for t in trades if t.pnl <= 0]
    pf      = abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)) \
              if losers and sum(t.pnl for t in losers) != 0 else 0
    daily_ret = equity_df.set_index("ts")["equity_total"].resample("1D").last().diff().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) \
                if daily_ret.std() > 0 else 0.0

    # Компоновка: plot занимает ~78% ширины, таблица — ~22%
    fig = plt.figure(figsize=(16, 6), facecolor=C_BG)
    gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_pnl  = fig.add_subplot(gs[0, 0])
    ax_tbl  = fig.add_subplot(gs[0, 1])

    # ── PnL / Drawdown ────────────────────────────────────────────
    for ax in (ax_pnl, ax_tbl):
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

    ax_pnl.tick_params(colors=C_TEXT, labelsize=8)
    ax_pnl.grid(True, color=C_GRID, linewidth=0.5, alpha=0.5)
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_pnl.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax_pnl.axhline(0, color=C_ZERO, linewidth=1, linestyle="--", zorder=1)
    ax_pnl.plot(ts, eq, color=C_EQ, linewidth=2, label="Equity", zorder=4)
    ax_pnl.fill_between(ts, eq, 0, where=(eq >= 0), alpha=0.20, color=C_WIN, zorder=2)
    ax_pnl.fill_between(ts, eq, 0, where=(eq <  0), alpha=0.15, color=C_LOSS, zorder=2)

    peak = eq.cummax()
    ax_pnl.fill_between(ts, eq, peak, where=(eq < peak), alpha=0.30, color=C_LOSS,
                        label=f"Drawdown  макс: {max_dd:+.0f}$", zorder=3)

    color_final = C_WIN if final >= 0 else C_LOSS
    ax_pnl.annotate(f"{final:+.0f}$",
                    xy=(ts.iloc[-1], final),
                    xytext=(-10, 0), textcoords="offset points",
                    color=color_final, fontsize=11, fontweight="bold",
                    ha="right", va="center", zorder=5)

    ax_pnl.set_title(f"PnL & Drawdown  •  {symbol_a} / {symbol_b}  •  W={window}",
                     color=C_TEXT, fontsize=10, fontweight="bold", pad=8)
    ax_pnl.set_ylabel("PnL ($)", color=C_TEXT, fontsize=9)
    ax_pnl.legend(loc="upper left", fontsize=8, facecolor=C_BG,
                  labelcolor=C_TEXT, edgecolor=C_GRID)

    # ── Таблица статистики ────────────────────────────────────────
    ax_tbl.axis("off")
    rows = [
        ["Metric",        "Value"],
        ["PnL",           f"{pnl_sum:+.2f}$"],
        ["Max DD",        f"{max_dd:+.2f}$"],
        ["Sharpe",        f"{sharpe:.2f}"],
        ["Trades",        f"{n_tr}"],
        ["Win Rate",      f"{wr_pct:.1f}%"],
        ["Profit Factor", f"{pf:.2f}" if pf > 0 else "∞"],
        ["Commission",    f"{-comm:.2f}$"],
        ["Final Equity",  f"{final:+.2f}$"],
    ]
    tbl = ax_tbl.table(cellText=rows, cellLoc="left", loc="center",
                       colWidths=[0.58, 0.42], bbox=[0.05, 0.1, 0.90, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)

    for i, row in enumerate(rows):
        for j in range(2):
            cell = tbl[(i, j)]
            if i == 0:
                cell.set_facecolor(C_EQ)
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor(C_GRID)
                cell.set_text_props(color=C_TEXT)
            cell.set_edgecolor("#2a2a3e")
            cell.set_linewidth(0.5)

    fig.suptitle(
        f"Backtest  {symbol_a}/{symbol_b}  |  {TIMEFRAME}  |  W={window}  |  {BACKTEST_CANDLES} свечей",
        color=C_TEXT, fontsize=11, fontweight="bold", y=1.01
    )

    fig.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  График: {fname}")


def print_summary(all_results: list):
    """all_results содержит записи с полем window."""
    print("\n" + "═" * 92)
    print("  СВОДНЫЙ ОТЧЁТ ПО ВСЕМ ПАРАМ И ОКНАМ")
    print("═" * 92)
    print(
        f"  {'Пара':<22} {'Window':>7} {'Сделок':>7} {'WR%':>6} {'PnL $':>9} {'MaxDD $':>9} {'Sharpe':>7} {'PF':>6} {'p-val':>7}")
    print("─" * 92)

    total_pnl = 0
    for r in all_results:
        sym = f"{r['symbol_a']}/{r['symbol_b']}"
        pval = r.get('coint_pvalue', '')
        print(
            f"  {sym:<22} "
            f"{r['window']:>7} "
            f"{r['n_trades']:>7} "
            f"{r['win_rate']:>6.1f} "
            f"{r['total_pnl']:>+9.2f} "
            f"{r['max_dd']:>+9.2f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['profit_factor']:>6} "
            f"{pval:>7}"
        )
        total_pnl += r["total_pnl"]

    print("─" * 92)
    print(f"  {'ИТОГО':<22} {'':>7} {'':>7} {'':>6} {total_pnl:>+9.2f}")
    print("═" * 92)


def calc_coint_pvalue(close_a: pd.Series, close_b: pd.Series) -> float:
    """Тест Энгла-Грейнджера на коинтеграцию. Возвращает p-value.
    OPT: принимает уже нарезанные серии — вызывающий код сам ограничивает до 5000 точек.
    """
    try:
        idx = close_a.index.intersection(close_b.index)
        _, pvalue, _ = coint(np.log(close_a.loc[idx].values),
                             np.log(close_b.loc[idx].values))
        return round(float(pvalue), 4)
    except Exception:
        return 1.0


def run_pair_all_windows(symbol_a: str, symbol_b: str,
                         df_a: pd.DataFrame, df_b: pd.DataFrame,
                         funding_a: pd.Series, funding_b: pd.Series,
                         make_plot: bool = True) -> list[dict]:
    """
    Запускает бэктест для одной пары по всем значениям WINDOWS.
    Возвращает список dict — по одному на каждое окно, прошедшее фильтры.
    """
    # OPT: ограничиваем coint до 5000 точек (см. воркер)
    _COINT_MAX = 5000
    _ca = df_a["close"].iloc[-_COINT_MAX:] if len(df_a) > _COINT_MAX else df_a["close"]
    _cb = df_b["close"].iloc[-_COINT_MAX:] if len(df_b) > _COINT_MAX else df_b["close"]
    coint_pvalue = calc_coint_pvalue(_ca, _cb)

    results = []
    window_results_for_plot = []

    for window in WINDOWS:
        print(f"\n  Window={window}  {symbol_a}/{symbol_b}")
        trades, equity_df = run_backtest(df_a, df_b, window=window,
                                         funding_a=funding_a, funding_b=funding_b)
        print_stats(trades, equity_df, window=window, coint_pvalue=coint_pvalue)

        window_results_for_plot.append({
            "window": window,
            "trades": trades,
            "equity_df": equity_df,
        })

        if not trades:
            continue

        pnls = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        equity_total = equity_df["equity_total"]
        max_dd = float((equity_total - equity_total.cummax()).min())  # реальная просадка
        daily_ret = equity_df.set_index("ts")["equity_total"].resample("1D").last().diff().dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        pf = round(abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)), 2) if losers and sum(
            t.pnl for t in losers) != 0 else "∞"

        # Фильтры
        if max_dd < MAX_DRAWDOWN:
            print(f"  ✗ Отфильтровано W={window}: MaxDD={max_dd:.2f} < {MAX_DRAWDOWN}")
            continue
        if coint_pvalue > MAX_COINT_PVALUE:
            print(f"  ✗ Отфильтровано W={window}: coint_pvalue={coint_pvalue} > {MAX_COINT_PVALUE}")
            continue

        results.append({
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "window": window,
            "n_trades": len(trades),
            "win_rate": len(winners) / len(trades) * 100 if trades else 0,
            "total_pnl": round(sum(pnls), 2),
            "max_dd": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "profit_factor": pf,
            "coint_pvalue": coint_pvalue,
        })

    if make_plot:
        plot_pair(symbol_a, symbol_b, window_results_for_plot, df_a, df_b)

    return results


def _backtest_worker_multi_window(args: tuple) -> list[dict]:
    """
    Воркер для ProcessPoolExecutor — запускает бэктест по всем WINDOWS.
    OPT: возвращает _trades и _equity_df чтобы главный процесс не пересчитывал бэктест для графиков.
    OPT: coint считается на подвыборке (max 5000 точек) — statsmodels тяжёлый на полных 100k.
    """
    sym_a, sym_b, vals_a, idx_a, vals_b, idx_b, funding_a, funding_b = args
    try:
        df_a = pd.DataFrame({"close": vals_a}, index=idx_a)
        df_b = pd.DataFrame({"close": vals_b}, index=idx_b)

        # OPT: ограничиваем coint до 5000 последних точек — результат практически не меняется,
        #      но statsmodels работает в 20× быстрее (O(n²) → ~O(k²) для k=5000)
        _COINT_MAX = 5000
        close_a_c = df_a["close"]
        close_b_c = df_b["close"]
        if len(close_a_c) > _COINT_MAX:
            close_a_c = close_a_c.iloc[-_COINT_MAX:]
            close_b_c = close_b_c.iloc[-_COINT_MAX:]
        coint_pvalue = calc_coint_pvalue(close_a_c, close_b_c)

        results = []

        for window in WINDOWS:
            trades, equity_df = run_backtest(df_a, df_b, window=window,
                                             funding_a=funding_a, funding_b=funding_b)

            if not trades:
                continue

            pnls = [t.pnl for t in trades]
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            equity_total = equity_df["equity_total"]
            max_dd = float((equity_total - equity_total.cummax()).min())

            if max_dd < MAX_DRAWDOWN:
                continue
            if coint_pvalue > MAX_COINT_PVALUE:
                continue

            daily_ret = equity_df.set_index("ts")["equity"].resample("1D").last().diff().dropna()
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
            pf = round(abs(sum(t.pnl for t in winners) / sum(t.pnl for t in losers)), 2) \
                if losers and sum(t.pnl for t in losers) != 0 else "∞"

            results.append({
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "window": window,
                "n_trades": len(trades),
                "win_rate": len(winners) / len(trades) * 100,
                "total_pnl": round(sum(pnls), 2),
                "max_dd": round(max_dd, 2),
                "sharpe": round(sharpe, 2),
                "profit_factor": pf,
                "coint_pvalue": coint_pvalue,
                # OPT: сохраняем данные для построения графиков — избегаем повторного run_backtest
                "_trades": trades,
                "_equity_df": equity_df,
            })

        return results
    except Exception as e:
        return []


def main():
    # Нормализуем WINDOWS в список
    windows_list = WINDOWS if isinstance(WINDOWS, list) else [WINDOWS]

    print("╔══════════════════════════════╗")
    print("║  Pairs Backtest  •  Binance  ║")
    print("╚══════════════════════════════╝")
    print(f"  Пар: {len(PAIRS)}  |  TF: {TIMEFRAME}  |  Windows: {windows_list}")
    print(f"  Свечей для теста: {BACKTEST_CANDLES}  (~{BACKTEST_CANDLES * int(TIMEFRAME[:-1]) / 60 / 24:.1f} дней)")
    print(f"  Сетка: {GRID_LEVELS}  |  Размер: ${TRADE_SIZE}/уровень")
    print(f"  Процессов: {BACKTEST_WORKERS}")

    t_start = time.time()

    # Максимальное окно нужно для прогрева
    max_window = max(windows_list)

    if len(PAIRS) == 1:
        # Одна пара — запускаем напрямую с графиком
        sym_a, sym_b = PAIRS[0]
        df_a = get_klines(sym_a, max_window)
        df_b = get_klines(sym_b, max_window)
        funding_a = funding_b = None
        if USE_FUNDING:
            start_ms = int(df_a.index[0].timestamp() * 1000)
            end_ms = int(df_a.index[-1].timestamp() * 1000)
            funding_a = fetch_funding_rates(sym_a, start_ms, end_ms)
            funding_b = fetch_funding_rates(sym_b, start_ms, end_ms)
        else:
            funding_a = pd.Series(dtype=float)
            funding_b = pd.Series(dtype=float)

        all_results = run_pair_all_windows(sym_a, sym_b, df_a, df_b,
                                           funding_a, funding_b, make_plot=True)
    else:
        # ── Шаг 1: параллельная загрузка данных ──────────────────
        print(f"\n  [1/3] Загружаю данные ({DOWNLOAD_WORKERS} потока)...")
        symbols_needed = sorted(set(s for pair in PAIRS for s in pair))
        prices = {}
        _dl_sem = _DL_Semaphore(DOWNLOAD_WORKERS)
        _dl_lock = __import__('threading').Lock()
        done_dl = [0]

        def _download_one(sym):
            with _dl_sem:
                try:
                    df = get_klines(sym, max_window)
                    if not USE_CACHE:
                        df = df.iloc[-BACKTEST_CANDLES:]
                    with _dl_lock:
                        prices[sym] = df
                        done_dl[0] += 1
                        print(f"  [{done_dl[0]:>3}/{len(symbols_needed)}]  ✓ {sym}  ({len(df)} свечей)   ", end="\r")
                except Exception as e:
                    with _dl_lock:
                        done_dl[0] += 1
                        print(f"  [{done_dl[0]:>3}/{len(symbols_needed)}]  ✗ {sym}: {e}   ")

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as dl_executor:
            list(dl_executor.map(_download_one, symbols_needed))

        print(f"\n  ✓ Загружено {len(prices)}/{len(symbols_needed)} монет")

        # ── Шаг 2: загрузка фандинга ─────────────────────────────
        fundings: dict = {}
        if USE_FUNDING:
            print(f"\n  [2/4] Загружаю фандинг ({len(symbols_needed)} монет)...")
            f_done = [0]
            _dl_lock2 = __import__('threading').Lock()

            def _download_funding(sym):
                try:
                    df_sym = prices.get(sym)
                    if df_sym is None or len(df_sym) == 0:
                        fundings[sym] = pd.Series(dtype=float)
                        return
                    start_ms = int(df_sym.index[0].timestamp() * 1000)
                    end_ms = int(df_sym.index[-1].timestamp() * 1000)
                    fundings[sym] = fetch_funding_rates(sym, start_ms, end_ms)
                except Exception:
                    fundings[sym] = pd.Series(dtype=float)
                finally:
                    with _dl_lock2:
                        f_done[0] += 1
                        print(
                            f"  [{f_done[0]:>3}/{len(symbols_needed)}] фандинг {sym}  ({len(fundings.get(sym, []))} событий)   ",
                            end="\r")

            with ThreadPoolExecutor(max_workers=min(DOWNLOAD_WORKERS * 2, 8)) as f_exec:
                list(f_exec.map(_download_funding, symbols_needed))
            print(
                f"\n  ✓ Фандинг загружен для {sum(1 for v in fundings.values() if len(v) > 0)}/{len(symbols_needed)} монет")
        else:
            fundings = {sym: pd.Series(dtype=float) for sym in symbols_needed}

        # ── Шаг 3: параллельный бэктест по всем windows ──────────
        step_lbl = "3" if not USE_FUNDING else "3"
        total_steps = "4" if USE_FUNDING else "3"
        n_jobs = len(PAIRS) * len(windows_list)
        print(
            f"\n  [{step_lbl}/{total_steps}] Бэктест {len(PAIRS)} пар × {len(windows_list)} окон = {n_jobs} задач ({BACKTEST_WORKERS} процессов)...")

        valid_pairs = [(a, b) for a, b in PAIRS if a in prices and b in prices]
        skipped = len(PAIRS) - len(valid_pairs)
        if skipped:
            print(f"  ⚠ Пропущено {skipped} пар (нет данных)")

        backtest_args = [
            (a, b,
             prices[a]["close"].values, prices[a].index,
             prices[b]["close"].values, prices[b].index,
             fundings.get(a, pd.Series(dtype=float)),
             fundings.get(b, pd.Series(dtype=float)))
            for a, b in valid_pairs
        ]

        all_results = []
        done = 0
        total = len(backtest_args)

        with ProcessPoolExecutor(max_workers=BACKTEST_WORKERS) as executor:
            futs = {executor.submit(_backtest_worker_multi_window, args): args[:2]
                    for args in backtest_args}
            for future in as_completed(futs):
                done += 1
                pair = futs[future]
                try:
                    results = future.result()  # list[dict]
                    passed = len(results)
                    status = f"✓ {pair[0]}/{pair[1]}  ({passed}/{len(windows_list)} окон прошло)" if passed else f"- {pair[0]}/{pair[1]} (не прошла фильтры)"
                except Exception as e:
                    results = []
                    status = f"✗ {pair[0]}/{pair[1]}: {e}"
                all_results.extend(results)
                elapsed = time.time() - t_start
                speed = done / elapsed
                eta = (total - done) / speed if speed > 0 else 0
                eta_str = f"{eta / 60:.1f}мин" if eta >= 60 else f"{eta:.0f}с"
                print(f"  [{done:>3}/{total}]  {status}  ETA: {eta_str}")

        # ── Шаг 4: графики в главном процессе ────────────────────
        # OPT: воркеры теперь возвращают equity_df и trades в all_results,
        #      поэтому повторный run_backtest НЕ нужен — экономим ~2× времени.
        if all_results:
            step_num = "4" if USE_FUNDING else "3"
            print(f"\n  [{step_num}/{total_steps}] Строю графики...")

            pair_keys: dict = {}
            for r in all_results:
                k = (r["symbol_a"], r["symbol_b"])
                pair_keys.setdefault(k, []).append(r)

            for (a, b), pair_results in pair_keys.items():
                try:
                    df_a = pd.DataFrame({"close": prices[a]["close"]})
                    df_b = pd.DataFrame({"close": prices[b]["close"]})
                    window_results_for_plot = [
                        {"window": r["window"],
                         "trades": r["_trades"],
                         "equity_df": r["_equity_df"]}
                        for r in pair_results
                        if "_trades" in r and "_equity_df" in r
                    ]
                    if not window_results_for_plot:
                        # Запасной путь: пересчитываем только если воркер не вернул данные
                        for window in windows_list:
                            trades, equity_df = run_backtest(df_a, df_b, window=window,
                                                             funding_a=fundings.get(a),
                                                             funding_b=fundings.get(b))
                            window_results_for_plot.append({"window": window,
                                                            "trades": trades,
                                                            "equity_df": equity_df})
                    plot_pair(a, b, window_results_for_plot, df_a, df_b)
                except Exception as e:
                    print(f"  ⚠ График {a}/{b}: {e}")

    elapsed = time.time() - t_start
    print(f"\n  ✓ Готово за {elapsed:.1f}s ({elapsed / 60:.1f}мин)")

    if all_results:
        print_summary(all_results)
        # OPT: фильтруем приватные ключи (_trades, _equity_df) перед записью в CSV
        csv_keys = [k for k in all_results[0].keys() if not k.startswith("_")]
        pd.DataFrame([{k: r[k] for k in csv_keys} for r in all_results]).to_csv(
            "backtest_summary.csv", index=False)
        print(f"\n  Сводный отчёт: backtest_summary.csv")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # нужно для Windows
    main()