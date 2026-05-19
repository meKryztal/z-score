import json, sys, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings("ignore")


try:
    import cupy as cp
    from numba import cuda
    import numba
    _gpu_available = cuda.is_available()
except ImportError:
    _gpu_available = False

if __name__ == "__mp_main__" or __name__ == "__main__":
    pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        НАСТРОЙКИ — МЕНЯЙ ЗДЕСЬ                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Биржа ─────────────────────────────────────────────────────────────────────
#  "binance"  →  кеш: cache/
#  "okx"      →  кеш: cache_okx/
#  "bybit"    →  кеш: cache_bybit/
#  Также можно передать аргументом:  python scanerGPU.py --exchange okx
EXCHANGE = "binance"

# ── Данные ────────────────────────────────────────────────────────────────────
TIMEFRAME       = "5m"      # таймфрейм свечей
CANDLES         = 100000     # сколько свечей загружать

# ── Окно(а) для z-score и бэктеста ───────────────────────────────────────────
#
#  Есть два режима — они НЕ конфликтуют, просто выбери один:
#
#  Режим 1 — одно окно (быстрее):
#    WINDOWS_TO_TEST = []       ← пустой список
#    WINDOW          = 300      ← используется это значение
#
#  Режим 2 — мульти-window (перебирает все окна, выбирает лучшее):
#    WINDOWS_TO_TEST = [300, 400, 500]   ← список окон для перебора
#    WINDOW          = 300               ← используется только в GPU-ядре
#                                          (GPU всегда берёт первый из списка)
#
WINDOW          = 300
WINDOWS_TO_TEST = []   # [] = одиночный режим, список = мульти-window

# ── Последовательный прогон по окнам (отдельный файл на каждое окно) ──────────
#  Если задан — игнорирует WINDOW и WINDOWS_TO_TEST, запускает полный скан
#  последовательно для каждого значения и сохраняет:
#    pairs_report_binance_w300.csv
#    pairs_report_binance_w400.csv  и т.д.
#  Загрузка свечей/фандинга делается ОДИН РАЗ перед циклом.
WINDOWS_SEQUENTIAL = [300]   # например [300, 400, 500]

# ── Сетка ─────────────────────────────────────────────────────────────────────
GRID_LEVELS     = [2.0, 3.0, 4.0]  # уровни z-score для открытия позиций
CLOSE_AT_ZERO   = 0                 # закрывать при z=0 (0) или допускать реверс (1)
DDOF            = 0

# ── Торговля ──────────────────────────────────────────────────────────────────
TRADE_SIZE      = 100.0     # размер сделки в USDT
COMMISSION      = 0.0005    # комиссия на сторону (0.05%)
LEVERAGE        = 10

# ── Фильтры результатов ───────────────────────────────────────────────────────
MIN_CORRELATION  = 0.6     # минимальная корреляция Pearson (предфильтр)
MIN_WINRATE      = 70.0     # минимальный win rate %
MIN_PNL          = 200.0     # минимальный итоговый PnL
MIN_VOLUME_24H   = 1000000  # минимальный объём за 24ч в USDT
MIN_UNIQUE_RATIO = 0.02     # доля уникальных цен (защита от "мёртвых" монет)
MAX_COINT_PVALUE = 1        # макс. p-value коинтеграции (1 = не фильтруем)
MAX_DRAWDOWN     = -50.0    # максимальная просадка (отрицательное число)

# ── Производительность ────────────────────────────────────────────────────────
TOP_N            = 20
MAX_WORKERS      = 3        # потоков для загрузки свечей
ANALYSIS_WORKERS = 10      # процессов для анализа пар (CPU-режим)

# ── Фандинг ───────────────────────────────────────────────────────────────────
USE_FUNDING      = False     # учитывать фандинг в бэктесте

# ── Кеш ───────────────────────────────────────────────────────────────────────
CACHE_MAX_AGE_H = 10000000000   # макс. возраст кеша в часах

# ── Ручной список символов ────────────────────────────────────────────────────
#  Если пустой [] — загружаются все активные USDT-M перпетуалы с биржи
SYMBOLS_MANUAL = []

# ── Первая нога (LEG1) ────────────────────────────────────────────────────────
#  Если задан — каждая монета из этого списка тестируется против ВСЕХ монет биржи.
#  SYMBOLS_MANUAL при этом игнорируется.
#
#  Пример:
#    SYMBOLS_LEG1 = ["BTCUSDT", "ETHUSDT"]
#    → тестируются пары: BTCUSDT×все, ETHUSDT×все
#
#  Фильтр объёма (MIN_VOLUME_24H) применяется к "всем монетам биржи" как обычно.
#  Сами монеты из SYMBOLS_LEG1 в фильтр объёма не попадают — они берутся всегда.
SYMBOLS_LEG1 = []


# ── GPU ────────────────────────────────────────────────────
GPU_BATCH_SIZE  = 256    # пар за раз на GPU (уменьши если OutOfMemory, напр. 256 или 128)
_N_LEVELS   = 3
_LEVELS     = (2.0, 3.0, 4.0)
_COMM       = 0.0005
_TSIZE      = 100.0
_CLOSE_ZERO = 0.0
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     ВНУТРЕННИЕ КОНСТАНТЫ (не трогай)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

EXCHANGE_URLS = {
    "binance": "https://fapi.binance.com",
    "okx":     "https://www.okx.com",
    "bybit":   "https://api.bybit.com",
}

EXCHANGE_CACHE_DIRS = {
    "binance": Path("cache"),
    "okx":     Path("cache_okx"),
    "bybit":   Path("cache_bybit"),
}


# Будут установлены в main() в зависимости от EXCHANGE
CACHE_DIR     = Path("cache")

# ─────────────────────────────────────────────────────────────────────────────

import requests as req
import time as _time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Semaphore, Lock


_progress_lock = Lock()
_req_done  = 0
_req_total = 0
_sym_done  = 0
_sym_total = 0
_t_start   = 0.0

def _tick_request(sym: str):
    global _req_done
    with _progress_lock:
        _req_done += 1
        pct     = _req_done / max(_req_total, 1) * 100
        bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        elapsed = _time.time() - _t_start
        eta     = (elapsed / _req_done) * (_req_total - _req_done)
        print(
            f"  [{bar}] {pct:5.1f}%  "
            f"запросов: {_req_done}/{_req_total}  "
            f"монет: {_sym_done}/{_sym_total}  "
            f"ETA: {eta:.0f}s  {sym:<18}",
            end="\r"
        )

def _tick_symbol():
    global _sym_done
    with _progress_lock:
        _sym_done += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  АДАПТЕРЫ БИРЖ
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────── Биньянс ─────────────────────────────────────────────────

def _binance_fetch_symbols() -> list[str]:
    url = f"{EXCHANGE_URLS['binance']}/fapi/v1/exchangeInfo"
    r = req.get(url, timeout=10)
    r.raise_for_status()
    symbols = [
        s["symbol"] for s in r.json()["symbols"]
        if s["symbol"].endswith("USDT")
        and s["status"] == "TRADING"
        and s["contractType"] == "PERPETUAL"
    ]
    return sorted(symbols)


def _binance_fetch_volume(symbols: list[str], min_vol: float) -> list[str]:
    for attempt in range(3):
        try:
            r = req.get(f"{EXCHANGE_URLS['binance']}/fapi/v1/ticker/24hr", timeout=15)
            if r.status_code == 418:
                wait = 10 * (attempt + 1)
                print(f"  ⚠ Binance rate limit (418), жду {wait}s...")
                _time.sleep(wait)
                continue
            r.raise_for_status()
            vol_map = {t["symbol"]: float(t["quoteVolume"]) for t in r.json()}
            filtered = [s for s in symbols if vol_map.get(s, 0) >= min_vol]
            removed  = len(symbols) - len(filtered)
            min_in   = min((vol_map.get(s, 0) for s in filtered), default=0)
            print(f"  ✓ Прошло фильтр объёма: {len(filtered)} монет  "
                  f"(убрано {removed}, мин. ${min_in/1e6:.0f}M/24ч)")
            return filtered
        except Exception as e:
            print(f"  ✗ Попытка {attempt+1}/3 — ошибка фильтра объёма: {e}")
            _time.sleep(5)
    return symbols


def _binance_fetch_klines(symbol: str) -> pd.Series | None:
    CHUNK = 1500
    all_rows = []
    chunks_needed = (CANDLES + CHUNK - 1) // CHUNK
    end_time = None
    for chunk_i in range(chunks_needed - 1, -1, -1):
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": CHUNK}
        if end_time is not None:
            params["endTime"] = end_time
        r = req.get(f"{EXCHANGE_URLS['binance']}/fapi/v1/klines", params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
        _tick_request(symbol)
        if not rows:
            break
        all_rows = rows + all_rows
        end_time = int(rows[0][0]) - 1
        if chunk_i > 0:
            _time.sleep(0.05)
    if not all_rows:
        return None
    all_rows = all_rows[:-1][-CANDLES:]
    df = pd.DataFrame(all_rows, columns=[
        "ts","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    df["ts"]    = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df = df.drop_duplicates("ts").sort_values("ts")
    return df.set_index("ts")["close"]


def _binance_fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    url = f"{EXCHANGE_URLS['binance']}/fapi/v1/fundingRate"
    all_rows = []
    limit = 1000
    cur_start = start_ms
    while True:
        try:
            r = req.get(url, params={
                "symbol": symbol, "startTime": cur_start,
                "endTime": end_ms, "limit": limit,
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
        _time.sleep(0.05)
    if not all_rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_rows)
    df["ts"]   = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df.set_index("ts")["rate"].sort_index()


# ─────────────────── OKX ─────────────────────────────────────────────────────

# Маппинг таймфреймов: формат scanerGPU → OKX
_OKX_TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1H", "2h": "2H", "4h": "4H",
    "6h": "6H", "12h": "12H", "1d": "1D", "1w": "1W",
}


def _okx_fetch_symbols() -> list[str]:
    """
    Возвращает список символов OKX USDT-перпетуалов в формате 'BTCUSDT'
    (конвертируем из OKX-формата 'BTC-USDT-SWAP').
    """
    url = f"{EXCHANGE_URLS['okx']}/api/v5/public/instruments"
    r = req.get(url, params={"instType": "SWAP"}, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    symbols = []
    for inst in data:
        inst_id = inst.get("instId", "")
        # Берём только USDT-M: BTC-USDT-SWAP
        if inst_id.endswith("-USDT-SWAP") and inst.get("state") == "live":
            sym = inst_id.replace("-USDT-SWAP", "USDT")   # BTC-USDT-SWAP → BTCUSDT
            symbols.append(sym)
    return sorted(symbols)


def _okx_sym_to_instid(symbol: str) -> str:
    """BTCUSDT → BTC-USDT-SWAP"""
    base = symbol.replace("USDT", "")
    return f"{base}-USDT-SWAP"


def _okx_fetch_volume(symbols: list[str], min_vol: float) -> list[str]:
    """Фильтр по 24h объёму (quoteVolume в USDT)."""
    try:
        url = f"{EXCHANGE_URLS['okx']}/api/v5/market/tickers"
        r = req.get(url, params={"instType": "SWAP"}, timeout=15)
        r.raise_for_status()
        tickers = r.json().get("data", [])
        # volCcy24h — объём в базовой валюте, volCcyQuote24h — в котировочной (USDT)
        vol_map = {}
        for t in tickers:
            inst_id = t.get("instId", "")
            if inst_id.endswith("-USDT-SWAP"):
                sym = inst_id.replace("-USDT-SWAP", "USDT")
                # volCcyQuote24h = объём в USDT за 24ч
                vol_map[sym] = float(t.get("volCcyQuote24h") or t.get("vol24h") or 0)
        filtered = [s for s in symbols if vol_map.get(s, 0) >= min_vol]
        removed  = len(symbols) - len(filtered)
        min_in   = min((vol_map.get(s, 0) for s in filtered), default=0)
        print(f"  ✓ Прошло фильтр объёма: {len(filtered)} монет  "
              f"(убрано {removed}, мин. ${min_in/1e6:.0f}M/24ч)")
        return filtered
    except Exception as e:
        print(f"  ⚠ Фильтр объёма OKX недоступен: {e}")
        return symbols


def _okx_fetch_klines(symbol: str) -> pd.Series | None:
    """
    OKX /api/v5/market/history-candles
    Лимит: 100 свечей за запрос. Пагинация по 'before' (timestamp старейшей свечи).
    """
    CHUNK = 100
    tf    = _OKX_TF_MAP.get(TIMEFRAME, "5m")
    inst_id = _okx_sym_to_instid(symbol)
    all_rows = []
    chunks_needed = (CANDLES + CHUNK - 1) // CHUNK
    after_ts = None   # OKX: after — возвращает свечи СТАРШЕ этого ts

    for _ in range(chunks_needed):
        params = {"instId": inst_id, "bar": tf, "limit": CHUNK}
        if after_ts is not None:
            params["after"] = after_ts
        try:
            r = req.get(f"{EXCHANGE_URLS['okx']}/api/v5/market/history-candles",
                        params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception:
            break
        _tick_request(symbol)
        if not data:
            break
        # OKX возвращает [ts, open, high, low, close, vol, volCcy, ...] — новые сначала
        all_rows = data + all_rows  # добавляем в начало (старые первые)
        after_ts = data[-1][0]      # самая старая свеча в этом чанке
        if len(data) < CHUNK:
            break
        _time.sleep(0.05)

    if not all_rows:
        return None

    all_rows = all_rows[-CANDLES:]
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","vol","volCcy"] + ["x"]*(len(all_rows[0])-7) if len(all_rows[0]) > 7 else ["ts","open","high","low","close","vol","volCcy"])
    df["ts"]    = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df = df.drop_duplicates("ts").sort_values("ts")
    return df.set_index("ts")["close"]


def _okx_fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """OKX /api/v5/public/funding-rate-history"""
    inst_id  = _okx_sym_to_instid(symbol)
    url      = f"{EXCHANGE_URLS['okx']}/api/v5/public/funding-rate-history"
    all_rows = []
    after_ts = None
    while True:
        params = {"instId": inst_id, "limit": 100}
        if after_ts is not None:
            params["after"] = after_ts
        try:
            r = req.get(url, params=params, timeout=10)
            data = r.json().get("data", [])
        except Exception:
            break
        if not data:
            break
        filtered = [d for d in data
                    if start_ms <= int(d["fundingTime"]) <= end_ms]
        all_rows.extend(filtered)
        # Если все свечи уже старше start_ms — стоп
        if int(data[-1]["fundingTime"]) < start_ms:
            break
        after_ts = data[-1]["fundingTime"]
        _time.sleep(0.05)
    if not all_rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_rows)
    df["ts"]   = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df.set_index("ts")["rate"].sort_index()


# ─────────────────── Bybit ───────────────────────────────────────────────────

_BYBIT_TF_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W",
}


def _bybit_fetch_symbols() -> list[str]:
    """Bybit linear perpetuals (USDT-margined)."""
    url = f"{EXCHANGE_URLS['bybit']}/v5/market/instruments-info"
    symbols = []
    cursor  = None
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = req.get(url, params=params, timeout=10)
        r.raise_for_status()
        result = r.json().get("result", {})
        for inst in result.get("list", []):
            sym = inst.get("symbol", "")
            status = inst.get("status", "")
            # Берём только USDT-перпетуалы (не USDC, не инверсные)
            if sym.endswith("USDT") and status == "Trading" and inst.get("contractType") == "LinearPerpetual":
                symbols.append(sym)
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
    return sorted(symbols)


def _bybit_fetch_volume(symbols: list[str], min_vol: float) -> list[str]:
    try:
        url = f"{EXCHANGE_URLS['bybit']}/v5/market/tickers"
        r = req.get(url, params={"category": "linear"}, timeout=15)
        r.raise_for_status()
        tickers = r.json().get("result", {}).get("list", [])
        vol_map = {t["symbol"]: float(t.get("turnover24h") or 0) for t in tickers}
        filtered = [s for s in symbols if vol_map.get(s, 0) >= min_vol]
        removed  = len(symbols) - len(filtered)
        min_in   = min((vol_map.get(s, 0) for s in filtered), default=0)
        print(f"  ✓ Прошло фильтр объёма: {len(filtered)} монет  "
              f"(убрано {removed}, мин. ${min_in/1e6:.0f}M/24ч)")
        return filtered
    except Exception as e:
        print(f"  ⚠ Фильтр объёма Bybit недоступен: {e}")
        return symbols


def _bybit_fetch_klines(symbol: str) -> pd.Series | None:
    """
    Bybit /v5/market/kline
    Лимит: 200 свечей за запрос. Пагинация по end (timestamp в ms).
    """
    CHUNK = 200
    tf    = _BYBIT_TF_MAP.get(TIMEFRAME, "5")
    all_rows = []
    chunks_needed = (CANDLES + CHUNK - 1) // CHUNK
    end_time = None

    for _ in range(chunks_needed):
        params = {"category": "linear", "symbol": symbol,
                  "interval": tf, "limit": CHUNK}
        if end_time is not None:
            params["end"] = end_time
        try:
            r = req.get(f"{EXCHANGE_URLS['bybit']}/v5/market/kline",
                        params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("result", {}).get("list", [])
        except Exception:
            break
        _tick_request(symbol)
        if not data:
            break
        # Bybit возвращает [ts, open, high, low, close, vol, turnover] — новые сначала
        all_rows = data + all_rows
        end_time = int(data[-1][0]) - 1
        if len(data) < CHUNK:
            break
        _time.sleep(0.05)

    if not all_rows:
        return None

    all_rows = all_rows[-CANDLES:]
    rows_parsed = []
    for row in all_rows:
        rows_parsed.append({"ts": int(row[0]), "close": float(row[4])})
    df = pd.DataFrame(rows_parsed)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts")
    return df.set_index("ts")["close"]


def _bybit_fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """Bybit /v5/market/funding/history"""
    url      = f"{EXCHANGE_URLS['bybit']}/v5/market/funding/history"
    all_rows = []
    cursor   = None
    while True:
        params = {"category": "linear", "symbol": symbol, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            r = req.get(url, params=params, timeout=10)
            result = r.json().get("result", {})
            data   = result.get("list", [])
        except Exception:
            break
        if not data:
            break
        filtered = [d for d in data
                    if start_ms <= int(d["fundingRateTimestamp"]) <= end_ms]
        all_rows.extend(filtered)
        if int(data[-1]["fundingRateTimestamp"]) < start_ms:
            break
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
        _time.sleep(0.05)
    if not all_rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_rows)
    df["ts"]   = pd.to_datetime(df["fundingRateTimestamp"].astype(np.int64), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df.set_index("ts")["rate"].sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
#  УНИВЕРСАЛЬНЫЕ ФУНКЦИИ (диспетчер по EXCHANGE)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_symbols() -> list[str]:
    try:
        if EXCHANGE == "binance":
            symbols = _binance_fetch_symbols()
        elif EXCHANGE == "okx":
            symbols = _okx_fetch_symbols()
        elif EXCHANGE == "bybit":
            symbols = _bybit_fetch_symbols()
        else:
            raise ValueError(f"Неизвестная биржа: {EXCHANGE}")
        print(f"  ✓ [{EXCHANGE.upper()}] Найдено {len(symbols)} активных USDT перпетуалов")
        return symbols
    except Exception as e:
        print(f"  ✗ {e} → использую встроенный список")
        return SYMBOLS_MANUAL


def filter_by_volume(symbols: list[str], min_vol: float) -> list[str]:
    if min_vol <= 0:
        return symbols
    if EXCHANGE == "binance":
        return _binance_fetch_volume(symbols, min_vol)
    elif EXCHANGE == "okx":
        return _okx_fetch_volume(symbols, min_vol)
    elif EXCHANGE == "bybit":
        return _bybit_fetch_volume(symbols, min_vol)
    return symbols


def _fetch_klines_for_exchange(symbol: str) -> pd.Series | None:
    """Скачать свечи с выбранной биржи."""
    if EXCHANGE == "binance":
        return _binance_fetch_klines(symbol)
    elif EXCHANGE == "okx":
        return _okx_fetch_klines(symbol)
    elif EXCHANGE == "bybit":
        return _bybit_fetch_klines(symbol)
    return None


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    """Загрузить фандинг с выбранной биржи."""
    if EXCHANGE == "binance":
        return _binance_fetch_funding(symbol, start_ms, end_ms)
    elif EXCHANGE == "okx":
        return _okx_fetch_funding(symbol, start_ms, end_ms)
    elif EXCHANGE == "bybit":
        return _bybit_fetch_funding(symbol, start_ms, end_ms)
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
#  КЕШ (раздельный по биржам)
# ═══════════════════════════════════════════════════════════════════════════════

def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_{TIMEFRAME}_{CANDLES}.parquet"


def cache_valid(symbol: str) -> bool:
    p = cache_path(symbol)
    if not p.exists():
        return False
    age_h = (_time.time() - p.stat().st_mtime) / 3600
    return age_h < CACHE_MAX_AGE_H


def load_cache(symbol: str) -> pd.Series | None:
    try:
        return pd.read_parquet(cache_path(symbol))["close"]
    except Exception:
        return None


def save_cache(symbol: str, series: pd.Series):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        series.to_frame("close").to_parquet(cache_path(symbol))
    except Exception:
        pass




def cache_stats(symbols: list[str]) -> tuple[int, int]:
    fresh = sum(1 for s in symbols if cache_valid(s))
    return fresh, len(symbols) - fresh


# ═══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА СВЕЧЕЙ (с кешем)
# ═══════════════════════════════════════════════════════════════════════════════

_semaphore = None


def fetch_klines(symbol: str, force: bool = False) -> tuple:
    if not force and cache_valid(symbol):
        series = load_cache(symbol)
        if series is not None:
            _tick_request(symbol)
            _tick_symbol()
            return symbol, series

    with _semaphore:
        try:
            series = _fetch_klines_for_exchange(symbol)
            _tick_symbol()

            if series is None or len(series) == 0:
                return symbol, None

            save_cache(symbol, series)
            return symbol, series

        except Exception as e:
            _tick_symbol()
            with _progress_lock:
                print(f"\n  ✗ {symbol}: {e}")
            return symbol, None


def build_logprice_matrix(prices: dict) -> pd.DataFrame:
    # НЕ делаем dropna() глобально — оставляем NaN.
    # Попарная корреляция будет считаться только по общим timestamp.
    df = pd.DataFrame({sym: np.log(s) for sym, s in prices.items()})
    return df


def fast_corr_filter(prices_df: pd.DataFrame, min_corr: float) -> list[tuple]:
    """
    Попарный корреляционный фильтр по пересечению индексов.
    Каждая пара считается независимо — монеты с разными датами листинга
    не портят корреляцию между старыми монетами.
    """
    syms = list(prices_df.columns)
    n = len(syms)
    # Предвычисляем нормализованные векторы для каждого символа
    vecs = {}
    for sym in syms:
        s = prices_df[sym].dropna().values.astype(float)
        if len(s) < 2:
            continue
        s = s - s.mean()
        norm = np.linalg.norm(s)
        if norm == 0:
            continue
        vecs[sym] = s / norm

    pairs = []
    valid_syms = list(vecs.keys())
    for i in range(len(valid_syms)):
        for j in range(i + 1, len(valid_syms)):
            sa, sb = valid_syms[i], valid_syms[j]
            # Пересечение индексов
            idx_a = prices_df[sa].dropna().index
            idx_b = prices_df[sb].dropna().index
            idx   = idx_a.intersection(idx_b)
            if len(idx) < 50:
                continue
            va = prices_df[sa].loc[idx].values.astype(float)
            vb = prices_df[sb].loc[idx].values.astype(float)
            va = va - va.mean(); vb = vb - vb.mean()
            na = np.linalg.norm(va); nb = np.linalg.norm(vb)
            if na == 0 or nb == 0:
                continue
            corr = float(np.dot(va / na, vb / nb))
            if corr >= min_corr:
                pairs.append((sa, sb, corr))
    return pairs


def fast_corr_filter_leg1(prices_df: pd.DataFrame, leg1_syms: list, min_corr: float) -> list[tuple]:
    """
    LEG1-режим: тестирует каждую монету из leg1_syms против ВСЕХ остальных монет.
    Возвращает пары (sym_a, sym_b, corr), где sym_a всегда из leg1_syms.
    Пары между двумя leg1-монетами тоже включаются (без дублей).
    """
    all_syms = list(prices_df.columns)
    leg1_set = set(leg1_syms)

    pairs = []
    seen  = set()

    for sa in leg1_syms:
        if sa not in prices_df.columns:
            continue
        idx_a = prices_df[sa].dropna().index
        va_raw = prices_df[sa].dropna().values.astype(float)
        if len(va_raw) < 50:
            continue

        for sb in all_syms:
            if sb == sa:
                continue
            key = tuple(sorted([sa, sb]))
            if key in seen:
                continue
            seen.add(key)

            idx_b = prices_df[sb].dropna().index
            idx   = idx_a.intersection(idx_b)
            if len(idx) < 50:
                continue

            va = prices_df[sa].loc[idx].values.astype(float)
            vb = prices_df[sb].loc[idx].values.astype(float)
            va = va - va.mean(); vb = vb - vb.mean()
            na = np.linalg.norm(va); nb = np.linalg.norm(vb)
            if na == 0 or nb == 0:
                continue
            corr = float(np.dot(va / na, vb / nb))
            if corr >= min_corr:
                # Всегда ставим leg1-монету первой
                if sa in leg1_set:
                    pairs.append((sa, sb, corr))
                else:
                    pairs.append((sb, sa, corr))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU — CUDA KERNEL + БАТЧЕВЫЙ БЭКТЕСТ
# ═══════════════════════════════════════════════════════════════════════════════



@cuda.jit
def _backtest_kernel(
    z_mat, pa_mat, pb_mat, n_candles,
    out_trades, out_wins, out_pnl, out_max_dd,
):
    pid = cuda.grid(1)
    if pid >= z_mat.shape[0]:
        return

    open_pa     = cuda.local.array(10, numba.float32)
    open_pb     = cuda.local.array(10, numba.float32)
    open_active = cuda.local.array(10, numba.uint8)
    for k in range(10):
        open_active[k] = numba.uint8(0)

    cum_pnl  = numba.float32(0.0)
    peak_pnl = numba.float32(0.0)
    max_dd   = numba.float32(0.0)
    n_trades = numba.int32(0)
    n_wins   = numba.int32(0)

    for i in range(n_candles):
        z = z_mat[pid, i]
        if z != z:
            continue

        pa = pa_mat[pid, i]
        pb = pb_mat[pid, i]

        for dir_idx in range(2):
            if dir_idx == 0:
                should_close = z >= -_CLOSE_ZERO
            else:
                should_close = z <= _CLOSE_ZERO
            if not should_close:
                continue
            gross     = numba.float32(0.0)
            exit_comm = numba.float32(0.0)
            has_pos   = numba.uint8(0)
            for li in range(_N_LEVELS):
                slot = dir_idx * _N_LEVELS + li
                if open_active[slot] == 0:
                    continue
                has_pos = numba.uint8(1)
                half = numba.float32(_TSIZE * 0.5)
                ra = (pa - open_pa[slot]) / open_pa[slot]
                rb = (pb - open_pb[slot]) / open_pb[slot]
                if dir_idx == 0:
                    gross += ra * half - rb * half
                else:
                    gross += -ra * half + rb * half
                exit_comm += numba.float32(_TSIZE * _COMM * 2.0)
                open_active[slot] = numba.uint8(0)
            if has_pos:
                trade_pnl = gross - exit_comm
                cum_pnl  += trade_pnl
                n_trades += numba.int32(1)
                if trade_pnl > numba.float32(0.0):
                    n_wins += numba.int32(1)

        for li in range(_N_LEVELS):
            lv = numba.float32(_LEVELS[li])
            if z <= -lv:
                dir_idx = 0
            elif z >= lv:
                dir_idx = 1
            else:
                continue
            slot = dir_idx * _N_LEVELS + li
            if open_active[slot] != 0:
                continue
            open_pa[slot]     = pa
            open_pb[slot]     = pb
            open_active[slot] = numba.uint8(1)
            cum_pnl -= numba.float32(_TSIZE * _COMM)

        # Нереализованный PnL по открытым позициям (mark-to-market для просадки)
        unrealized = numba.float32(0.0)
        for dir_idx in range(2):
            for li in range(_N_LEVELS):
                slot = dir_idx * _N_LEVELS + li
                if open_active[slot] == 0:
                    continue
                half = numba.float32(_TSIZE * 0.5)
                ra = (pa - open_pa[slot]) / open_pa[slot]
                rb = (pb - open_pb[slot]) / open_pb[slot]
                if dir_idx == 0:
                    unrealized += ra * half - rb * half
                else:
                    unrealized += -ra * half + rb * half

        total_equity = cum_pnl + unrealized   # реальная эквити с открытыми
        if total_equity > peak_pnl:
            peak_pnl = total_equity
        dd = total_equity - peak_pnl
        if dd < max_dd:
            max_dd = dd

    out_trades[pid] = n_trades
    out_wins[pid]   = n_wins
    out_pnl[pid]    = cum_pnl
    out_max_dd[pid] = max_dd

def gpu_rolling_zscore(vals_a: np.ndarray, vals_b: np.ndarray, window: int) -> tuple:
    n_pairs, n_candles = vals_a.shape
    ca_gpu = cp.asarray(vals_a, dtype=cp.float32)
    cb_gpu = cp.asarray(vals_b, dtype=cp.float32)
    log_a  = cp.log(ca_gpu)
    log_b  = cp.log(cb_gpu)
    spread = log_a - log_b

    # Определяем NaN-паддинг (короткие пары выровнены вправо)
    nan_mask  = cp.isnan(spread)
    # Индекс первого валидного элемента в каждой строке
    first_valid = cp.argmax(~nan_mask, axis=1)  # (n_pairs,)

    # Заменяем NaN нулём для cumsum (потом маскируем результат)
    spread_clean = cp.where(nan_mask, cp.float32(0.0), spread)

    cumsum  = cp.cumsum(spread_clean, axis=1)
    cumsum2 = cp.cumsum(spread_clean ** 2, axis=1)

    roll_sum  = cp.empty_like(spread)
    roll_sum2 = cp.empty_like(spread)
    roll_sum[:, :window - 1]  = cp.nan
    roll_sum2[:, :window - 1] = cp.nan
    roll_sum[:, window - 1]   = cumsum[:, window - 1]
    roll_sum2[:, window - 1]  = cumsum2[:, window - 1]
    if n_candles > window:
        roll_sum[:, window:]  = cumsum[:, window:] - cumsum[:, :n_candles - window]
        roll_sum2[:, window:] = cumsum2[:, window:] - cumsum2[:, :n_candles - window]

    rm  = roll_sum / window
    var = roll_sum2 / window - rm ** 2
    var = cp.maximum(var, 0)
    rs  = cp.sqrt(var)
    rs  = cp.where(rs < 1e-10, cp.nan, rs)
    z   = (spread - rm) / rs

    # Маскируем: NaN там где паддинг, и где rolling-окно захватывает паддинг
    col_idx       = cp.arange(n_candles)[cp.newaxis, :]   # (1, n_candles)
    min_valid_col = first_valid[:, cp.newaxis] + (window - 1)  # (n_pairs, 1)
    z = cp.where(col_idx < min_valid_col, cp.nan, z)

    return z, ca_gpu, cb_gpu


def gpu_compute_metrics(out_trades, out_wins, out_pnl, out_max_dd):
    results = []
    for i in range(len(out_trades)):
        n_t = int(out_trades[i])
        if n_t == 0:
            results.append(None)
            continue
        n_w = int(out_wins[i])
        pnl = float(out_pnl[i])
        wr  = round(n_w / n_t * 100, 1)
        results.append({
            "total_trades":  n_t,
            "win_rate":      wr,
            "total_pnl":     round(pnl, 2),
            "avg_pnl":       round(pnl / n_t, 2),
            "max_dd":        round(float(out_max_dd[i]), 2),
            "sharpe":        0.0,
            "profit_factor": np.inf,
        })
    return results


def run_gpu_batch(candidate_pairs, prices, windows, batch_size=4096):
    """
    Запускает GPU-бэктест для каждого окна из списка windows.
    Возвращает список записей — одна на каждую (пара, window), прошедшую фильтры.

    Матрица данных строится ПОБАТЧЕВО, чтобы избежать выделения
    всего массива (n_total × max_len) сразу — при 52k пар × 100k свечей
    это ~19 ГиБ, что вызывает MemoryError.
    """
    if isinstance(windows, int):
        windows = [windows]

    valid_pairs = [(a, b, c) for a, b, c in candidate_pairs if a in prices and b in prices]
    n_total = len(valid_pairs)
    if n_total == 0:
        return []

    # Максимальная длина пересечения индексов по всем парам (≤ CANDLES)
    max_len = min(
        max(len(prices[a].index.intersection(prices[b].index)) for a, b, _ in valid_pairs),
        CANDLES,
    )

    print(f"  GPU: {n_total:,} пар  |  до {max_len} свечей  |  батч {batch_size}  |  окна {windows}")

    all_results = []

    for window in windows:
        print(f"  → Window {window}...")
        for batch_start in range(0, n_total, batch_size):
            batch_slice = valid_pairs[batch_start: batch_start + batch_size]
            bp = len(batch_slice)

            # ── Строим матрицы только для текущего батча ──────────
            vals_a = np.full((bp, max_len), np.nan, dtype=np.float32)
            vals_b = np.full((bp, max_len), np.nan, dtype=np.float32)
            for local_k, (sym_a, sym_b, _) in enumerate(batch_slice):
                idx = prices[sym_a].index.intersection(prices[sym_b].index)[-max_len:]
                n   = len(idx)
                vals_a[local_k, max_len - n:] = prices[sym_a].loc[idx].values.astype(np.float32)
                vals_b[local_k, max_len - n:] = prices[sym_b].loc[idx].values.astype(np.float32)

            z_gpu, pa_gpu, pb_gpu = gpu_rolling_zscore(vals_a, vals_b, window)
            z_np  = cp.asnumpy(z_gpu).astype(np.float32)
            pa_np = cp.asnumpy(pa_gpu).astype(np.float32)
            pb_np = cp.asnumpy(pb_gpu).astype(np.float32)

            # Освобождаем GPU-память до запуска ядра
            del z_gpu, pa_gpu, pb_gpu
            cp.get_default_memory_pool().free_all_blocks()

            z_dev  = cuda.to_device(z_np)
            pa_dev = cuda.to_device(pa_np)
            pb_dev = cuda.to_device(pb_np)
            out_trades = cuda.to_device(np.zeros(bp, dtype=np.int32))
            out_wins   = cuda.to_device(np.zeros(bp, dtype=np.int32))
            out_pnl    = cuda.to_device(np.zeros(bp, dtype=np.float32))
            out_max_dd = cuda.to_device(np.zeros(bp, dtype=np.float32))

            threads_per_block = 128
            blocks = (bp + threads_per_block - 1) // threads_per_block
            _backtest_kernel[blocks, threads_per_block](
                z_dev, pa_dev, pb_dev, max_len,
                out_trades, out_wins, out_pnl, out_max_dd,
            )
            cuda.synchronize()

            h_trades = out_trades.copy_to_host()
            h_wins   = out_wins.copy_to_host()
            h_pnl    = out_pnl.copy_to_host()
            h_max_dd = out_max_dd.copy_to_host()
            metrics_list = gpu_compute_metrics(h_trades, h_wins, h_pnl, h_max_dd)

            for local_k, (sym_a, sym_b, corr) in enumerate(batch_slice):
                bt = metrics_list[local_k]
                if bt is None:
                    continue
                if bt["win_rate"] < MIN_WINRATE or bt["total_pnl"] < MIN_PNL:
                    continue
                if bt["max_dd"] < MAX_DRAWDOWN:
                    continue
                all_results.append((sym_a, sym_b, window, corr, bt))

            pct = min(batch_start + batch_size, n_total) / n_total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    [{bar}] {pct:5.1f}%  батч {batch_start // batch_size + 1}  найдено: {len(all_results)}", end="\r")
        print()

    print(f"  GPU готов: {len(all_results)} записей (пара × window) прошли фильтры")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  АНАЛИЗ ПАРЫ
# ═══════════════════════════════════════════════════════════════════════════════

def calc_correlation(close_a, close_b):
    idx = close_a.index.intersection(close_b.index)
    la  = np.log(close_a.loc[idx].values)
    lb  = np.log(close_b.loc[idx].values)
    return float(np.corrcoef(la, lb)[0, 1])


def calc_cointegration(close_a, close_b):
    idx = close_a.index.intersection(close_b.index)
    ca  = close_a.loc[idx].values
    cb  = close_b.loc[idx].values
    score, pvalue, _ = coint(np.log(ca), np.log(cb))
    return float(pvalue), float(score)


def calc_halflife(close_a, close_b, window):
    idx    = close_a.index.intersection(close_b.index)
    spread = np.log(close_a.loc[idx]) - np.log(close_b.loc[idx])
    rm = spread.rolling(window).mean()
    rs = spread.rolling(window).std(ddof=0)
    z  = ((spread - rm) / rs).dropna()
    if len(z) < 50:
        return np.nan
    dz   = z.diff().dropna()
    z_   = z.shift(1).dropna()
    idx2 = dz.index.intersection(z_.index)
    dz   = dz.loc[idx2].values
    z_   = z_.loc[idx2].values
    X = np.column_stack([z_, np.ones(len(z_))])
    try:
        beta = np.linalg.lstsq(X, dz, rcond=None)[0]
        lam  = beta[0]
        if lam >= 0:
            return np.nan
        return float(-np.log(2) / lam)
    except Exception:
        return np.nan


def backtest(close_a, close_b, window, funding_a=None, funding_b=None):
    idx    = close_a.index.intersection(close_b.index)
    ca     = close_a.loc[idx]
    cb     = close_b.loc[idx]
    spread = np.log(ca.values) - np.log(cb.values)

    rm = np.empty(len(spread)); rm[:] = np.nan
    rs = np.empty(len(spread)); rs[:] = np.nan
    for k in range(window - 1, len(spread)):
        sl = spread[k - window + 1: k + 1]
        rm[k] = sl.mean()
        rs[k] = sl.std(ddof=DDOF)

    valid = ~np.isnan(rm) & (rs > 0)
    if valid.sum() < window + 10:
        return None

    z_full  = np.where(valid, (spread - rm) / rs, np.nan)
    start   = np.argmax(valid)
    z_arr   = z_full[start:]
    pa_arr  = ca.values[start:]
    pb_arr  = cb.values[start:]
    ts_arr  = ca.index[start:]
    n       = len(z_arr)
    if n < 10:
        return None

    f_dict_a = {}
    f_dict_b = {}
    if funding_a is not None and len(funding_a) > 0:
        f_dict_a = {ts: rate for ts, rate in funding_a.items()}
    if funding_b is not None and len(funding_b) > 0:
        f_dict_b = {ts: rate for ts, rate in funding_b.items()}
    all_funding_ts = set(f_dict_a.keys()) | set(f_dict_b.keys())

    closed_trades  = []
    cumulative_pnl = 0.0
    equity         = np.empty(n)
    equity_total   = np.empty(n)
    unrealized_a_arr = np.zeros(n)
    unrealized_b_arr = np.zeros(n)
    open_state  = {1: {}, -1: {}}
    open_trades = {1: [], -1: []}
    levels_arr = np.array(GRID_LEVELS)

    for i in range(n):
        z_val = z_arr[i]
        if np.isnan(z_val):
            equity[i] = cumulative_pnl
            equity_total[i] = cumulative_pnl
            continue
        pa = pa_arr[i]
        pb = pb_arr[i]
        ts = ts_arr[i]

        if ts in all_funding_ts:
            rate_a = f_dict_a.get(ts, 0.0)
            rate_b = f_dict_b.get(ts, 0.0)
            for direction in (1, -1):
                for (ep_a, ep_b, size, _) in open_trades[direction]:
                    half = size / 2.0
                    funding_cost = (half * rate_a - half * rate_b) if direction == 1 \
                                   else (-half * rate_a + half * rate_b)
                    cumulative_pnl -= funding_cost

        for direction in (1, -1):
            if not open_trades[direction]:
                continue
            should_close = (direction == 1 and z_val >= -CLOSE_AT_ZERO) or \
                           (direction == -1 and z_val <= CLOSE_AT_ZERO)
            if not should_close:
                continue
            gross = exit_comm = 0.0
            for (ep_a, ep_b, size, _) in open_trades[direction]:
                half = size / 2
                ra = (pa - ep_a) / ep_a
                rb = (pb - ep_b) / ep_b
                gross     += (ra * half - rb * half) if direction == 1 else (-ra * half + rb * half)
                exit_comm += size * COMMISSION * 2
            trade_pnl = gross - exit_comm
            cumulative_pnl += trade_pnl
            closed_trades.append(trade_pnl)
            open_trades[direction] = []
            open_state[direction]  = {}

        for li, level in enumerate(levels_arr):
            direction = 0
            if   z_val <= -level: direction =  1
            elif z_val >=  level: direction = -1
            if direction == 0 or li in open_state[direction]:
                continue
            open_trades[direction].append((pa, pb, TRADE_SIZE, li))
            open_state[direction][li] = True
            cumulative_pnl -= TRADE_SIZE * COMMISSION

        # Нереализованный PnL по открытым позициям (mark-to-market)
        unrealized = unrealized_a = unrealized_b = 0.0
        for direction in (1, -1):
            for (ep_a, ep_b, size, _) in open_trades[direction]:
                half  = size / 2.0
                ra    = (pa - ep_a) / ep_a
                rb    = (pb - ep_b) / ep_b
                la    =  ra * half if direction == 1 else -ra * half
                lb    = -rb * half if direction == 1 else  rb * half
                unrealized_a += la
                unrealized_b += lb
                unrealized   += la + lb
        equity[i]        = cumulative_pnl
        equity_total[i]  = cumulative_pnl + unrealized
        unrealized_a_arr[i] = unrealized_a
        unrealized_b_arr[i] = unrealized_b

    if not closed_trades:
        return None

    pnls      = np.array(closed_trades)
    wins      = pnls[pnls > 0]
    losses    = pnls[pnls < 0]
    eq_ts     = pd.Series(equity,       index=ts_arr)
    eq_tot_ts = pd.Series(equity_total, index=ts_arr)
    max_dd      = float((eq_ts     - eq_ts.cummax()    ).min())   # только закрытые
    max_dd_real = float((eq_tot_ts - eq_tot_ts.cummax()).min())   # с открытыми (реальная)
    min_leg_a   = float(pd.Series(unrealized_a_arr, index=ts_arr).min())
    min_leg_b   = float(pd.Series(unrealized_b_arr, index=ts_arr).min())
    daily_ret = eq_tot_ts.resample("1D").last().diff().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    total_pnl = float(cumulative_pnl)
    pf        = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.inf

    return {
        "total_trades":  len(pnls),
        "win_rate":      round(len(wins) / len(pnls) * 100, 1),
        "total_pnl":     round(total_pnl, 2),
        "avg_pnl":       round(total_pnl / len(pnls), 2),
        "max_dd":        round(max_dd_real, 2),        # ← реальная просадка (фильтр)
        "max_dd_closed": round(max_dd, 2),             # ← только закрытые (справочно)
        "min_leg_a":     round(min_leg_a, 2),          # ← макс. просадка ноги A
        "min_leg_b":     round(min_leg_b, 2),          # ← макс. просадка ноги B
        "sharpe":        round(sharpe, 3),
        "profit_factor": round(pf, 2),
    }


def backtest_multiwindow(close_a, close_b, windows, funding_a=None, funding_b=None):
    idx = close_a.index.intersection(close_b.index)
    ca  = close_a.loc[idx]
    cb  = close_b.loc[idx]
    spread = np.log(ca.values) - np.log(cb.values)
    n      = len(spread)
    cumsum  = np.cumsum(spread)
    cumsum2 = np.cumsum(spread ** 2)
    results: dict = {}

    for window in windows:
        if n < window + 50:
            results[window] = None
            continue

        rm = np.empty(n); rm[:] = np.nan
        rs = np.empty(n); rs[:] = np.nan
        end_idx   = np.arange(window - 1, n)
        start_idx = end_idx - window + 1
        s1 = cumsum[end_idx]
        s1[start_idx > 0] -= cumsum[start_idx[start_idx > 0] - 1]
        s2 = cumsum2[end_idx]
        s2[start_idx > 0] -= cumsum2[start_idx[start_idx > 0] - 1]
        mean_w = s1 / window
        var_w  = s2 / window - mean_w ** 2
        var_w  = np.maximum(var_w, 0.0)
        std_w  = np.sqrt(var_w)
        rm[window - 1:] = mean_w
        rs[window - 1:] = std_w

        valid = ~np.isnan(rm) & (rs > 0)
        if valid.sum() < window + 10:
            results[window] = None
            continue

        z_full  = np.where(valid, (spread - rm) / rs, np.nan)
        start   = np.argmax(valid)
        z_arr   = z_full[start:]
        pa_arr  = ca.values[start:]
        pb_arr  = cb.values[start:]
        ts_arr  = ca.index[start:]
        nn      = len(z_arr)
        if nn < 10:
            results[window] = None
            continue

        f_dict_a = {}
        f_dict_b = {}
        if funding_a is not None and len(funding_a) > 0:
            f_dict_a = {ts: rate for ts, rate in funding_a.items()}
        if funding_b is not None and len(funding_b) > 0:
            f_dict_b = {ts: rate for ts, rate in funding_b.items()}
        all_funding_ts = set(f_dict_a.keys()) | set(f_dict_b.keys())

        closed_trades  = []
        cumulative_pnl = 0.0
        equity         = np.empty(nn)
        equity_total   = np.empty(nn)
        unrealized_a_arr = np.zeros(nn)
        unrealized_b_arr = np.zeros(nn)
        open_state     = {1: {}, -1: {}}
        open_trades_w  = {1: [], -1: []}
        levels_arr     = np.array(GRID_LEVELS)

        for i in range(nn):
            z_val = z_arr[i]
            if np.isnan(z_val):
                equity[i] = cumulative_pnl
                equity_total[i] = cumulative_pnl
                continue
            pa = pa_arr[i]
            pb = pb_arr[i]
            ts = ts_arr[i]

            if ts in all_funding_ts:
                rate_a = f_dict_a.get(ts, 0.0)
                rate_b = f_dict_b.get(ts, 0.0)
                for direction in (1, -1):
                    for (ep_a, ep_b, size, _) in open_trades_w[direction]:
                        half = size / 2.0
                        funding_cost = (half * rate_a - half * rate_b) if direction == 1 \
                                       else (-half * rate_a + half * rate_b)
                        cumulative_pnl -= funding_cost

            for direction in (1, -1):
                if not open_trades_w[direction]:
                    continue
                should_close = (direction == 1 and z_val >= -CLOSE_AT_ZERO) or \
                               (direction == -1 and z_val <= CLOSE_AT_ZERO)
                if not should_close:
                    continue
                gross = exit_comm = 0.0
                for (ep_a, ep_b, size, _) in open_trades_w[direction]:
                    half = size / 2
                    ra = (pa - ep_a) / ep_a
                    rb = (pb - ep_b) / ep_b
                    gross     += (ra * half - rb * half) if direction == 1 \
                                 else (-ra * half + rb * half)
                    exit_comm += size * COMMISSION * 2
                trade_pnl = gross - exit_comm
                cumulative_pnl += trade_pnl
                closed_trades.append(trade_pnl)
                open_trades_w[direction] = []
                open_state[direction]    = {}

            for li, level in enumerate(levels_arr):
                direction = 0
                if   z_val <= -level: direction =  1
                elif z_val >=  level: direction = -1
                if direction == 0 or li in open_state[direction]:
                    continue
                open_trades_w[direction].append((pa, pb, TRADE_SIZE, li))
                open_state[direction][li] = True
                cumulative_pnl -= TRADE_SIZE * COMMISSION

            # Нереализованный PnL по открытым позициям (mark-to-market)
            unrealized = unrealized_a = unrealized_b = 0.0
            for direction in (1, -1):
                for (ep_a, ep_b, size, _) in open_trades_w[direction]:
                    half = size / 2.0
                    ra   = (pa - ep_a) / ep_a
                    rb   = (pb - ep_b) / ep_b
                    la   =  ra * half if direction == 1 else -ra * half
                    lb   = -rb * half if direction == 1 else  rb * half
                    unrealized_a += la
                    unrealized_b += lb
                    unrealized   += la + lb
            equity[i]           = cumulative_pnl
            equity_total[i]     = cumulative_pnl + unrealized
            unrealized_a_arr[i] = unrealized_a
            unrealized_b_arr[i] = unrealized_b

        if not closed_trades:
            results[window] = None
            continue

        pnls      = np.array(closed_trades)
        wins      = pnls[pnls > 0]
        losses    = pnls[pnls < 0]
        eq_ts     = pd.Series(equity,       index=ts_arr)
        eq_tot_ts = pd.Series(equity_total, index=ts_arr)
        max_dd      = float((eq_ts     - eq_ts.cummax()    ).min())
        max_dd_real = float((eq_tot_ts - eq_tot_ts.cummax()).min())
        min_leg_a   = float(pd.Series(unrealized_a_arr, index=ts_arr).min())
        min_leg_b   = float(pd.Series(unrealized_b_arr, index=ts_arr).min())
        daily_ret = eq_tot_ts.resample("1D").last().diff().dropna()
        sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) \
                    if daily_ret.std() > 0 else 0.0
        total_pnl = float(cumulative_pnl)
        pf        = float(wins.sum() / abs(losses.sum())) \
                    if len(losses) > 0 and losses.sum() != 0 else np.inf

        results[window] = {
            "total_trades":  len(pnls),
            "win_rate":      round(len(wins) / len(pnls) * 100, 1),
            "total_pnl":     round(total_pnl, 2),
            "avg_pnl":       round(total_pnl / len(pnls), 2),
            "max_dd":        round(max_dd_real, 2),
            "max_dd_closed": round(max_dd, 2),
            "min_leg_a":     round(min_leg_a, 2),
            "min_leg_b":     round(min_leg_b, 2),
            "sharpe":        round(sharpe, 3),
            "profit_factor": round(pf, 2),
        }

    results["_windows"] = list(windows)
    return results


def analyze_pair_multiwindow(args: tuple) -> list:
    sym_a, sym_b, vals_a, idx_a, vals_b, idx_b = args[:6]
    funding_a = args[6] if len(args) > 6 else None
    funding_b = args[7] if len(args) > 7 else None
    windows   = args[8] if len(args) > 8 else WINDOWS_TO_TEST

    ca_full = pd.Series(vals_a, index=idx_a)
    cb_full = pd.Series(vals_b, index=idx_b)
    idx = ca_full.index.intersection(cb_full.index)
    min_window = min(windows)
    if len(idx) < min_window + 50:
        return []
    ca = ca_full.loc[idx]
    cb = cb_full.loc[idx]

    corr = calc_correlation(ca, cb)
    unique_ratio_a = round(ca.nunique() / len(ca), 4)
    unique_ratio_b = round(cb.nunique() / len(cb), 4)

    try:
        pvalue, _ = calc_cointegration(ca, cb)
    except Exception:
        return []

    if pvalue > MAX_COINT_PVALUE:
        return []

    mw = backtest_multiwindow(ca, cb, windows, funding_a, funding_b)
    rows = []
    for w in windows:
        bt = mw.get(w)
        if bt is None:
            continue
        if bt["win_rate"] < MIN_WINRATE or bt["total_pnl"] < MIN_PNL:
            continue
        if bt["max_dd"] < MAX_DRAWDOWN:
            continue
        hl = calc_halflife(ca, cb, w)
        sc = score_pair(corr, pvalue, hl, bt)
        rows.append({
            "symbol_a":       sym_a,
            "symbol_b":       sym_b,
            "window":         w,
            "score":          sc,
            "correlation":    round(corr, 4),
            "coint_pvalue":   round(pvalue, 4),
            "halflife":       round(hl, 1) if not np.isnan(hl) else None,
            "trades":         bt["total_trades"],
            "win_rate":       bt["win_rate"],
            "total_pnl":      bt["total_pnl"],
            "avg_pnl":        bt["avg_pnl"],
            "max_dd":         bt["max_dd"],
            "sharpe":         bt["sharpe"],
            "profit_factor":  bt["profit_factor"],
            "unique_ratio_a": unique_ratio_a,
            "unique_ratio_b": unique_ratio_b,
        })
    return rows


_COINT_MAX_LEN = 3000   # Используем не более 3000 последних свечей для коинтеграции
                        # statsmodels.coint масштабируется как O(n^2) — на 30k точках зависает

def _cpu_coint_halflife(args: tuple) -> tuple:
    import os
    sym_a, sym_b, vals_a, idx_a, vals_b, idx_b = args
    ca = pd.Series(vals_a, index=idx_a)
    cb = pd.Series(vals_b, index=idx_b)
    idx = ca.index.intersection(cb.index)
    ca  = ca.loc[idx]
    cb  = cb.loc[idx]

    # Обрезаем до _COINT_MAX_LEN последних точек — достаточно для коинтеграции
    if len(ca) > _COINT_MAX_LEN:
        ca = ca.iloc[-_COINT_MAX_LEN:]
        cb = cb.iloc[-_COINT_MAX_LEN:]

    # Подавляем шум LAPACK через переменную окружения (без dup/dup2 и утечки дескрипторов)
    old_env = os.environ.get("OPENBLAS_VERBOSE", None)
    os.environ["OPENBLAS_VERBOSE"] = "0"
    try:
        try:
            pvalue, _ = calc_cointegration(ca, cb)
        except Exception:
            pvalue = 1.0
        hl = calc_halflife(ca, cb, WINDOW)
        return sym_a, sym_b, pvalue, hl
    finally:
        if old_env is None:
            os.environ.pop("OPENBLAS_VERBOSE", None)
        else:
            os.environ["OPENBLAS_VERBOSE"] = old_env


def analyze_pair(args: tuple) -> dict | None:
    sym_a, sym_b, vals_a, idx_a, vals_b, idx_b = args[:6]
    funding_a = args[6] if len(args) > 6 else None
    funding_b = args[7] if len(args) > 7 else None

    ca_full = pd.Series(vals_a, index=idx_a)
    cb_full = pd.Series(vals_b, index=idx_b)
    idx = ca_full.index.intersection(cb_full.index)
    if len(idx) < WINDOW + 50:
        return None
    ca = ca_full.loc[idx]
    cb = cb_full.loc[idx]

    corr = calc_correlation(ca, cb)
    unique_ratio_a = round(ca.nunique() / len(ca), 4)
    unique_ratio_b = round(cb.nunique() / len(cb), 4)

    try:
        pvalue, _ = calc_cointegration(ca, cb)
    except Exception:
        return None

    hl = calc_halflife(ca, cb, WINDOW)
    bt = backtest(ca, cb, WINDOW, funding_a, funding_b)
    if bt is None:
        return None
    if bt["win_rate"] < MIN_WINRATE or bt["total_pnl"] < MIN_PNL:
        return None
    if pvalue > MAX_COINT_PVALUE:
        return None
    if bt["max_dd"] < MAX_DRAWDOWN:
        return None

    sc = score_pair(corr, pvalue, hl, bt)
    return {
        "symbol_a":      sym_a,
        "symbol_b":      sym_b,
        "score":         sc,
        "correlation":   round(corr, 4),
        "coint_pvalue":  round(pvalue, 4),
        "halflife":      round(hl, 1) if not np.isnan(hl) else None,
        "trades":        bt["total_trades"],
        "win_rate":      bt["win_rate"],
        "total_pnl":     bt["total_pnl"],
        "avg_pnl":       bt["avg_pnl"],
        "max_dd":        bt["max_dd"],
        "sharpe":        bt["sharpe"],
        "profit_factor": bt["profit_factor"],
        "unique_ratio_a": unique_ratio_a,
        "unique_ratio_b": unique_ratio_b,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  СКОРИНГ
# ═══════════════════════════════════════════════════════════════════════════════

def score_pair(corr, pvalue, halflife, bt) -> float:
    s = 0.0
    s += min(corr, 1.0) * 25
    s += max(0, (1 - pvalue / 0.05)) * 25
    if not np.isnan(halflife):
        optimal = 20
        dist = abs(halflife - optimal)
        s += max(0, 20 - dist * 0.3)
    if bt:
        s += min(bt["win_rate"] / 100, 1.0) * 10
        s += min(max(bt["sharpe"], 0), 3) / 3 * 10
        s += min(max(bt["profit_factor"], 0), 3) / 3 * 5
        s += 5 if bt["total_pnl"] > 0 else 0
    return round(s, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def _run_scan_for_window(window, prices, fundings):
    """
    Запускает шаги 2-4 (корреляция → анализ → сохранение) для одного window.
    Возвращает имя сохранённого файла или None.
    """
    global WINDOW
    WINDOW = window   # обновляем глобал — используется в backtest/analyze_pair
    multi_window = False
    windows_list = [window]
    t0 = _time.time()

    # ── 2. Предфильтр корреляций ─────────────────────────────────
    leg1_mode = bool(SYMBOLS_LEG1)
    if leg1_mode:
        print(f"\n[2/4] LEG1-режим: корреляции монет {SYMBOLS_LEG1} × все биржевые монеты (>= {MIN_CORRELATION})...")
    else:
        print(f"\n[2/4] Предфильтр корреляций >= {MIN_CORRELATION} (Pearson log-price, numpy)...")
    t0 = _time.time()
    # Убираем монеты с коротким рядом ДО построения матрицы.
    # Монеты с малым количеством свечей обрезают общий индекс через dropna()
    # и портят корреляцию между старыми монетами (они считаются на коротком периоде).
    min_candles_for_corr = window + 50
    prices_for_corr = {s: v for s, v in prices.items() if len(v) >= min_candles_for_corr}
    dropped = len(prices) - len(prices_for_corr)
    if dropped:
        print(f"  Исключено из корреляции: {dropped} монет с < {min_candles_for_corr} свечей")
    logprice_df = build_logprice_matrix(prices_for_corr)

    if leg1_mode:
        # Проверяем, что LEG1-монеты присутствуют в загруженных данных
        leg1_available = [s for s in SYMBOLS_LEG1 if s in prices_for_corr]
        leg1_missing   = [s for s in SYMBOLS_LEG1 if s not in prices_for_corr]
        if leg1_missing:
            print(f"  ⚠ LEG1-монеты не найдены/не загружены: {leg1_missing}")
        if not leg1_available:
            print("  ✗ Ни одна LEG1-монета не доступна — прерываю скан.")
            return
        print(f"  LEG1-монеты: {leg1_available}  |  Всего монет для сравнения: {len(prices_for_corr)}")
        candidate_pairs = fast_corr_filter_leg1(logprice_df, leg1_available, MIN_CORRELATION)
        total_possible  = len(leg1_available) * (len(prices_for_corr) - 1)
    else:
        candidate_pairs = fast_corr_filter(logprice_df, MIN_CORRELATION)
        total_possible  = len(prices) * (len(prices) - 1) // 2

    elapsed = _time.time() - t0
    pct = len(candidate_pairs) / max(total_possible, 1) * 100
    print(f"  ✓ Прошло: {len(candidate_pairs):,} из {total_possible:,} пар ({pct:.1f}%) за {elapsed:.1f}s")

    # ── 3. Анализ ────────────────────────────────────────────────
    mode_str = f"GPU (CuPy + CUDA kernel)" if _gpu_available else f"{ANALYSIS_WORKERS} CPU процессов"
    print(f"\n[3/4] Коинтеграция + Half-Life + Бэктест  ({mode_str})...")

    pairs_todo = [
        (sym_a, sym_b, corr)
        for sym_a, sym_b, corr in candidate_pairs
        if sym_a in prices and sym_b in prices
    ]
    print(f"  Задач: {len(pairs_todo):,}")

    results = []
    t0 = _time.time()

    if len(pairs_todo) > 0:
        if _gpu_available:
            gpu_found = run_gpu_batch(pairs_todo, prices, windows_list, batch_size=GPU_BATCH_SIZE)
            print(f"  GPU бэктест готов: {len(gpu_found)} записей прошли фильтры")
            if USE_FUNDING:
                print(f"  Пересчёт с фандингом + коинтеграция + half-life на CPU ({ANALYSIS_WORKERS} потоков)...")
            else:
                print(f"  Считаю коинтеграцию + half-life на CPU ({ANALYSIS_WORKERS} потоков)...")

            # Уникальные пары для CPU-расчёта коинтеграции
            unique_pairs = list({(a, b): (a, b, corr) for a, b, w, corr, bt in gpu_found}.values())
            coint_cache = {}  # {(sym_a, sym_b): (pvalue, hl_by_window)}

            coint_args = [
                (sym_a, sym_b,
                 prices[sym_a].values, prices[sym_a].index,
                 prices[sym_b].values, prices[sym_b].index)
                for sym_a, sym_b, _ in unique_pairs
            ]
            done  = 0
            total = len(coint_args)
            t_coint = _time.time()
            print(f"  Коинтеграция: {total} пар  (макс. {_COINT_MAX_LEN} свечей/пара, {ANALYSIS_WORKERS} потоков)")
            with ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
                futs = {executor.submit(_cpu_coint_halflife, args): args[:2] for args in coint_args}
                for future in as_completed(futs):
                    done += 1
                    try:
                        sym_a, sym_b, pvalue, hl = future.result()
                        coint_cache[(sym_a, sym_b)] = (pvalue, hl)
                    except Exception:
                        pass
                    if done % 10 == 0 or done == total:
                        pct_c = done / total * 100
                        bar_c = "█" * int(pct_c / 5) + "░" * (20 - int(pct_c / 5))
                        elapsed_c = _time.time() - t_coint
                        eta_c = (elapsed_c / done) * (total - done) if done > 0 else 0
                        spd_c = done / elapsed_c if elapsed_c > 0 else 0
                        print(f"  [{bar_c}] {pct_c:5.1f}%  {done}/{total}  "
                              f"{spd_c:.1f} пар/с  ETA: {eta_c:.0f}s   ", end="\r")
            elapsed_c = _time.time() - t_coint
            print(f"\n  ✓ Коинтеграция готова за {elapsed_c:.1f}s")

            # Формируем строки: одна на каждую (пара, window)
            for sym_a, sym_b, window, corr, bt in gpu_found:
                pvalue, hl = coint_cache.get((sym_a, sym_b), (1.0, np.nan))
                if pvalue > MAX_COINT_PVALUE:
                    continue

                if USE_FUNDING:
                    ca = pd.Series(prices[sym_a].values, index=prices[sym_a].index)
                    cb = pd.Series(prices[sym_b].values, index=prices[sym_b].index)
                    bt = backtest(ca, cb, window, fundings.get(sym_a), fundings.get(sym_b))
                    if bt is None:
                        continue
                    if bt["win_rate"] < MIN_WINRATE or bt["total_pnl"] < MIN_PNL:
                        continue
                    if bt["max_dd"] < MAX_DRAWDOWN:
                        continue

                sc = score_pair(corr, pvalue, hl, bt)
                key = (sym_a, sym_b, window)
                r = {
                    "symbol_a":       sym_a,
                    "symbol_b":       sym_b,
                    "window":         window,
                    "score":          sc,
                    "correlation":    round(corr, 4),
                    "coint_pvalue":   round(pvalue, 4),
                    "halflife":       round(hl, 1) if not np.isnan(hl) else None,
                    "trades":         bt["total_trades"],
                    "win_rate":       bt["win_rate"],
                    "total_pnl":      bt["total_pnl"],
                    "avg_pnl":        bt["avg_pnl"],
                    "max_dd":         bt["max_dd"],
                    "sharpe":         bt["sharpe"],
                    "profit_factor":  bt["profit_factor"],
                    "unique_ratio_a": round(len(set(prices[sym_a].values)) / len(prices[sym_a]), 4),
                    "unique_ratio_b": round(len(set(prices[sym_b].values)) / len(prices[sym_b]), 4),
                }
                results.append(r)


        else:
            if multi_window:
                print(f"  Мульти-window бэктест: {windows_list}  ({ANALYSIS_WORKERS} процессов)...")
                args_list = [
                    (sym_a, sym_b,
                     prices[sym_a].values, prices[sym_a].index,
                     prices[sym_b].values, prices[sym_b].index,
                     fundings.get(sym_a, pd.Series(dtype=float)),
                     fundings.get(sym_b, pd.Series(dtype=float)),
                     windows_list)
                    for sym_a, sym_b, _ in pairs_todo
                ]
                worker_fn = analyze_pair_multiwindow
            else:
                args_list = [
                    (sym_a, sym_b,
                     prices[sym_a].values, prices[sym_a].index,
                     prices[sym_b].values, prices[sym_b].index,
                     fundings.get(sym_a, pd.Series(dtype=float)),
                     fundings.get(sym_b, pd.Series(dtype=float)))
                    for sym_a, sym_b, _ in pairs_todo
                ]
                worker_fn = analyze_pair

            total = len(args_list)
            done  = 0
            found = len(results)
            last_milestone = -1

            with ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
                futs = {executor.submit(worker_fn, args): args[:2] for args in args_list}
                for future in as_completed(futs):
                    done += 1
                    try:
                        r = future.result()
                        if multi_window:
                            # r — список строк (одна на window)
                            if r:
                                for row in r:
                                    results.append(row)
                                    found += 1
                        else:
                            if r is not None:
                                results.append(r)
                                found += 1
                    except Exception:
                        pass
                    pct       = done / total * 100
                    milestone = int(pct // 5) * 5
                    if milestone > last_milestone:
                        last_milestone = milestone
                        elapsed   = _time.time() - t0
                        speed     = done / elapsed
                        remaining = (total - done) / speed if speed > 0 else 0
                        eta_str   = f"{remaining/60:.1f}мин" if remaining >= 60 else f"{remaining:.0f}с"
                        bar = "█" * (milestone // 5) + "░" * (20 - milestone // 5)
                        print(
                            f"  [{bar}] {pct:5.1f}%  "
                            f"{done:,}/{total:,} пар  "
                            f"найдено: {found}  "
                            f"скорость: {speed:.0f} пар/с  "
                            f"ETA: {eta_str}"
                        )

    elapsed = _time.time() - t0
    speed   = len(pairs_todo) / elapsed if elapsed > 0 else 0
    print(f"  {'█'*20}  100.0%  ✓ Готово за {elapsed:.1f}s  "
          f"({elapsed/60:.1f}мин)  {speed:.0f} пар/с")

    if not results:
        print("\n  Ни одна пара не прошла фильтры. Попробуй смягчить условия.")
        return

    # ── 4. Сохраняем ────────────────────────────────────────────
    print(f"\n[4/4] Сохраняю результаты...")
    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1

    if multi_window:
        out_file = f"pairs_report_multiwindow_{EXCHANGE}.csv"
        base_cols = [
            "symbol_a", "symbol_b", "window", "score",
            "correlation", "coint_pvalue", "halflife",
            "trades", "win_rate", "total_pnl", "avg_pnl",
            "max_dd", "sharpe", "profit_factor",
            "unique_ratio_a", "unique_ratio_b",
        ]
        ordered = [c for c in base_cols if c in df.columns]
        df = df[ordered]
        df.to_csv(out_file, index_label="rank")
        window_counts = df["window"].value_counts().sort_index().to_dict() if "window" in df.columns else {}
        print(f"  ✓ {out_file}  ({len(df)} строк, окна: {window_counts})")
    else:
        # Если запущен из sequential режима — добавляем w{window} в имя файла
        leg1_suffix = "_leg1" if SYMBOLS_LEG1 else ""
        if WINDOWS_SEQUENTIAL:
            out_file = f"pairs_report_{EXCHANGE}{leg1_suffix}_w{window}.csv"
        else:
            out_file = f"pairs_report_{EXCHANGE}{leg1_suffix}.csv"
        df.to_csv(out_file, index_label="rank")
        print(f"  ✓ {out_file}  ({len(df)} пар)")
    return out_file


def main():
    global _semaphore, _req_done, _req_total, _sym_done, _sym_total, _t_start
    global EXCHANGE, CACHE_DIR

    # ── Парсинг аргументов командной строки ──────────────────────
    force_refresh = "--refresh"     in sys.argv
    multi_window  = "--multiwindow" in sys.argv or bool(WINDOWS_TO_TEST)

    # --exchange <name>
    if "--exchange" in sys.argv:
        idx = sys.argv.index("--exchange")
        if idx + 1 < len(sys.argv):
            arg_exchange = sys.argv[idx + 1].lower()
            if arg_exchange in EXCHANGE_URLS:
                EXCHANGE = arg_exchange
            else:
                print(f"  ⚠ Неизвестная биржа '{arg_exchange}'. Допустимые: binance, okx, bybit")
                print(f"  → Использую дефолт: {EXCHANGE}")

    # Применяем раздельные пути кеша
    CACHE_DIR     = EXCHANGE_CACHE_DIRS[EXCHANGE]

    _semaphore   = Semaphore(MAX_WORKERS)
    windows_list = WINDOWS_TO_TEST if multi_window else [WINDOW]

    print("=" * 68)
    print("  PAIRS SCANNER — Binance / OKX / Bybit Futures")
    print(f"  Биржа: {EXCHANGE.upper()}  |  TF: {TIMEFRAME}  |  Candles: {CANDLES}  |  Window: {WINDOW}")
    if SYMBOLS_LEG1:
        print(f"  ⚡ LEG1-режим: {SYMBOLS_LEG1} × все монеты биржи")
    if multi_window:
        print(f"  ⚡ Мульти-window режим: {windows_list}")
    print(f"  Потоков: {MAX_WORKERS}  |  Кеш: {CACHE_DIR}/  (макс. возраст: {CACHE_MAX_AGE_H}ч)")
    print(f"  Фильтры: WR>={MIN_WINRATE}%  PnL>={MIN_PNL}  p-value<={MAX_COINT_PVALUE}  MaxDD>={MAX_DRAWDOWN}  UniqueRatio>={MIN_UNIQUE_RATIO}")
    if _gpu_available:
        print(f"  ⚡ GPU режим: CuPy + Numba CUDA")
    else:
        print(f"  ⚠  GPU недоступен — CPU режим ({ANALYSIS_WORKERS} процессов)")
    if force_refresh:
        print("  ⚡ Режим --refresh: кеш игнорируется")
    print("=" * 68)
    print(f"  Каталог кеша: {CACHE_DIR}/")
    print("  Запуск с другой биржей:  python scanerGPU.py --exchange okx")
    print("                           python scanerGPU.py --exchange bybit")
    print("=" * 68)

    # ── 0. Символы ───────────────────────────────────────────────
    use_leg1   = bool(SYMBOLS_LEG1)
    use_manual = bool(SYMBOLS_MANUAL) and not use_leg1

    if use_leg1:
        # LEG1-режим: загружаем все монеты биржи + гарантируем наличие LEG1-монет
        print(f"\n[0/4] LEG1-режим: первая нога {SYMBOLS_LEG1}")
        print(f"  Загружаю все символы с {EXCHANGE.upper()} + LEG1-монеты...")
        exchange_symbols = fetch_all_symbols()
        # Добавляем LEG1-монеты, если их нет на бирже (маловероятно, но на всякий)
        symbols = sorted(set(exchange_symbols) | set(SYMBOLS_LEG1))
        if MIN_VOLUME_24H > 0:
            print(f"  Фильтр объёма >= ${MIN_VOLUME_24H/1e6:.0f}M за 24ч (LEG1-монеты исключены из фильтра)...")
            filtered = filter_by_volume(exchange_symbols, MIN_VOLUME_24H)
            # LEG1-монеты берём в любом случае
            symbols = sorted(set(filtered) | set(SYMBOLS_LEG1))
        leg1_str = ", ".join(SYMBOLS_LEG1)
        print(f"  Итого монет для загрузки: {len(symbols)}  (LEG1: {leg1_str})")

    elif use_manual:
        symbols = SYMBOLS_MANUAL
        if symbols and isinstance(symbols[0], (tuple, list)):
            flat = []
            for item in symbols:
                flat.extend(item)
            symbols = sorted(set(flat))
            print(f"\n[0/4] Ручной список пар → {len(symbols)} уникальных символов")
        else:
            print(f"\n[0/4] Ручной список: {len(symbols)} монет")
    else:
        print(f"\n[0/4] SYMBOLS_MANUAL пуст — загружаю все символы с {EXCHANGE.upper()}...")
        symbols = fetch_all_symbols()

    if not use_manual and not use_leg1 and MIN_VOLUME_24H > 0:
        print(f"  Фильтр объёма >= ${MIN_VOLUME_24H/1e6:.0f}M за 24ч...")
        symbols = filter_by_volume(symbols, MIN_VOLUME_24H)
    elif use_manual:
        print(f"  Фильтр объёма пропущен (ручной список)")

    total_possible = len(symbols) * (len(symbols) - 1) // 2

    fresh, stale = cache_stats(symbols)
    need_download = stale if not force_refresh else len(symbols)
    chunks_per_sym = (CANDLES + 1499) // 1500
    net_requests = need_download * chunks_per_sym
    _req_total = (fresh if not force_refresh else 0) + net_requests
    _sym_total = len(symbols)

    print(f"\n  Монет: {len(symbols)}  |  Пар: {total_possible:,}")
    print(f"  Кеш: {fresh} свежих ✓  |  Качать: {need_download} монет × {chunks_per_sym} чанков = {net_requests} запросов")
    if need_download > 0:
        t_est = (net_requests * 0.3) / MAX_WORKERS
        print(f"  Оценка загрузки: ~{t_est:.0f}s")

    # ── 1. Загрузка ──────────────────────────────────────────────
    print(f"\n[1/4] Загружаю свечи с {EXCHANGE.upper()}...")
    _req_done = _sym_done = 0
    _t_start = _time.time()
    prices = {}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = {executor.submit(fetch_klines, sym, force_refresh): sym for sym in symbols}
        for future in as_completed(futs):
            sym, series = future.result()
            if series is None:
                failed.append((sym, "ошибка загрузки"))
            elif len(series) < CANDLES:
                failed.append((sym, f"мало свечей: {len(series)} < {CANDLES} (CANDLES)"))
            elif len(series) < WINDOW + 50:
                failed.append((sym, f"мало свечей: {len(series)} < {WINDOW+50}"))
            else:
                prices[sym] = series

    elapsed = _time.time() - _t_start
    print(f"\n  ✓ {len(prices)} монет готово за {elapsed:.1f}s  ({len(failed)} пропущено)   ")
    if failed:
        for sym, reason in failed:
            print(f"  ✗ {sym}: {reason}")

    # ── Фильтр "мёртвых" монет ────────────────────────────────────
    if MIN_UNIQUE_RATIO > 0:
        dead = [s for s, series in prices.items()
                if series.nunique() / len(series) < MIN_UNIQUE_RATIO]
        for s in dead:
            del prices[s]
        if dead:
            print(f"  🚫 Мёртвые монеты убрано: {len(dead)}  ({', '.join(sorted(dead))})")

    # ── 1b. Фандинг ───────────────────────────────────────────────
    fundings: dict = {}
    if USE_FUNDING:
        print(f"\n[1b] Загружаю фандинг ({len(prices)} монет) с {EXCHANGE.upper()}...")
        f_done = [0]
        f_lock = Lock()
        def _dl_funding(sym):
            series = prices.get(sym)
            if series is None or len(series) == 0:
                fundings[sym] = pd.Series(dtype=float)
                return
            start_ms = int(series.index[0].timestamp() * 1000)
            end_ms   = int(series.index[-1].timestamp() * 1000)
            try:
                fundings[sym] = fetch_funding_rates(sym, start_ms, end_ms)
            except Exception:
                fundings[sym] = pd.Series(dtype=float)
            with f_lock:
                f_done[0] += 1
                print(f"  [{f_done[0]:>3}/{len(prices)}] фандинг {sym}  ({len(fundings[sym])} событий)   ", end="\r")
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS * 2, 8)) as f_exec:
            list(f_exec.map(_dl_funding, list(prices.keys())))
        loaded = sum(1 for v in fundings.values() if len(v) > 0)
        print(f"\n  ✓ Фандинг загружен для {loaded}/{len(prices)} монет")

    # ── 2-4. Анализ и сохранение ────────────────────────────────
    if WINDOWS_SEQUENTIAL:
        # Последовательный режим: отдельный полный прогон на каждое окно
        print(f"\n  ⚡ ПОСЛЕДОВАТЕЛЬНЫЙ РЕЖИМ: {WINDOWS_SEQUENTIAL}")
        print(f"  Данные уже загружены — буду переиспользовать для каждого окна")
        for w in WINDOWS_SEQUENTIAL:
            print(f"\n{'='*68}")
            print(f"  ОКНО {w}  ({WINDOWS_SEQUENTIAL.index(w)+1}/{len(WINDOWS_SEQUENTIAL)})")
            print(f"{'='*68}")
            _run_scan_for_window(w, prices, fundings)
        print(f"\n✅ Все окна завершены: {WINDOWS_SEQUENTIAL}")
    else:
        _run_scan_for_window(WINDOW if not multi_window else windows_list[0],
                             prices, fundings)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()