import os, time, json, threading
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np
import requests

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

PAIRS = [
    ("API3USDT", "EDENUSDT"),
    ("COLLECTUSDT", "HUSDT"),
    ("IOUSDT", "JCTUSDT"),
    ("SOONUSDT", "XANUSDT"),
    ("ALTUSDT", "ZKJUSDT"),
    ("MBOXUSDT", "ZKJUSDT"),
]

TIMEFRAME         = "5m"
WINDOW            = 500
CACHE_MAX_CANDLES = 5000  # потолок кеша свечей на монету (~17 дней на 5m)
DDOF              = 0
GRID_LEVELS       = [2.0, 3.0, 4.0]
CLOSE_AT_ZERO     = 0
TRADE_SIZE        = 40.0    # $ на уровень
COMMISSION        = 0.0005
LEVERAGE          = 3
MARGIN_TYPE       = "CROSSED"
HEDGE_MODE        = True    # True если в аккаунте включён Hedge Mode (двусторонние позиции)

# Алерт если unrealizedPnl по любой позиции хуже порога
PNL_WARN_THRESHOLD = -20.0  # $

STATE_FILE   = "trader_state.json"
HISTORY_FILE = "trader_history.json"   # все закрытые сделки — отдельный файл
API_PORT     = 51125

# ──────────────────────────────────────────────────────────────────────────────

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── ГЛОБАЛЬНОЕ СОСТОЯНИЕ ────────────────────────────────────────────────────
g_pair_states     = {}
g_pair_history    = {}
g_pair_zscore     = {}
g_pair_equity     = {}
g_lock            = threading.Lock()

g_pairs           = list(PAIRS)
g_paused_pairs    = {f"{a}_{b}" for a, b in PAIRS}  # все пары стартуют на паузе
g_pair_trade_size = {}   # { key: float }
g_pair_tf_window  = {}   # { key: {"window": int} }

# Сохранённые итоговые статы пар (заполняются при удалении пары, сохраняются в state)
# { key: {"total_pnl": float, "total_trades": int, "wins": int, "total_commission": float} }
g_pair_stats_snapshot = {}

# ─── КЭШ СВЕЧЕЙ ──────────────────────────────────────────────────────────────
g_klines_cache = {}
g_cache_lock   = threading.Lock()

# ─── АНТИСПАМ ДЛЯ PNL-АЛЕРТОВ ────────────────────────────────────────────────
g_pnl_alerted = {}  # { symbol: True }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntryInfo:
    level: float
    entry_time: str
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    size_usd: float = TRADE_SIZE
    qty_a: float = 0.0
    qty_b: float = 0.0

@dataclass
class TradeInfo:
    direction: int
    symbol_a: str
    symbol_b: str
    entries: list = field(default_factory=list)
    exit_time: Optional[str] = None
    exit_z: float = 0.0
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    closed: bool = False

    @property
    def n_entries(self): return len(self.entries)
    @property
    def levels(self): return sorted(e.level for e in self.entries)
    @property
    def total_size(self): return sum(e.size_usd for e in self.entries)
    @property
    def first_entry_time(self): return self.entries[0].entry_time if self.entries else None
    @property
    def first_entry_z(self): return self.entries[0].entry_z if self.entries else None


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def tg_error(text: str):
    _tg_send(text)

def tg_notify(text: str):
    _tg_send(text)

def _tg_send(text: str):
    token   = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        ).raise_for_status()
    except Exception as e:
        log.error(f"TG send failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  BINANCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_client():
    from binance.client import Client
    import time as _time
    client = Client(
        os.getenv("BINANCE_API_KEY"),
        os.getenv("BINANCE_API_SECRET"),
        tld="com",
        requests_params={"timeout": 30}
    )
    server_time = client.get_server_time()["serverTime"]
    client.timestamp_offset = server_time - int(_time.time() * 1000)
    return client


def fetch_closed_klines(client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Загружает свечи с кэшированием — при каждом тике докачивает только новые."""
    with g_cache_lock:
        cached = g_klines_cache.get(symbol)

    if cached is not None:
        df_cached = cached["df"]
        raw = client.futures_klines(symbol=symbol, interval=interval, limit=4)
        raw = raw[:-1]
        df_new = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
        ])
        df_new["ts"]    = pd.to_datetime(df_new["ts"].astype(np.int64), unit="ms", utc=True)
        df_new["close"] = df_new["close"].astype(float)
        df_new = df_new.set_index("ts")
        df_merged = pd.concat([df_cached, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep="last")].sort_index()
        df_merged = df_merged.iloc[-CACHE_MAX_CANDLES:]
        with g_cache_lock:
            g_klines_cache[symbol] = {"df": df_merged, "last_ts": df_merged.index[-1]}
        return df_merged.iloc[-limit:]
    else:
        raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit + 1)
        raw = raw[:-1]
        df = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
        ])
        df["ts"]    = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        df = df.set_index("ts")
        with g_cache_lock:
            g_klines_cache[symbol] = {"df": df, "last_ts": df.index[-1]}
        return df


def compute_z(close_a: pd.Series, close_b: pd.Series, window: int = WINDOW) -> float:
    spread = np.log(close_a) - np.log(close_b)
    rm     = spread.rolling(window).mean()
    rs     = spread.rolling(window).std(ddof=DDOF)
    z      = (spread - rm) / rs
    return float(z.iloc[-1])


def get_precision(client, symbol: str) -> int:
    info = client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            return s["quantityPrecision"]
    return 3


def market_order(client, symbol: str, side: str, usdt: float, price: float,
                 qty_prec: int, position_side: str = "BOTH") -> tuple[float, float]:
    """Возвращает (executed_qty, avg_price)."""
    qty = round(usdt / price, qty_prec)
    params = dict(symbol=symbol, side=side, type="MARKET", quantity=qty)
    if HEDGE_MODE:
        params["positionSide"] = position_side
    order     = client.futures_create_order(**params)
    executed  = float(order.get("executedQty") or 0)
    avg_price = float(order.get("avgPrice") or price)
    if executed == 0:
        executed = float(order.get("origQty") or qty)
    log.info(f"  ORD {symbol} {side} qty={executed} avgPrice={avg_price} → {order.get('status')}")
    return executed, avg_price


def close_symbol_qty(client, symbol: str, side: str, qty: float, position_side: str = "BOTH"):
    """Закрывает ровно qty контрактов по символу."""
    if qty <= 0:
        for p in client.futures_position_information(symbol=symbol):
            if p["symbol"] == symbol:
                amt = float(p["positionAmt"])
                if abs(amt) == 0:
                    return
                qty  = abs(amt)
                side = "SELL" if amt > 0 else "BUY"
                log.warning(f"  CLOSE {symbol} qty=0, аварийно закрываем positionAmt={amt}")
                break
        else:
            return
    if HEDGE_MODE:
        order = client.futures_create_order(
            symbol=symbol, side=side, type="MARKET",
            quantity=qty, positionSide=position_side
        )
    else:
        order = client.futures_create_order(
            symbol=symbol, side=side, type="MARKET",
            quantity=qty, reduceOnly=True
        )
    log.info(f"  CLOSE {symbol} {side} qty={qty} → {order.get('status')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  СОСТОЯНИЕ (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def save_state():
    with g_lock:
        data = {
            # Список активных пар — источник правды при рестарте
            "active_pairs": [[a, b] for a, b in g_pairs],
            "pairs": {},
            "paused_pairs": list(g_paused_pairs),
            "pair_trade_size": dict(g_pair_trade_size),
            "pair_tf_window": dict(g_pair_tf_window),
            # Снапшоты статов удалённых пар — хранятся вечно
            "pair_stats_snapshot": dict(g_pair_stats_snapshot),
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        for key, state in g_pair_states.items():
            open_levels_serializable = {
                f"{d}_{lv}": v
                for (d, lv), v in state.get("open_levels", {}).items()
            }
            data["pairs"][key] = {
                "open_levels": open_levels_serializable,
                "trades": {
                    str(d): {
                        "direction": t.direction,
                        "symbol_a": t.symbol_a,
                        "symbol_b": t.symbol_b,
                        "entries": [asdict(e) for e in t.entries]
                    }
                    for d, t in state["open_trades"].items()
                }
            }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def asdict_trade(t: TradeInfo) -> dict:
    return {
        "direction": t.direction,
        "symbol_a": t.symbol_a,
        "symbol_b": t.symbol_b,
        "entries": [asdict(e) for e in t.entries],
        "exit_time": t.exit_time,
        "exit_z": t.exit_z,
        "exit_price_a": t.exit_price_a,
        "exit_price_b": t.exit_price_b,
        "pnl": t.pnl,
        "commission": t.commission,
        "closed": t.closed,
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        with g_lock:
            # ── Восстанавливаем список активных пар ──────────────────────────
            # Приоритет: active_pairs из файла (дэшборд), иначе остаётся PAIRS из кода
            if "active_pairs" in data:
                saved = [tuple(p) for p in data["active_pairs"]]
                existing_keys = {f"{a}_{b}" for a, b in g_pairs}
                # Добавляем пары из файла которых нет в g_pairs
                for pair in saved:
                    k = f"{pair[0]}_{pair[1]}"
                    if k not in existing_keys:
                        g_pairs.append(pair)
                        existing_keys.add(k)
                # Удаляем пары которых нет в файле (удалены через дэшборд)
                saved_keys = {f"{a}_{b}" for a, b in saved}
                to_remove  = [(a, b) for a, b in g_pairs if f"{a}_{b}" not in saved_keys]
                for pair in to_remove:
                    g_pairs.remove(pair)

            # ── Снапшоты статов удалённых пар ────────────────────────────────
            if "pair_stats_snapshot" in data:
                g_pair_stats_snapshot.update(data["pair_stats_snapshot"])

            # ── Состояния и история пар ───────────────────────────────────────
            for key, pair_data in data.get("pairs", {}).items():
                sym_a, sym_b = key.split("_", 1)
                if key not in g_pair_states:
                    g_pair_states[key] = {"open_trades": {}, "open_levels": {}, "warmed_up": False}

                if "trades" in pair_data:
                    trades_data = pair_data["trades"]
                    raw_levels  = pair_data.get("open_levels", {})
                    for lv_key, v in raw_levels.items():
                        d_str, lv_str = lv_key.split("_", 1)
                        g_pair_states[key]["open_levels"][(int(d_str), float(lv_str))] = v
                else:
                    trades_data = pair_data

                for d_str, td in trades_data.items():
                    d = int(d_str)
                    trade = TradeInfo(direction=d, symbol_a=sym_a, symbol_b=sym_b)
                    for e in td["entries"]:
                        ei = EntryInfo(
                            level=e["level"],
                            entry_time=e["entry_time"],
                            entry_z=e["entry_z"],
                            entry_price_a=e["entry_price_a"],
                            entry_price_b=e["entry_price_b"],
                            size_usd=e.get("size_usd", TRADE_SIZE),
                            qty_a=e.get("qty_a", 0.0),
                            qty_b=e.get("qty_b", 0.0),
                        )
                        trade.entries.append(ei)
                        lv_key_tuple = (d, ei.level)
                        if lv_key_tuple not in g_pair_states[key]["open_levels"]:
                            g_pair_states[key]["open_levels"][lv_key_tuple] = True
                    g_pair_states[key]["open_trades"][d] = trade

                if g_pair_states[key]["open_trades"]:
                    g_pair_states[key]["warmed_up"] = True


            if "paused_pairs" in data:
                g_paused_pairs.clear()
                g_paused_pairs.update(data["paused_pairs"])
            if "pair_trade_size" in data:
                g_pair_trade_size.update(data["pair_trade_size"])
            if "pair_tf_window" in data:
                g_pair_tf_window.update(data["pair_tf_window"])

        log.info(f"Состояние загружено из {STATE_FILE}")
    except Exception as e:
        log.error(f"Ошибка загрузки состояния: {e}")
        tg_error(f"❌ <b>Ошибка загрузки состояния:</b>\n<code>{e}</code>")


def save_history():
    """Сохраняет все закрытые сделки в отдельный файл HISTORY_FILE."""
    with g_lock:
        data = []
        for key, trades in g_pair_history.items():
            for t in trades:
                d = asdict_trade(t)
                d["pair_key"] = key
                data.append(d)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Ошибка сохранения истории: {e}")


def load_history():
    """Загружает историю закрытых сделок из HISTORY_FILE."""
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        with g_lock:
            by_pair: dict = {}
            for td in data:
                key = td.get("pair_key", f"{td['symbol_a']}_{td['symbol_b']}")
                by_pair.setdefault(key, []).append(td)

            for key, items in by_pair.items():
                trades = []
                for td in items:
                    t = TradeInfo(direction=td["direction"],
                                  symbol_a=td["symbol_a"], symbol_b=td["symbol_b"])
                    for e in td["entries"]:
                        ei = EntryInfo(
                            level=e["level"], entry_time=e["entry_time"],
                            entry_z=e["entry_z"],
                            entry_price_a=e["entry_price_a"], entry_price_b=e["entry_price_b"],
                            size_usd=e.get("size_usd", TRADE_SIZE),
                            qty_a=e.get("qty_a", 0.0), qty_b=e.get("qty_b", 0.0),
                        )
                        t.entries.append(ei)
                    t.exit_time    = td["exit_time"]
                    t.exit_z       = td["exit_z"]
                    t.exit_price_a = td["exit_price_a"]
                    t.exit_price_b = td["exit_price_b"]
                    t.pnl          = td["pnl"]
                    t.commission   = td["commission"]
                    t.closed       = td["closed"]
                    trades.append(t)
                g_pair_history[key] = trades

                # Восстанавливаем equity curve
                equity  = []
                running = 0.0
                for t in trades:
                    running += t.pnl
                    equity.append({"ts": t.exit_time, "pnl": round(running, 4)})
                g_pair_equity[key] = equity

        log.info(f"История загружена из {HISTORY_FILE} ({sum(len(v) for v in by_pair.values())} сделок)")
    except Exception as e:
        log.error(f"Ошибка загрузки истории: {e}")


def _compute_pair_stats(key: str) -> dict:
    """Считает итоговые статы по паре из текущей истории."""
    hist  = g_pair_history.get(key, [])
    pnls  = [t.pnl for t in hist]
    wins  = [p for p in pnls if p > 0]
    comms = sum(t.commission for t in hist)
    return {
        "total_pnl":        round(sum(pnls), 4),
        "total_trades":     len(pnls),
        "wins":             len(wins),
        "total_commission": round(comms, 4),
        "win_rate":         round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PNL МОНИТОРИНГ (раз в 5 минут вместе с основным циклом)
# ═══════════════════════════════════════════════════════════════════════════════

def check_unrealized_pnl(client):
    """
    Запрашивает все открытые фьючерсные позиции и шлёт алерт
    если unrealizedPnl по любому символу хуже PNL_WARN_THRESHOLD.
    Антиспам: повторный алерт только после восстановления выше порога.
    """
    try:
        positions = client.futures_position_information()
    except Exception as e:
        log.error(f"[PNL_CHECK] Ошибка запроса позиций: {e}")
        return

    for p in positions:
        amt = float(p.get("positionAmt", 0))
        if amt == 0:
            g_pnl_alerted.pop(p["symbol"], None)
            continue

        symbol = p["symbol"]
        pnl    = float(p.get("unrealizedProfit", 0))
        side   = "LONG" if amt > 0 else "SHORT"

        if pnl < PNL_WARN_THRESHOLD:
            if not g_pnl_alerted.get(symbol):
                log.warning(f"[PNL_CHECK] {symbol} {side} unrealizedPnL={pnl:.2f}")
                tg_notify(
                    f"⚠️ <b>Убыток по позиции!</b>\n"
                    f"<code>{symbol}</code> {side}\n"
                    f"unrealizedPnL: <b>${pnl:.2f}</b>\n"
                    f"Порог: ${PNL_WARN_THRESHOLD:.2f}"
                )
                g_pnl_alerted[symbol] = True
        else:
            if g_pnl_alerted.pop(symbol, False):
                log.info(f"[PNL_CHECK] {symbol} {side} восстановился: unrealizedPnL={pnl:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  РЕАЛЬНЫЙ PNL ЗАКРЫТОЙ СДЕЛКИ
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_real_pnl(client, sym_a: str, sym_b: str, trade: "TradeInfo") -> tuple[float, float]:
    """
    Запрашивает реальный PnL сделки с биржи через futures_income_history.
    Возвращает (pnl_net, commission_total).
    """
    try:
        first_entry = trade.entries[0]
        start_ts = int(pd.Timestamp(first_entry.entry_time).timestamp() * 1000) - 60_000
        end_ts   = int(datetime.now(timezone.utc).timestamp() * 1000) + 60_000

        realized_pnl = 0.0
        funding_fee  = 0.0
        commission   = 0.0

        for sym in (sym_a, sym_b):
            try:
                records = client.futures_income_history(
                    symbol=sym, startTime=start_ts, endTime=end_ts, limit=1000,
                )
            except Exception as e:
                log.warning(f"[fetch_real_pnl] futures_income_history {sym}: {e}")
                continue

            for rec in records:
                income_type = rec.get("incomeType", "")
                amount      = float(rec.get("income", 0))
                if income_type == "REALIZED_PNL":
                    realized_pnl += amount
                elif income_type == "FUNDING_FEE":
                    funding_fee  += amount
                elif income_type == "COMMISSION":
                    commission   += amount

        pnl_net  = realized_pnl + funding_fee + commission
        comm_abs = abs(commission)
        log.info(
            f"[fetch_real_pnl] {sym_a}/{sym_b}  "
            f"realizedPnl={realized_pnl:+.4f}  "
            f"funding={funding_fee:+.4f}  "
            f"commission={commission:+.4f}  "
            f"→ net={pnl_net:+.4f}"
        )
        return pnl_net, comm_abs

    except Exception as e:
        log.error(f"[fetch_real_pnl] fallback to manual calc: {e}")
        pa        = trade.exit_price_a
        pb        = trade.exit_price_b
        direction = trade.direction
        gross     = 0.0
        for entry in trade.entries:
            half = entry.size_usd / 2
            ra = (pa - entry.entry_price_a) / entry.entry_price_a
            rb = (pb - entry.entry_price_b) / entry.entry_price_b
            gross += (ra * half - rb * half) if direction == 1 else (-ra * half + rb * half)
        comm = sum(e.size_usd * COMMISSION * 2 for e in trade.entries)
        return gross - comm, comm


# ═══════════════════════════════════════════════════════════════════════════════
#  ADL / SYNC
# ═══════════════════════════════════════════════════════════════════════════════

def check_adl_and_sync(client, key: str, sym_a: str, sym_b: str, prec_a: int, prec_b: int,
                        open_trades: dict, open_levels: dict, pa: float, pb: float):
    """
    Сверяет фактические позиции на бирже с внутренним состоянием бота.
    Если одна нога была ликвидирована/закрыта ADL — принудительно закрывает вторую.
    """
    if not open_trades:
        return []

    try:
        if HEDGE_MODE:
            all_pos = client.futures_position_information()
            positions_hedge = {
                (p["symbol"], p["positionSide"]): abs(float(p["positionAmt"]))
                for p in all_pos if p["symbol"] in (sym_a, sym_b)
            }
        else:
            positions = {p["symbol"]: abs(float(p["positionAmt"]))
                         for p in client.futures_position_information()
                         if p["symbol"] in (sym_a, sym_b)}
    except Exception as e:
        log.error(f"[{key}] check_adl: не удалось получить позиции: {e}")
        return []

    def real_qty(symbol, expected_side):
        if HEDGE_MODE:
            return positions_hedge.get((symbol, expected_side), 0.0)
        else:
            return positions.get(symbol, 0.0)

    dirs_to_force_close = []

    for direction, trade in list(open_trades.items()):
        if not trade.entries:
            continue

        expected_qty_a = round(sum(e.qty_a for e in trade.entries), prec_a)
        expected_qty_b = round(sum(e.qty_b for e in trade.entries), prec_b)

        ps_a = "LONG"  if direction == 1 else "SHORT"
        ps_b = "SHORT" if direction == 1 else "LONG"

        actual_qty_a = real_qty(sym_a, ps_a)
        actual_qty_b = real_qty(sym_b, ps_b)

        a_closed = actual_qty_a < expected_qty_a * 0.1
        b_closed = actual_qty_b < expected_qty_b * 0.1

        if not a_closed and not b_closed:
            continue

        # ── DOUBLE-CHECK: ждём 60с и перепроверяем ───────────────────────────
        _dir_label_pre = "LONG" if direction == 1 else "SHORT"
        log.warning(f"[{key}] Подозрение на ADL (direction={direction}) — жду 60s...")
        tg_error(
            f"⚠️ <b>Подозрение на ADL — проверяю...</b>\n"
            f"Пара: <code>{key}</code> ({_dir_label_pre})\n"
            f"{sym_a}: ожидал {expected_qty_a} / факт {actual_qty_a:.4f}\n"
            f"{sym_b}: ожидал {expected_qty_b} / факт {actual_qty_b:.4f}\n"
            f"Жду 60s и перепроверю."
        )
        import time as _adl_time
        _adl_time.sleep(60)
        try:
            if HEDGE_MODE:
                _recheck = client.futures_position_information()
                _pos2 = {
                    (p["symbol"], p["positionSide"]): abs(float(p["positionAmt"]))
                    for p in _recheck if p["symbol"] in (sym_a, sym_b)
                }
                actual_qty_a = _pos2.get((sym_a, ps_a), 0.0)
                actual_qty_b = _pos2.get((sym_b, ps_b), 0.0)
            else:
                _recheck = client.futures_position_information()
                _pos2 = {p["symbol"]: abs(float(p["positionAmt"]))
                         for p in _recheck if p["symbol"] in (sym_a, sym_b)}
                actual_qty_a = _pos2.get(sym_a, 0.0)
                actual_qty_b = _pos2.get(sym_b, 0.0)
        except Exception as e:
            log.error(f"[{key}] ADL double-check ошибка: {e}")

        a_closed = actual_qty_a < expected_qty_a * 0.1
        b_closed = actual_qty_b < expected_qty_b * 0.1

        if not a_closed and not b_closed:
            log.info(f"[{key}] ✅ ADL double-check пройден — ложная тревога")
            continue

        dir_label    = "LONG" if direction == 1 else "SHORT"
        close_side_a = "SELL" if direction == 1 else "BUY"
        close_side_b = "BUY"  if direction == 1 else "SELL"

        closed_legs = []
        errors      = []

        if a_closed and not b_closed:
            try:
                close_symbol_qty(client, sym_b, close_side_b, actual_qty_b, position_side=ps_b)
                closed_legs.append(f"{sym_b} ({actual_qty_b:.4f})")
            except Exception as e:
                errors.append(f"{sym_b}: {e}")

        elif b_closed and not a_closed:
            try:
                close_symbol_qty(client, sym_a, close_side_a, actual_qty_a, position_side=ps_a)
                closed_legs.append(f"{sym_a} ({actual_qty_a:.4f})")
            except Exception as e:
                errors.append(f"{sym_a}: {e}")

        which = []
        if a_closed: which.append(sym_a)
        if b_closed: which.append(sym_b)

        msg = (
            f"🚨 <b>ADL / Принудительная ликвидация!</b>\n"
            f"Пара: <code>{key}</code> ({dir_label})\n"
            f"Закрытые ноги: {', '.join(which)}\n"
        )
        if closed_legs:
            msg += f"Принудительно закрыто: {', '.join(closed_legs)}\n"
        if errors:
            msg += f"⚠️ Ошибки: {'; '.join(errors)}\n"
        msg += f"Уровни были: {trade.levels}"

        tg_error(msg)
        dirs_to_force_close.append(direction)

    return dirs_to_force_close


# ═══════════════════════════════════════════════════════════════════════════════
#  ЛОГИКА ОДНОЙ ПАРЫ
# ═══════════════════════════════════════════════════════════════════════════════

def process_pair(client, sym_a: str, sym_b: str, prec_a: int, prec_b: int):
    key = f"{sym_a}_{sym_b}"

    with g_lock:
        if key not in g_pair_states:
            g_pair_states[key] = {"open_trades": {}, "open_levels": {}, "warmed_up": False}
        if key not in g_pair_history:
            g_pair_history[key] = []
        if key not in g_pair_equity:
            g_pair_equity[key] = []

        state       = g_pair_states[key]
        open_trades = state["open_trades"]
        open_levels = state["open_levels"]
        pair_window = g_pair_tf_window.get(key, {}).get("window", WINDOW)

    limit   = pair_window * 2
    df_a    = fetch_closed_klines(client, sym_a, TIMEFRAME, limit)
    df_b    = fetch_closed_klines(client, sym_b, TIMEFRAME, limit)
    idx     = df_a.index.intersection(df_b.index)
    close_a = df_a.loc[idx, "close"]
    close_b = df_b.loc[idx, "close"]

    z_val   = compute_z(close_a, close_b, pair_window)
    last_ts = idx[-1].isoformat()
    pa      = float(close_a.iloc[-1])
    pb      = float(close_b.iloc[-1])

    z_str = f"{z_val:.4f}" if not (z_val != z_val) else "nan"
    log.info(f"[{key}] ts={last_ts}  z={z_str}  A={pa}  B={pb}  open={len(open_trades)}")

    with g_lock:
        g_pair_zscore[key] = {"z": round(z_val, 4), "ts": last_ts, "price_a": pa, "price_b": pb}

    # ── ПРОГРЕВ ───────────────────────────────────────────────────────────────
    with g_lock:
        warmed_up = state.get("warmed_up", True)

    if not warmed_up:
        min_level = min(GRID_LEVELS)
        if abs(z_val) < min_level:
            with g_lock:
                state["warmed_up"] = True
            log.info(f"[{key}] ✅ Прогрев завершён z={z_val:.4f}")
        else:
            log.info(f"[{key}] ⏳ Прогрев: |z|<{min_level}, сейчас z={z_val:.4f}")
        return

    # ── ПРОВЕРКА ADL ──────────────────────────────────────────────────────────
    if open_trades:
        adl_closed = check_adl_and_sync(
            client, key, sym_a, sym_b, prec_a, prec_b, open_trades, open_levels, pa, pb
        )
        if adl_closed:
            with g_lock:
                for d in adl_closed:
                    for lv in GRID_LEVELS:
                        open_levels.pop((d, lv), None)
                    open_trades.pop(d, None)
            save_state()

    # ── ЗАКРЫТИЕ ─────────────────────────────────────────────────────────────
    dirs_to_close = []
    for direction, trade in list(open_trades.items()):
        should_close = (
            (direction == 1  and z_val >= -CLOSE_AT_ZERO) or
            (direction == -1 and z_val <=  CLOSE_AT_ZERO)
        )
        if not should_close:
            continue

        log.info(f"[{key}] ЗАКРЫТИЕ direction={direction} z={z_val:.4f}")

        total_qty_a  = round(sum(e.qty_a for e in trade.entries), prec_a)
        total_qty_b  = round(sum(e.qty_b for e in trade.entries), prec_b)
        close_side_a = "SELL" if direction == 1 else "BUY"
        close_side_b = "BUY"  if direction == 1 else "SELL"

        try:
            ps_a = "LONG"  if direction == 1 else "SHORT"
            ps_b = "SHORT" if direction == 1 else "LONG"
            close_symbol_qty(client, sym_a, close_side_a, total_qty_a, position_side=ps_a)
            close_symbol_qty(client, sym_b, close_side_b, total_qty_b, position_side=ps_b)
        except Exception as e:
            log.error(f"[{key}] Ошибка закрытия: {e}")
            tg_error(f"❌ <b>{key}</b> Ошибка закрытия:\n<code>{e}</code>")
            continue

        pnl, comm = fetch_real_pnl(client, sym_a, sym_b, trade)

        trade.exit_time    = last_ts
        trade.exit_z       = round(z_val, 4)
        trade.exit_price_a = pa
        trade.exit_price_b = pb
        trade.pnl          = round(pnl, 4)
        trade.commission   = round(comm, 4)
        trade.closed       = True

        with g_lock:
            g_pair_history[key].append(trade)
            total_pnl = sum(t.pnl for t in g_pair_history[key])
            g_pair_equity[key].append({"ts": last_ts, "pnl": round(total_pnl, 4)})

        dirs_to_close.append(direction)

        pnl_emoji = "✅" if pnl >= 0 else "🔴"
        tg_notify(f"{pnl_emoji} Пара закрыта {key}  PnL: {'+'if pnl>=0 else ''}${pnl:.2f}")

    with g_lock:
        for d in dirs_to_close:
            for lv in GRID_LEVELS:
                open_levels.pop((d, lv), None)
            open_trades.pop(d, None)

    # ── ОТКРЫТИЕ ─────────────────────────────────────────────────────────────
    for level in GRID_LEVELS:
        direction = 0
        if   z_val <= -level: direction = 1
        elif z_val >=  level: direction = -1
        if direction == 0:
            continue

        with g_lock:
            if open_levels.get((direction, level)):
                continue

            existing_trade = open_trades.get(direction)
            if existing_trade:
                already_taken = any(e.level == level for e in existing_trade.entries)
                if already_taken:
                    log.debug(f"[{key}] Уровень {level} dir={direction} уже был взят")
                    continue

            if direction not in open_trades:
                open_trades[direction] = TradeInfo(
                    direction=direction, symbol_a=sym_a, symbol_b=sym_b
                )

        log.info(f"[{key}] ВХОД level={level} direction={direction} z={z_val:.4f}")
        with g_lock:
            pair_size = g_pair_trade_size.get(key, TRADE_SIZE)
        half = pair_size / 2
        try:
            if direction == 1:
                qty_a, price_a = market_order(client, sym_a, "BUY",  half, pa, prec_a, position_side="LONG")
                qty_b, price_b = market_order(client, sym_b, "SELL", half, pb, prec_b, position_side="SHORT")
            else:
                qty_a, price_a = market_order(client, sym_a, "SELL", half, pa, prec_a, position_side="SHORT")
                qty_b, price_b = market_order(client, sym_b, "BUY",  half, pb, prec_b, position_side="LONG")
        except Exception as e:
            log.error(f"[{key}] Ошибка открытия: {e}")
            tg_error(f"❌ <b>{key}</b> Ошибка открытия lv{level}:\n<code>{e}</code>")
            continue

        entry = EntryInfo(
            level=level, entry_time=last_ts,
            entry_z=round(z_val, 4),
            entry_price_a=price_a, entry_price_b=price_b,
            size_usd=pair_size,
            qty_a=qty_a, qty_b=qty_b
        )
        with g_lock:
            open_trades[direction].entries.append(entry)
            open_levels[(direction, level)] = True
            all_levels = sorted(e.level for e in open_trades[direction].entries)
            total_size = sum(e.size_usd for e in open_trades[direction].entries)

        dir_label = "LONG" if direction == 1 else "SHORT"
        side_a    = "▲ BUY"  if direction == 1 else "▼ SELL"
        side_b    = "▼ SELL" if direction == 1 else "▲ BUY"




# ═══════════════════════════════════════════════════════════════════════════════
#  REST API (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

import math

def sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def start_api():
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, Response
        import uvicorn

        app = FastAPI(title="Pairs Trader API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        API_TOKEN = os.getenv("API_TOKEN", "")
        PUBLIC    = {"/", "/favicon.ico"}

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            if request.url.path in PUBLIC:
                return await call_next(request)
            if API_TOKEN and request.headers.get("X-Token") != API_TOKEN:
                return JSONResponse(
                    {"error": "Unauthorized"}, status_code=401,
                    headers={"Access-Control-Allow-Origin": "*"}
                )
            return await call_next(request)

        @app.get("/")
        def root():
            return {"status": "Pairs Trader API running"}

        @app.get("/favicon.ico")
        def favicon():
            return Response(status_code=204)

        # ── GET /api/history ──────────────────────────────────────────────────
        # Полная история ВСЕХ сделок всех пар — для вкладки History в дашборде
        @app.get("/api/history")
        def get_all_history():
            with g_lock:
                result = []
                for key, trades in g_pair_history.items():
                    for i, t in enumerate(trades):
                        d = asdict_trade(t)
                        d["pair_key"] = key
                        d["index"]    = i   # индекс для удаления
                        result.append(d)
            # Сортируем по времени закрытия (новые сверху)
            result.sort(key=lambda x: x.get("exit_time") or "", reverse=True)
            return sanitize(result)

        # ── DELETE /api/history/{pair_key}/{index} ────────────────────────────
        @app.delete("/api/history/{pair_key}/{index}")
        def delete_history_trade(pair_key: str, index: int):
            """Удаляет конкретную сделку из истории по паре и индексу."""
            with g_lock:
                hist = g_pair_history.get(pair_key)
                if hist is None:
                    raise HTTPException(404, f"Pair {pair_key} not found in history")
                if index < 0 or index >= len(hist):
                    raise HTTPException(400, f"Index {index} out of range")
                removed = hist.pop(index)
                # Пересчитываем equity
                equity  = []
                running = 0.0
                for t in hist:
                    running += t.pnl
                    equity.append({"ts": t.exit_time, "pnl": round(running, 4)})
                g_pair_equity[pair_key] = equity
            save_state()
            save_history()
            log.info(f"[API] История: удалена сделка {pair_key}[{index}] pnl={removed.pnl}")
            return {"ok": True, "pair_key": pair_key, "removed_index": index}

        # ── GET /api/pairs ────────────────────────────────────────────────────
        @app.get("/api/pairs")
        def get_pairs():
            result = []
            with g_lock:
                for sym_a, sym_b in g_pairs:
                    key    = f"{sym_a}_{sym_b}"
                    hist   = g_pair_history.get(key, [])
                    state  = g_pair_states.get(key, {})
                    open_t = state.get("open_trades", {})
                    pnls   = [t.pnl for t in hist]
                    wins   = [p for p in pnls if p > 0]
                    comms  = sum(t.commission for t in hist)

                    result.append({
                        "key":              key,
                        "symbol_a":         sym_a,
                        "symbol_b":         sym_b,
                        "paused":           key in g_paused_pairs,
                        "trade_size":       g_pair_trade_size.get(key, TRADE_SIZE),
                        "window":           g_pair_tf_window.get(key, {}).get("window", WINDOW),
                        "zscore":           g_pair_zscore.get(key, {}),
                        "open_trades":      len(open_t),
                        "total_trades":     len(pnls),
                        "win_rate":         round(len(wins)/len(pnls)*100, 1) if pnls else 0,
                        "total_pnl":        round(sum(pnls), 2),
                        "total_commission": round(comms, 2),
                        "open_positions": [
                            {
                                "direction":        d,
                                "n_entries":        t.n_entries,
                                "levels":           t.levels,
                                "first_entry_time": t.first_entry_time,
                                "first_entry_z":    t.first_entry_z,
                                "total_size":       t.total_size,
                                "entries":          [asdict(e) for e in t.entries],
                            }
                            for d, t in open_t.items()
                        ],
                    })
            return sanitize(result)

        # ── GET /api/pairs/{key}/trades ───────────────────────────────────────
        @app.get("/api/pairs/{key}/trades")
        def get_trades(key: str):
            with g_lock:
                hist = g_pair_history.get(key, [])
                return sanitize([asdict_trade(t) for t in hist])

        # ── DELETE /api/pairs/{key}/trades/{index} ────────────────────────────
        @app.delete("/api/pairs/{key}/trades/{index}")
        def delete_trade(key: str, index: int):
            """Удаляет сделку из истории по индексу (0-based).
            Пересчитывает equity curve и снапшот статов после удаления."""
            with g_lock:
                hist = g_pair_history.get(key)
                if hist is None:
                    raise HTTPException(404, f"Pair {key} not found")
                if index < 0 or index >= len(hist):
                    raise HTTPException(400, f"Index {index} out of range (0..{len(hist)-1})")
                removed = hist.pop(index)
                # Пересчитываем equity curve
                equity  = []
                running = 0.0
                for t in hist:
                    running += t.pnl
                    equity.append({"ts": t.exit_time, "pnl": round(running, 4)})
                g_pair_equity[key] = equity
            save_state()
            log.info(f"[API] Удалена сделка #{index} из {key} (pnl={removed.pnl})")
            return {"ok": True, "key": key, "removed_index": index, "removed_pnl": removed.pnl}

        # ── GET /api/pairs/{key}/equity ───────────────────────────────────────
        @app.get("/api/pairs/{key}/equity")
        def get_equity(key: str):
            with g_lock:
                return sanitize(g_pair_equity.get(key, []))

        # ── GET /api/stats/all ────────────────────────────────────────────────
        # Статистика по ВСЕМ парам: активным + удалённым (снапшоты)
        @app.get("/api/stats/all")
        def get_all_stats():
            result = {}
            with g_lock:
                for sym_a, sym_b in g_pairs:
                    key = f"{sym_a}_{sym_b}"
                    result[key] = _compute_pair_stats(key)
                    result[key]["active"] = True
                for key, snap in g_pair_stats_snapshot.items():
                    if key not in result:
                        result[key] = dict(snap)
                        result[key]["active"] = False
            return sanitize(result)

        # ── GET /api/status ───────────────────────────────────────────────────
        @app.get("/api/status")
        def get_status():
            with g_lock:
                total  = len(g_pairs)
                paused = len(g_paused_pairs)
            return {
                "running":            True,
                "pairs":              total,
                "paused":             paused,
                "timeframe":          TIMEFRAME,
                "window":             WINDOW,
                "grid_levels":        GRID_LEVELS,
                "trade_size":         TRADE_SIZE,
                "leverage":           LEVERAGE,
                "margin_type":        MARGIN_TYPE,
                "pnl_warn_threshold": PNL_WARN_THRESHOLD,
            }

        # ── POST /api/pairs/add ───────────────────────────────────────────────
        @app.post("/api/pairs/add")
        async def add_pair(request: Request):
            body  = await request.json()
            sym_a = body.get("symbol_a", "").upper().strip()
            sym_b = body.get("symbol_b", "").upper().strip()
            if not sym_a or not sym_b:
                raise HTTPException(400, "symbol_a and symbol_b required")
            if sym_a == sym_b:
                raise HTTPException(400, "symbol_a and symbol_b must be different")
            key = f"{sym_a}_{sym_b}"
            with g_lock:
                existing = [f"{a}_{b}" for a, b in g_pairs]
                if key in existing:
                    raise HTTPException(409, f"Pair {key} already exists")
                g_pairs.append((sym_a, sym_b))
                g_paused_pairs.add(key)
                if key not in g_pair_states:
                    g_pair_states[key] = {"open_trades": {}, "open_levels": {}, "warmed_up": False}
                if key not in g_pair_history:
                    g_pair_history[key] = []
                if key not in g_pair_equity:
                    g_pair_equity[key] = []

            setup_errors = []
            try:
                _cl = get_client()
                for sym in (sym_a, sym_b):
                    try:
                        _cl.futures_change_leverage(symbol=sym, leverage=LEVERAGE)
                    except Exception as e:
                        setup_errors.append(f"leverage {sym}: {e}")
                    try:
                        _cl.futures_change_margin_type(symbol=sym, marginType=MARGIN_TYPE)
                    except Exception as e:
                        if "No need to change" not in str(e):
                            setup_errors.append(f"margin_type {sym}: {e}")
            except Exception as e:
                setup_errors.append(str(e))

            save_state()
            log.info(f"[API] Добавлена пара {key}")
            result = {"ok": True, "key": key, "leverage": LEVERAGE, "margin_type": MARGIN_TYPE}
            if setup_errors:
                result["warnings"] = setup_errors
            return result

        # ── DELETE /api/pairs/{key} ───────────────────────────────────────────
        @app.delete("/api/pairs/{key}")
        def delete_pair(key: str):
            with g_lock:
                found = [(a, b) for a, b in g_pairs if f"{a}_{b}" == key]
                if not found:
                    raise HTTPException(404, f"Pair {key} not found")
                state  = g_pair_states.get(key, {})
                open_t = state.get("open_trades", {})
                if open_t:
                    raise HTTPException(409, f"Pair {key} has open positions — close them first")
                sym_a, sym_b = found[0]
                # Сохраняем снапшот статистики перед удалением
                g_pair_stats_snapshot[key] = _compute_pair_stats(key)
                g_pairs.remove((sym_a, sym_b))
                g_paused_pairs.discard(key)
            save_state()
            log.info(f"[API] Удалена пара {key}, снапшот статов сохранён")
            return {"ok": True, "key": key, "stats_snapshot": g_pair_stats_snapshot[key]}

        # ── PATCH /api/pairs/{key}/trade_size ─────────────────────────────────
        @app.patch("/api/pairs/{key}/trade_size")
        async def set_trade_size(key: str, request: Request):
            body = await request.json()
            try:
                size = float(body.get("trade_size", 0))
            except (TypeError, ValueError):
                raise HTTPException(400, "trade_size must be a number")
            if size <= 0:
                raise HTTPException(400, "trade_size must be > 0")
            with g_lock:
                existing = [f"{a}_{b}" for a, b in g_pairs]
                if key not in existing:
                    raise HTTPException(404, f"Pair {key} not found")
                g_pair_trade_size[key] = size
            log.info(f"[API] Пара {key}: trade_size={size}")
            return {"ok": True, "key": key, "trade_size": size}

        # ── PATCH /api/pairs/{key}/tf_window ──────────────────────────────────
        @app.patch("/api/pairs/{key}/tf_window")
        async def set_tf_window(key: str, request: Request):
            body = await request.json()
            try:
                win = int(body.get("window", 0))
            except (TypeError, ValueError):
                raise HTTPException(400, "window must be an integer")
            if win < 10:
                raise HTTPException(400, "window must be >= 10")
            with g_lock:
                existing = [f"{a}_{b}" for a, b in g_pairs]
                if key not in existing:
                    raise HTTPException(404, f"Pair {key} not found")
                g_pair_tf_window[key] = {"window": win}
            log.info(f"[API] Пара {key}: window={win}")
            return {"ok": True, "key": key, "window": win}

        # ── POST /api/pairs/{key}/pause ───────────────────────────────────────
        @app.post("/api/pairs/{key}/pause")
        def pause_pair(key: str):
            with g_lock:
                existing = [f"{a}_{b}" for a, b in g_pairs]
                if key not in existing:
                    raise HTTPException(404, f"Pair {key} not found")
                g_paused_pairs.add(key)
            log.info(f"[API] Пара {key} приостановлена")
            tg_notify(f"⏸️ Пара <b>{key}</b> приостановлена")
            return {"ok": True, "key": key, "paused": True}

        # ── POST /api/pairs/{key}/resume ──────────────────────────────────────
        @app.post("/api/pairs/{key}/resume")
        def resume_pair(key: str):
            with g_lock:
                existing = [f"{a}_{b}" for a, b in g_pairs]
                if key not in existing:
                    raise HTTPException(404, f"Pair {key} not found")
                g_paused_pairs.discard(key)
            log.info(f"[API] Пара {key} возобновлена")
            return {"ok": True, "key": key, "paused": False}

        uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
    except Exception as e:
        log.error(f"API error: {e}")
        tg_error(f"❌ <b>API упал:</b>\n<code>{e}</code>")


# ═══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ ЦИКЛ
# ═══════════════════════════════════════════════════════════════════════════════

def seconds_to_next_candle(tf_min: int = 5) -> float:
    now     = datetime.now(timezone.utc)
    elapsed = (now.minute % tf_min) * 60 + now.second
    return tf_min * 60 - elapsed + 2


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    log.info("Инициализация бота...")
    client = get_client()

    # load_state первым — он восстанавливает g_pairs из файла
    load_state()
    load_history()   # история сделок — из отдельного файла

    precisions  = {}
    all_symbols = list(set(s for pair in g_pairs for s in pair))
    for sym in all_symbols:
        try:
            client.futures_change_leverage(symbol=sym, leverage=LEVERAGE)
        except Exception as e:
            log.warning(f"Плечо {sym}: {e}")
        try:
            client.futures_change_margin_type(symbol=sym, marginType=MARGIN_TYPE)
            log.info(f"  {sym}: marginType={MARGIN_TYPE}")
        except Exception as e:
            if "No need to change" not in str(e):
                log.warning(f"Маржа {sym}: {e}")
        precisions[sym] = get_precision(client, sym)
        log.info(f"  {sym}: precision={precisions[sym]}")

    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    log.info(f"API запущен на http://localhost:{API_PORT}")

    tf_min = int(TIMEFRAME[:-1])
    log.info(f"Бот запущен. Пар: {len(g_pairs)}. PNL порог алерта: ${PNL_WARN_THRESHOLD}. Ждём закрытия свечи...")

    while True:
        sleep_sec = seconds_to_next_candle(tf_min)
        log.info(f"Ждём {sleep_sec:.0f}s до следующей свечи [{TIMEFRAME}]...")
        time.sleep(sleep_sec)

        with g_lock:
            active_pairs = [(a, b) for a, b in g_pairs if f"{a}_{b}" not in g_paused_pairs]

        for sym_a, sym_b in active_pairs:
            if sym_a not in precisions:
                try: precisions[sym_a] = get_precision(client, sym_a)
                except: precisions[sym_a] = 3
            if sym_b not in precisions:
                try: precisions[sym_b] = get_precision(client, sym_b)
                except: precisions[sym_b] = 3
            try:
                process_pair(client, sym_a, sym_b, precisions[sym_a], precisions[sym_b])
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error(f"[{sym_a}_{sym_b}] Ошибка: {e}", exc_info=True)
                tg_error(f"⚠️ <b>{sym_a}/{sym_b}</b> Ошибка:\n<code>{e}</code>")

        # ── PNL МОНИТОРИНГ ────────────────────────────────────────────────────
        try:
            check_unrealized_pnl(client)
        except Exception as e:
            log.error(f"[PNL_CHECK] Ошибка: {e}")

        save_state()
        save_history()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Бот остановлен.")