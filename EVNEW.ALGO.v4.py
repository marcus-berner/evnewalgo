import requests
from time import sleep
import numpy as np

# =========================
# CONFIG
# =========================
API_KEY = "F0Q44YQM"
BASE_URL = "http://localhost:9999/v1"

TICKERS = ["TP", "AS", "BA"]

MAX_GROSS = 100000
MAX_NET = 50000
ORDER_LIMIT = 10000
COMMISSION = 0.01
SLEEP = 0.20

# Per-ticker caps (press BA, keep others smaller)
MAX_POS = {"TP": 15000, "AS": 15000, "BA": 45000}

# Controls to avoid chop
COOLDOWN_TICKS = 3                 # per ticker cooldown
MAX_ORDERS_PER_20TICKS = 6         # per ticker rate limit
WINDOW = 20

# Late-game behavior
NO_NEW_TRADES_AFTER = 296
FLATTEN_START = 298
FLATTEN_STEP = 5000

# Kill switch (stop trading if things go bad)
STOP_TRADING_NLV = -75000          # adjust if you want tighter/looser

# =========================
# CASE CONSTANTS
# =========================
LAST_YEAR = np.array([
    [0.40, 0.33, 0.33, 0.37],  # TP
    [0.35, 0.45, 0.50, 0.25],  # AS
    [0.15, 0.50, 0.60, 0.25],  # BA
])

COST_OF_EQUITY = [0.05, 0.075, 0.10]
PAYOUT = [0.80, 0.50, 0.00]
RETAIL_PE = [12, 16, 20]

EPS_RANGE = [0.02, 0.04, 0.06]
OWN_RANGE = [20, 25, 30]

# =========================
# SESSION
# =========================
s = requests.Session()
s.headers.update({"X-API-key": API_KEY})


# =========================
# API HELPERS
# =========================
def get_tick():
    r = s.get(f"{BASE_URL}/case")
    if r.ok:
        j = r.json()
        return j["tick"], j["status"]
    return None, None


def get_book(ticker):
    r = s.get(f"{BASE_URL}/securities/book", params={"ticker": ticker})
    if not r.ok:
        return None, None
    j = r.json()
    if (not j.get("bids")) or (not j.get("asks")):
        return None, None
    return float(j["bids"][0]["price"]), float(j["asks"][0]["price"])


def get_positions():
    r = s.get(f"{BASE_URL}/securities")
    if not r.ok:
        return 0, 0, {}
    book = r.json()
    pos = {x["ticker"]: int(x.get("position", 0)) for x in book}
    gross = sum(abs(pos.get(t, 0)) for t in TICKERS)
    net = sum(pos.get(t, 0) for t in TICKERS)
    return gross, net, pos


def get_trader_nlv():
    # RIT typically exposes /trader with NLV
    r = s.get(f"{BASE_URL}/trader")
    if not r.ok:
        return None
    j = r.json()
    # keys vary; try common ones
    for k in ["nlv", "NLV", "net_liquidation_value"]:
        if k in j:
            return float(j[k])
    return None


def place_order(ticker, action, qty, order_type="MARKET", price=None):
    qty = int(max(0, min(qty, ORDER_LIMIT)))
    if qty <= 0:
        return False

    params = {"ticker": ticker, "type": order_type, "quantity": qty, "action": action}
    if order_type == "LIMIT":
        params["price"] = float(price)

    resp = s.post(f"{BASE_URL}/orders", params=params)
    if not resp.ok:
        # keep this short so console doesn't become unusable
        print(f"ORDER FAILED {ticker} {action} {order_type} qty={qty} | {resp.status_code}")
        return False
    return True


# =========================
# NEWS PARSING (lightweight)
# =========================
def get_news(eps_est, own_est, eps_actual, has_own, has_any_eps):
    r = s.get(f"{BASE_URL}/news", params={"limit": 50})
    if not r.ok:
        return eps_est, own_est, eps_actual, has_own, has_any_eps

    news = r.json()

    for item in news[::-1]:
        headline = item.get("headline", "")
        body = item.get("body", "")

        for idx, ticker in enumerate(TICKERS):
            if (ticker in headline) and ("Analyst" in headline):
                for q in range(4):
                    key = f"Q{q+1}:"
                    loc = body.find(key)
                    if loc != -1:
                        try:
                            eps_est[idx, q] = float(body[loc + 5: loc + 9])
                            has_any_eps[idx] = True
                        except:
                            pass

        for idx, ticker in enumerate(TICKERS):
            if (ticker in headline) and ("institutional" in headline):
                pct_loc = body.find("%")
                if pct_loc > 0:
                    try:
                        own_est[idx] = float(body[pct_loc - 5: pct_loc])
                        has_own[idx] = True
                    except:
                        pass

        if "Earnings release" in headline:
            for idx, ticker in enumerate(TICKERS):
                for q in range(4):
                    key = f"{ticker} Q{q+1}:"
                    loc = body.find(key)
                    if loc != -1:
                        try:
                            eps_actual[idx, q] = float(body[loc + 32: loc + 36])
                        except:
                            pass

    return eps_est, own_est, eps_actual, has_own, has_any_eps


# =========================
# VALUATION ENGINE
# =========================
def compute_value(idx, eps_total, ownership_pct):
    last_total = float(LAST_YEAR[idx].sum())
    g = (eps_total / last_total) - 1.0

    if idx < 2:
        div = eps_total * PAYOUT[idx]
        ke = COST_OF_EQUITY[idx]
        stage1 = ((div * (1 + g)) / (ke - g)) * (1 - ((1 + g) / (1 + ke)) ** 5)
        stage2 = ((div * ((1 + g) ** 5) * (1 + 0.02)) / (ke - 0.02)) / ((1 + ke) ** 5)
        inst_val = stage1 + stage2
        retail_val = eps_total * RETAIL_PE[idx]
    else:
        inst_val = eps_total * 20 * (1 + g)
        retail_val = eps_total * 20

    w = ownership_pct / 100.0
    return w * inst_val + (1 - w) * retail_val


def edge_threshold(tick, ticker, is_market):
    """
    Require bigger edge for market orders (since you pay spread).
    Smaller edge ok for limit orders (maker-first).
    """
    time_factor = max(0.02, (300 - tick) / 300.0)

    # base buffer beyond commission
    base = 0.010 * time_factor + 0.002  # early ~0.012, late ~0.002

    # BA can be a bit more aggressive
    if ticker == "BA":
        base *= 0.80
    else:
        base *= 1.20

    if is_market:
        # market needs more edge to justify spread impact
        return COMMISSION + base + 0.020
    return COMMISSION + base


def qty_from_edge(edge, ticker):
    if edge < 0.03:
        q = 2500
    elif edge < 0.06:
        q = 5000
    elif edge < 0.10:
        q = 7500
    else:
        q = 10000
    if ticker != "BA":
        q = int(q * 0.6)
    return max(1000, min(q, ORDER_LIMIT))


# =========================
# MAIN
# =========================
def main():
    print("EVNEW bot starting...")

    # Wait until ACTIVE
    while True:
        tick, status = get_tick()
        print(f"Case status = {status}, tick = {tick}")
        if status == "ACTIVE":
            break
        sleep(0.5)

    print("Case is ACTIVE. Entering trading loop...")

    eps_est = np.zeros((3, 4))
    eps_actual = np.zeros((3, 4))
    eps_path = LAST_YEAR.copy()

    own_est = np.array([50.0, 50.0, 50.0])
    has_own = [False, False, False]
    has_any_eps = [False, False, False]

    last_trade_tick = {t: -999 for t in TICKERS}
    orders_in_window = {t: 0 for t in TICKERS}
    window_start = 0

    while True:
        tick, status = get_tick()
        if status != "ACTIVE":
            sleep(0.25)
            continue

        # reset per-window counters
        if tick - window_start >= WINDOW:
            window_start = tick
            orders_in_window = {t: 0 for t in TICKERS}

        # kill switch
        nlv = get_trader_nlv()
        if nlv is not None and nlv <= STOP_TRADING_NLV:
            print(f"NLV hit kill switch ({nlv:.2f}). Flattening and stopping new trades.")
            # flatten
            _, _, pos = get_positions()
            for t in TICKERS:
                p = pos.get(t, 0)
                if p > 0:
                    place_order(t, "SELL", min(ORDER_LIMIT, p), "MARKET")
                elif p < 0:
                    place_order(t, "BUY", min(ORDER_LIMIT, abs(p)), "MARKET")
            sleep(SLEEP)
            continue

        gross, net, pos = get_positions()

        # endgame flatten
        if tick >= FLATTEN_START:
            for t in TICKERS:
                p = pos.get(t, 0)
                if p > 0:
                    place_order(t, "SELL", min(FLATTEN_STEP, p), "MARKET")
                elif p < 0:
                    place_order(t, "BUY", min(FLATTEN_STEP, abs(p)), "MARKET")
            sleep(SLEEP)
            continue

        if tick >= NO_NEW_TRADES_AFTER:
            sleep(SLEEP)
            continue

        # Update news
        eps_est, own_est, eps_actual, has_own, has_any_eps = get_news(
            eps_est, own_est, eps_actual, has_own, has_any_eps
        )

        # EPS path
        for i in range(3):
            for q in range(4):
                if eps_actual[i, q] != 0:
                    eps_path[i, q] = eps_actual[i, q]
                elif eps_est[i, q] != 0:
                    eps_path[i, q] = eps_est[i, q]

        time_factor = max(0.02, (300 - tick) / 300.0)

        # Compute mid + uncertainty box
        mid = [0.0] * 3
        lo = [0.0] * 3
        hi = [0.0] * 3

        for i, ticker in enumerate(TICKERS):
            eps_total = float(eps_path[i].sum())

            # uncertainty scales down if we have no good signals yet
            eps_scale = 1.0 if has_any_eps[i] else 0.60
            eps_uncert = eps_scale * EPS_RANGE[i] * 4 * time_factor
            eps_lo = eps_total - eps_uncert
            eps_hi = eps_total + eps_uncert

            own_uncert = (OWN_RANGE[i] * time_factor) if has_own[i] else (8.0 * time_factor)
            own_lo = max(0.0, float(own_est[i]) - own_uncert)
            own_hi = min(100.0, float(own_est[i]) + own_uncert)

            mid[i] = compute_value(i, eps_total, float(own_est[i]))

            v1 = compute_value(i, eps_lo, own_lo)
            v2 = compute_value(i, eps_lo, own_hi)
            v3 = compute_value(i, eps_hi, own_lo)
            v4 = compute_value(i, eps_hi, own_hi)
            lo[i] = min(v1, v2, v3, v4)
            hi[i] = max(v1, v2, v3, v4)

        # Print occasionally
        if tick % 10 == 0:
            print(f"[tick={tick}] gross={gross} net={net} pos={pos} nlv={nlv}")

        # Trading: BA-first
        for i, ticker in enumerate(TICKERS):
            # cooldown
            if tick - last_trade_tick[ticker] <= COOLDOWN_TICKS:
                continue
            # rate limit
            if orders_in_window[ticker] >= MAX_ORDERS_PER_20TICKS:
                continue

            bid, ask = get_book(ticker)
            if bid is None:
                continue

            # Only trade TP/AS on BIG extremes (reduces chop)
            if ticker != "BA":
                # require price outside the full box by a margin
                sell_edge = bid - hi[i]
                buy_edge = lo[i] - ask
                is_market = True
                thresh = edge_threshold(tick, ticker, is_market=True)

                # enforce ticker cap space
                p = pos.get(ticker, 0)
                cap_long = min(MAX_POS[ticker] - p, MAX_NET - net, MAX_GROSS - gross)
                cap_short = min(MAX_POS[ticker] + p, MAX_NET + net, MAX_GROSS - gross)

                if sell_edge > thresh and cap_short > 0:
                    qty = min(qty_from_edge(sell_edge, ticker), cap_short, ORDER_LIMIT)
                    if place_order(ticker, "SELL", qty, "MARKET"):
                        orders_in_window[ticker] += 1
                        last_trade_tick[ticker] = tick
                    continue

                if buy_edge > thresh and cap_long > 0:
                    qty = min(qty_from_edge(buy_edge, ticker), cap_long, ORDER_LIMIT)
                    if place_order(ticker, "BUY", qty, "MARKET"):
                        orders_in_window[ticker] += 1
                        last_trade_tick[ticker] = tick
                    continue

                continue  # TP/AS done

            # BA logic: maker-first around mid with a tighter band
            width = max(mid[i] - lo[i], hi[i] - mid[i])
            # trade band around mid (tight enough to trade, but not micro-chop)
            band = max(0.35 * width, 0.06)  # floor prevents too-tight bands

            trade_low = mid[i] - band
            trade_high = mid[i] + band

            sell_edge = bid - trade_high
            buy_edge = trade_low - ask

            # capacities
            p = pos.get(ticker, 0)
            cap_long = min(MAX_POS[ticker] - p, MAX_NET - net, MAX_GROSS - gross)
            cap_short = min(MAX_POS[ticker] + p, MAX_NET + net, MAX_GROSS - gross)

            # Decide order type
            # - LIMIT for modest edge (avoid spread)
            # - MARKET for big edge (take it now)
            mkt_thresh = edge_threshold(tick, ticker, is_market=True)
            lim_thresh = edge_threshold(tick, ticker, is_market=False)

            # SELL BA
            if sell_edge > lim_thresh and cap_short > 0:
                qty = min(qty_from_edge(sell_edge, ticker), cap_short, ORDER_LIMIT)
                if sell_edge > mkt_thresh:
                    ok = place_order(ticker, "SELL", qty, "MARKET")
                else:
                    ok = place_order(ticker, "SELL", qty, "LIMIT", price=bid)
                if ok:
                    orders_in_window[ticker] += 1
                    last_trade_tick[ticker] = tick

            # BUY BA
            if buy_edge > lim_thresh and cap_long > 0:
                qty = min(qty_from_edge(buy_edge, ticker), cap_long, ORDER_LIMIT)
                if buy_edge > mkt_thresh:
                    ok = place_order(ticker, "BUY", qty, "MARKET")
                else:
                    ok = place_order(ticker, "BUY", qty, "LIMIT", price=ask)
                if ok:
                    orders_in_window[ticker] += 1
                    last_trade_tick[ticker] = tick

        sleep(SLEEP)


if __name__ == "__main__":
    main()