import requests
from time import sleep
import numpy as np

# =========================
# CONFIG
# =========================
API_KEY = "F0Q44YQM"
BASE_URL = "http://localhost:9999/v1"

MAX_GROSS = 100000
MAX_NET = 50000
ORDER_LIMIT = 10000
COMMISSION = 0.01
SLEEP = 0.20

TICKERS = ["TP", "AS", "BA"]

# Per-ticker position caps
MAX_POS = {"TP": 25000, "AS": 25000, "BA": 45000}

# Print + behavior
PRINT_EVERY_TICKS = 10
COOLDOWN_TICKS = 2
NO_NEW_TRADES_AFTER = 296
FLATTEN_START = 298
FLATTEN_STEP = 5000

# Last year EPS baselines
LAST_YEAR = np.array([
    [0.40, 0.33, 0.33, 0.37],  # TP
    [0.35, 0.45, 0.50, 0.25],  # AS
    [0.15, 0.50, 0.60, 0.25],  # BA
])

COST_OF_EQUITY = [0.05, 0.075, 0.10]
PAYOUT = [0.80, 0.50, 0.00]
RETAIL_PE = [12, 16, 20]

# Uncertainty assumptions
EPS_RANGE = [0.02, 0.04, 0.06]
OWN_RANGE = [20, 25, 30]

# Aggressiveness: BA gets full size
QTY_MULT = {"TP": 0.65, "AS": 0.65, "BA": 1.00}

# Trade band tightness (smaller = more trading)
# BA more aggressive (smaller k)
K_TRADE = {"TP": 0.45, "AS": 0.45, "BA": 0.30}

# =========================
# SESSION
# =========================
s = requests.Session()
s.headers.update({"X-API-key": API_KEY})


# =========================
# API HELPERS
# =========================
def get_tick():
    try:
        r = s.get(f"{BASE_URL}/case", timeout=0.75)
        if r.ok:
            j = r.json()
            return j["tick"], j["status"]
    except requests.exceptions.RequestException as e:
        print("API connection failed:", e)
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


def place_order(ticker, action, qty):
    qty = int(max(0, min(qty, ORDER_LIMIT)))
    if qty <= 0:
        return False

    resp = s.post(
        f"{BASE_URL}/orders",
        params={"ticker": ticker, "type": "MARKET", "quantity": qty, "action": action},
    )
    if not resp.ok:
        print(f"ORDER FAILED {ticker} {action} qty={qty} | {resp.status_code}: {resp.text[:200]}")
        return False
    return True


# =========================
# NEWS PARSING
# =========================
def get_news(eps_est, own_est, eps_actual, has_own, has_any_eps):
    r = s.get(f"{BASE_URL}/news", params={"limit": 50})
    if not r.ok:
        return eps_est, own_est, eps_actual, has_own, has_any_eps

    news = r.json()

    for item in news[::-1]:
        headline = item.get("headline", "")
        body = item.get("body", "")

        # Analyst EPS estimates
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

        # Institutional ownership estimate
        for idx, ticker in enumerate(TICKERS):
            if (ticker in headline) and ("institutional" in headline):
                pct_loc = body.find("%")
                if pct_loc > 0:
                    try:
                        own_est[idx] = float(body[pct_loc - 5: pct_loc])
                        has_own[idx] = True
                    except:
                        pass

        # Earnings releases
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


def edge_buffer(tick, ticker):
    # Small extra edge beyond commission to reduce churn
    time_factor = max(0.02, (300 - tick) / 300.0)
    base = 0.012 * time_factor + 0.002  # early ~0.014, late ~0.0023
    if ticker == "BA":
        base *= 0.70
    return COMMISSION + base


def qty_from_edge(edge, cap_gross, cap_net, ticker):
    if edge <= 0:
        return 0

    # Sizing by how wrong it is
    if edge < 0.03:
        base = 2500
    elif edge < 0.06:
        base = 5000
    elif edge < 0.10:
        base = 7500
    else:
        base = 10000

    base = int(base * QTY_MULT.get(ticker, 1.0))
    return int(max(0, min(base, ORDER_LIMIT, cap_gross, cap_net)))


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
    fail_cooldown = 0

    while True:
        tick, status = get_tick()
        if status != "ACTIVE":
            sleep(0.25)
            continue

        gross, net, pos = get_positions()

        # Endgame flatten
        if tick >= FLATTEN_START:
            for t in TICKERS:
                p = pos.get(t, 0)
                if p > 0:
                    place_order(t, "SELL", min(FLATTEN_STEP, p, ORDER_LIMIT))
                elif p < 0:
                    place_order(t, "BUY", min(FLATTEN_STEP, abs(p), ORDER_LIMIT))
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

        # Compute full risk box (low/high) AND mid
        low_box = [0.0] * 3
        mid_val = [0.0] * 3
        high_box = [0.0] * 3

        # Compute trade bands around mid
        trade_low = [0.0] * 3
        trade_high = [0.0] * 3

        for i, ticker in enumerate(TICKERS):
            eps_total = float(eps_path[i].sum())

            # EPS uncertainty: smaller before any analyst info arrives
            eps_scale = 1.0 if has_any_eps[i] else 0.55
            eps_uncert = eps_scale * EPS_RANGE[i] * 4 * time_factor
            eps_lo = eps_total - eps_uncert
            eps_hi = eps_total + eps_uncert

            # Ownership uncertainty: MUCH smaller until we see an ownership news item
            if has_own[i]:
                own_uncert = OWN_RANGE[i] * time_factor
            else:
                own_uncert = 8.0 * time_factor  # small until real ownership update appears

            own_lo = max(0.0, float(own_est[i]) - own_uncert)
            own_hi = min(100.0, float(own_est[i]) + own_uncert)

            mid = compute_value(i, eps_total, float(own_est[i]))

            v1 = compute_value(i, eps_lo, own_lo)
            v2 = compute_value(i, eps_lo, own_hi)
            v3 = compute_value(i, eps_hi, own_lo)
            v4 = compute_value(i, eps_hi, own_hi)

            lo = min(v1, v2, v3, v4)
            hi = max(v1, v2, v3, v4)

            low_box[i], mid_val[i], high_box[i] = lo, mid, hi

            # Trade band width derived from the box, but we only trade a fraction of it
            width = max(mid - lo, hi - mid)
            k = K_TRADE[ticker]

            # Shrink k slightly late game so you converge harder
            k_eff = max(0.18, k * (0.85 + 0.30 * time_factor))

            trade_low[i] = mid - k_eff * width
            trade_high[i] = mid + k_eff * width

        # Debug print
        if tick % PRINT_EVERY_TICKS == 0:
            print(f"[tick={tick}] gross={gross} net={net} pos={{'TP':{pos.get('TP',0)}, 'AS':{pos.get('AS',0)}, 'BA':{pos.get('BA',0)}}}")
            for i, t in enumerate(TICKERS):
                bid, ask = get_book(t)
                if bid is None:
                    continue
                print(
                    f"  {t} bid/ask={bid:.2f}/{ask:.2f} "
                    f"| tradeL/tradeH={trade_low[i]:.2f}/{trade_high[i]:.2f} "
                    f"| mid={mid_val[i]:.2f}"
                )

        # Avoid spamming after failure
        if fail_cooldown > 0:
            fail_cooldown -= 1
            sleep(SLEEP)
            continue

        # Stop opening new positions late
        if tick >= NO_NEW_TRADES_AFTER:
            sleep(SLEEP)
            continue

        # Trading
        for i, ticker in enumerate(TICKERS):
            if tick - last_trade_tick[ticker] <= COOLDOWN_TICKS:
                continue

            bid, ask = get_book(ticker)
            if bid is None:
                continue

            buf = edge_buffer(tick, ticker)

            # Signals now based on TRADE band around MID
            sell_edge = bid - trade_high[i]
            buy_edge = trade_low[i] - ask

            cap_gross = MAX_GROSS - gross
            cap_long = MAX_NET - net
            cap_short = MAX_NET + net

            p = pos.get(ticker, 0)
            cap_ticker_long = MAX_POS[ticker] - p
            cap_ticker_short = MAX_POS[ticker] + p

            # SELL
            if sell_edge > buf and cap_gross > 0 and cap_short > 0 and cap_ticker_short > 0:
                qty = qty_from_edge(sell_edge, cap_gross, min(cap_short, cap_ticker_short), ticker)
                ok = place_order(ticker, "SELL", qty)
                if not ok:
                    fail_cooldown = 2
                    break
                last_trade_tick[ticker] = tick

            # BUY
            if buy_edge > buf and cap_gross > 0 and cap_long > 0 and cap_ticker_long > 0:
                qty = qty_from_edge(buy_edge, cap_gross, min(cap_long, cap_ticker_long), ticker)
                ok = place_order(ticker, "BUY", qty)
                if not ok:
                    fail_cooldown = 2
                    break
                last_trade_tick[ticker] = tick

        sleep(SLEEP)


if __name__ == "__main__":
    main()