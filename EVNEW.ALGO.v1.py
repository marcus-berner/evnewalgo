import requests
from time import sleep
import numpy as np
import math

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
    bid = float(j["bids"][0]["price"])
    ask = float(j["asks"][0]["price"])
    return bid, ask


def get_positions():
    r = s.get(f"{BASE_URL}/securities")
    if not r.ok:
        return 0, 0, {}
    book = r.json()
    pos = {x["ticker"]: int(x.get("position", 0)) for x in book}

    gross = 0
    net = 0
    for t in TICKERS:
        p = pos.get(t, 0)
        gross += abs(p)
        net += p

    return gross, net, pos


def place_order(ticker, action, qty):
    """
    Returns True if order accepted, False otherwise.
    Enforces hard cap at ORDER_LIMIT.
    """
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


def get_news(eps_est, own_est, eps_actual):
    r = s.get(f"{BASE_URL}/news", params={"limit": 50})
    if not r.ok:
        return eps_est, own_est, eps_actual

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
                        except:
                            pass

        for idx, ticker in enumerate(TICKERS):
            if (ticker in headline) and ("institutional" in headline):
                pct_loc = body.find("%")
                if pct_loc > 0:
                    try:
                        own_est[idx] = float(body[pct_loc - 5: pct_loc])
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

    return eps_est, own_est, eps_actual


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

    fail_cooldown = 0  # ticks to wait after an order failure

    while True:
        tick, status = get_tick()
        if status != "ACTIVE":
            sleep(0.25)
            continue

        gross, net, pos = get_positions()
        eps_est, own_est, eps_actual = get_news(eps_est, own_est, eps_actual)

        for i in range(3):
            for q in range(4):
                if eps_actual[i, q] != 0:
                    eps_path[i, q] = eps_actual[i, q]
                elif eps_est[i, q] != 0:
                    eps_path[i, q] = eps_est[i, q]

        time_factor = max(0.05, (300 - tick) / 300.0)

        values_low = [0.0, 0.0, 0.0]
        values_mid = [0.0, 0.0, 0.0]
        values_high = [0.0, 0.0, 0.0]

        for i in range(3):
            eps_total = float(eps_path[i].sum())

            eps_uncert = EPS_RANGE[i] * 4 * time_factor
            eps_low = eps_total - eps_uncert
            eps_high = eps_total + eps_uncert

            own_uncert = OWN_RANGE[i] * time_factor
            own_low = max(0.0, float(own_est[i]) - own_uncert)
            own_high = min(100.0, float(own_est[i]) + own_uncert)

            mid = compute_value(i, eps_total, float(own_est[i]))

            v1 = compute_value(i, eps_low, own_low)
            v2 = compute_value(i, eps_low, own_high)
            v3 = compute_value(i, eps_high, own_low)
            v4 = compute_value(i, eps_high, own_high)

            values_mid[i] = mid
            values_low[i] = min(v1, v2, v3, v4)
            values_high[i] = max(v1, v2, v3, v4)

        # Debug print every 5 ticks
        if tick % 5 == 0:
            print(f"[tick={tick}] gross={gross} net={net} pos={pos}")
            for i, t in enumerate(TICKERS):
                bid, ask = get_book(t)
                if bid is None:
                    continue
                print(f"  {t} bid/ask={bid:.2f}/{ask:.2f} | low/mid/high={values_low[i]:.2f}/{values_mid[i]:.2f}/{values_high[i]:.2f}")

        # If we recently failed an order, pause briefly to avoid spamming
        if fail_cooldown > 0:
            fail_cooldown -= 1
            sleep(SLEEP)
            continue

        # Trading
        for i, ticker in enumerate(TICKERS):
            bid, ask = get_book(ticker)
            if bid is None:
                continue

            sell_edge = bid - values_high[i]
            buy_edge = values_low[i] - ask

            cap_gross = MAX_GROSS - gross
            cap_long = MAX_NET - net
            cap_short = MAX_NET + net

            # Base qty, never above ORDER_LIMIT
            size_mult = min(1.20, 1.0 + 0.20 * time_factor)  # modest
            base_qty = int(ORDER_LIMIT * size_mult)
            base_qty = min(base_qty, ORDER_LIMIT)  # HARD CAP

            # SELL
            if sell_edge > COMMISSION and cap_gross > 0 and cap_short > 0:
                qty = min(base_qty, cap_gross, cap_short, ORDER_LIMIT)
                ok = place_order(ticker, "SELL", qty)
                if not ok:
                    fail_cooldown = 2
                    break
                gross += abs(qty)
                net -= qty

            # BUY
            if buy_edge > COMMISSION and cap_gross > 0 and cap_long > 0:
                qty = min(base_qty, cap_gross, cap_long, ORDER_LIMIT)
                ok = place_order(ticker, "BUY", qty)
                if not ok:
                    fail_cooldown = 2
                    break
                gross += abs(qty)
                net += qty

        sleep(SLEEP)


if __name__ == "__main__":
    main()