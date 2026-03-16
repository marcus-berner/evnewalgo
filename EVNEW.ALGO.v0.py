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

TICKERS = ['TP','AS','BA']

LAST_YEAR = np.array([
    [0.40,0.33,0.33,0.37],  # TP
    [0.35,0.45,0.50,0.25],  # AS
    [0.15,0.50,0.60,0.25]   # BA
])

COST_OF_EQUITY = [0.05,0.075,0.10]
PAYOUT = [0.80,0.50,0.00]
RETAIL_PE = [12,16,20]

EPS_RANGE = [0.02,0.04,0.06]  # per quarter per time unit
OWN_RANGE = [20,25,30]        # initial ownership error %

# =========================
# SESSION
# =========================
s = requests.Session()
s.headers.update({'X-API-key': API_KEY})

# =========================
# API HELPERS
# =========================

def get_tick():
    r = s.get(f"{BASE_URL}/case")
    if r.ok:
        j = r.json()
        return j['tick'], j['status']

def get_book(ticker):
    r = s.get(f"{BASE_URL}/securities/book", params={'ticker':ticker})
    if not r.ok:
        return None, None
    j = r.json()
    if not j['bids'] or not j['asks']:
        return None, None
    return j['bids'][0]['price'], j['asks'][0]['price']

def get_positions():
    r = s.get(f"{BASE_URL}/securities")
    if not r.ok:
        return 0,0,{}
    book = r.json()
    pos = {x['ticker']:x['position'] for x in book}
    gross = sum(abs(pos[t]) for t in TICKERS)
    net = sum(pos[t] for t in TICKERS)
    return gross, net, pos

# =========================
# NEWS PARSING
# =========================

def get_news(eps_est, own_est, eps_actual):
    r = s.get(f"{BASE_URL}/news", params={'limit':50})
    if not r.ok:
        return eps_est, own_est, eps_actual
    
    news = r.json()
    
    for item in news[::-1]:
        for idx,ticker in enumerate(TICKERS):

            if ticker in item['headline']:

                # Analyst EPS
                if "Analyst" in item['headline']:
                    for q in range(4):
                        key = f"Q{q+1}:"
                        if key in item['body']:
                            val = float(item['body'][item['body'].find(key)+5:
                                                     item['body'].find(key)+9])
                            eps_est[idx,q] = val

                # Ownership
                if "institutional" in item['headline']:
                    pct_loc = item['body'].find("%")
                    if pct_loc > 0:
                        val = float(item['body'][pct_loc-5:pct_loc])
                        own_est[idx] = val

        # Earnings release
        if "Earnings release" in item['headline']:
            for idx,ticker in enumerate(TICKERS):
                for q in range(4):
                    key = f"{ticker} Q{q+1}:"
                    if key in item['body']:
                        val = float(item['body'][item['body'].find(key)+32:
                                                 item['body'].find(key)+36])
                        eps_actual[idx,q] = val

    return eps_est, own_est, eps_actual

# =========================
# VALUATION ENGINE
# =========================

def compute_value(idx, eps_total, ownership):

    last_total = LAST_YEAR[idx].sum()
    g = (eps_total / last_total) - 1

    if idx < 2:  # TP / AS
        div = eps_total * PAYOUT[idx]
        ke = COST_OF_EQUITY[idx]

        stage1 = ((div*(1+g))/(ke-g)) * (1 - ((1+g)/(1+ke))**5)
        stage2 = ((div*((1+g)**5)*(1+0.02))/(ke-0.02)) / ((1+ke)**5)

        inst_val = stage1 + stage2
        retail_val = eps_total * RETAIL_PE[idx]

    else:  # BA
        inst_val = eps_total * 20 * (1+g)
        retail_val = eps_total * 20

    final = (ownership/100)*inst_val + (1-ownership/100)*retail_val
    return final

# =========================
# MAIN
# =========================

def main():

    eps_est = np.zeros((3,4))
    eps_actual = np.zeros((3,4))
    eps_path = LAST_YEAR.copy()
    own_est = np.array([50.0,50.0,50.0])

    print("EVNEW bot starting...")

    # Wait until case becomes ACTIVE
    while True:
        tick, status = get_tick()
        print(f"Case status = {status}, tick = {tick}")
        if status == "ACTIVE":
            break
        sleep(0.5)

    print("Case is ACTIVE. Entering trading loop...")

    while True:

        gross, net, pos = get_positions()
        eps_est, own_est, eps_actual = get_news(eps_est, own_est, eps_actual)


        # Update EPS path
        for i in range(3):
            for q in range(4):
                if eps_actual[i,q] != 0:
                    eps_path[i,q] = eps_actual[i,q]
                elif eps_est[i,q] != 0:
                    eps_path[i,q] = eps_est[i,q]

        values_low = []
        values_high = []
        values_mid = []

        # Shrinking uncertainty
        time_factor = max(0.05, (300 - tick)/300)

        for i in range(3):

            eps_total = eps_path[i].sum()

            # EPS uncertainty band
            eps_uncert = EPS_RANGE[i] * 4 * time_factor

            eps_low = eps_total - eps_uncert
            eps_high = eps_total + eps_uncert

            # Ownership uncertainty shrinks
            own_uncert = OWN_RANGE[i] * time_factor
            own_low = max(0, own_est[i] - own_uncert)
            own_high = min(100, own_est[i] + own_uncert)

            # Mid valuation
            mid = compute_value(i, eps_total, own_est[i])

            # Proper min/max propagation
            v1 = compute_value(i, eps_low, own_low)
            v2 = compute_value(i, eps_low, own_high)
            v3 = compute_value(i, eps_high, own_low)
            v4 = compute_value(i, eps_high, own_high)

            low = min(v1,v2,v3,v4)
            high = max(v1,v2,v3,v4)

            values_mid.append(mid)
            values_low.append(low)
            values_high.append(high)

        # ======================
        # TRADING LOGIC
        # ======================

        for i,ticker in enumerate(TICKERS):

            bid, ask = get_book(ticker)
            if bid is None:
                continue

            # Distance from band
            sell_edge = bid - values_high[i]
            buy_edge = values_low[i] - ask

            # Dynamic sizing
            capacity_gross = MAX_GROSS - gross
            capacity_net_long = MAX_NET - net
            capacity_net_short = MAX_NET + net

            size_multiplier = min(1.5, 1 + time_factor)

            qty = int(min(ORDER_LIMIT * size_multiplier,
                          capacity_gross))

            if qty <= 0:
                continue

            # SELL
            if sell_edge > COMMISSION and capacity_net_short > 0:
                s.post(f"{BASE_URL}/orders",
                       params={'ticker':ticker,'type':'MARKET',
                               'quantity':qty,'action':'SELL'})

            # BUY
            if buy_edge > COMMISSION and capacity_net_long > 0:
                s.post(f"{BASE_URL}/orders",
                       params={'ticker':ticker,'type':'MARKET',
                               'quantity':qty,'action':'BUY'})

        sleep(SLEEP)
        tick, status = get_tick()

if __name__ == "__main__":
    main()