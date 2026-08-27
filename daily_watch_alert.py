"""
Daily NSE/BSE Watch & Email Alert
----------------------------------
Checks your chosen stocks every time it runs. If today's price data triggers
a BUY or SELL signal (SMA20/SMA50 crossover + RSI, same logic as before),
it emails you. If nothing has changed, it stays silent — no daily spam.

This file is meant to be run automatically once a day by GitHub Actions
(free, cloud-based, works even if your computer is off). Setup steps are
in README.txt — no coding needed, just following steps and clicking.

NOT financial advice. Rule-based signals only, not a guarantee of profit.
"""

import os
import time
import json
import smtplib
from email.mime.text import MIMEText
import yfinance as yf
import pandas as pd

# ---- STOCK UNIVERSE ----
# Set to True to automatically watch all ~500 stocks in the NIFTY 500 index
# (a broad list covering large, mid, and small companies on the NSE).
# This reads from a local file (NIFTY_500_CSV_PATH below) instead of
# fetching from NSE live, because NSE blocks automated requests from cloud
# servers like GitHub Actions. See README.txt for how to get/update this file.
# Set to False to use your own manual list in MANUAL_TICKERS instead.
USE_NIFTY_500 = True

# Path to the NIFTY 500 list CSV, downloaded from NSE and committed to this
# repo (same folder as this script). Refresh it every few months by
# re-downloading from https://archives.nseindia.com/content/indices/ind_nifty500list.csv
# in your own browser and re-uploading it to the repo.
NIFTY_500_CSV_PATH = "ind_nifty500list.csv"

# Only used if USE_NIFTY_500 is False
MANUAL_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# ---- YOUR HOLDINGS (PORTFOLIO MEMORY) ----
# Path to a CSV file where you record stocks you've actually bought, at what
# price. The script re-reads this file every run and reports your live
# profit/loss on each one. This file is your "memory" — edit it on GitHub
# (pencil icon) whenever you buy or sell something. Columns:
#   Symbol,BuyPrice,BuyDate,TargetPrice,StopLoss
# TargetPrice and StopLoss are optional — leave blank if you don't want an
# automatic sell alert at a specific price.
HOLDINGS_CSV_PATH = "holdings.csv"

# Minimum profit percentage you're happy to book automatically. If a
# holding doesn't have a specific TargetPrice set in holdings.csv, this
# percentage is used instead - so you always get a sell alert once a
# holding gains at least this much, even if you never set a target price.
# Also used to suggest a sell price alongside every new BUY signal below.
DEFAULT_PROFIT_TARGET_PCT = 5

# If a holding doesn't have a specific StopLoss set in holdings.csv, this
# percentage drop from your buy price triggers an automatic stop-loss
# alert instead.
DEFAULT_STOP_LOSS_PCT = 6

# ---- PARTIAL PROFIT BOOKING ----
# When a holding first reaches DEFAULT_PROFIT_TARGET_PCT profit, instead
# of a one-time "sell everything" alert, you'll be told to book profit on
# HALF the position and let the rest ride. The remaining half is then
# tracked with a trailing stop: if the price falls TRAILING_STOP_PCT from
# its peak (after the partial sell), you'll get a final alert to exit the
# rest.
TRAILING_STOP_PCT = 3

# File used to remember what's already been alerted, so you don't get the
# same "consider selling" email every single day. This file is updated by
# the script and committed back to your repo automatically by the
# workflow - you never need to edit it yourself.
ALERT_STATE_PATH = "alert_state.json"

# ---- SECTOR / DIVERSIFICATION CHECK ----
# Warns you if too many of your holdings are concentrated in one industry
# (e.g. all banking stocks). Uses the Industry column from your NIFTY 500
# CSV to look up each holding's sector. Set the % that counts as "too
# concentrated".
SECTOR_CONCENTRATION_WARNING_PCT = 40

# How many tickers to download from Yahoo Finance at once. Batching avoids
# rate-limit errors and timeouts that can happen if you request 500 at once.
BATCH_SIZE = 50
# Small pause between batches, in seconds, to be polite to Yahoo Finance's servers.
BATCH_DELAY = 2

SMA_FAST = 20
SMA_SLOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

# ---- ADDITIONAL "CHART READING" INDICATORS ----
# These mimic what a trader would check on a TradingView-style chart beyond
# just moving averages: momentum (MACD), volatility (Bollinger Bands), and
# whether a move is backed by real trading activity (Volume). A signal only
# counts as CONFIRMED when enough of these indicators agree, which cuts
# down on false signals compared to using SMA/RSI alone.
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

VOLUME_AVG_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.5  # today's volume must be 1.5x the 20-day average to "confirm"

# Minimum number of confirming indicators (out of MACD, Bollinger, Volume)
# required, in addition to the base SMA/RSI signal, for a signal to be
# reported. Lower = more signals but more false positives. Higher = fewer,
# stronger signals.
MIN_CONFIRMATIONS = 2

# Email settings are read from environment variables (kept secret, set in
# GitHub Actions "Secrets" — see README.txt). Never type your password
# directly into this file.
EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD")  # Gmail "App Password"
EMAIL_TO = os.environ.get("ALERT_EMAIL_TO")

def get_nifty500_tickers():
    """
    Reads the NIFTY 500 list from a local CSV file committed to this repo
    (downloaded manually from NSE, since NSE blocks live requests from cloud
    servers such as GitHub Actions). Converts symbols into Yahoo Finance
    format (e.g. RELIANCE -> RELIANCE.NS). Falls back to the manual list if
    the file is missing or can't be read.
    """
    try:
        df = pd.read_csv(NIFTY_500_CSV_PATH)
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        tickers = [f"{sym}.NS" for sym in symbols]
        print(f"Loaded {len(tickers)} tickers from {NIFTY_500_CSV_PATH}.")
        return tickers
    except Exception as e:
        print(f"Could not read {NIFTY_500_CSV_PATH} ({e}). Falling back to manual list.")
        return MANUAL_TICKERS


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD shows momentum: is the trend speeding up or slowing down.
    Returns (macd_line, signal_line). A crossover of macd_line above
    signal_line suggests strengthening upward momentum, and vice versa.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(series, period=20, std_dev=2):
    """
    Bollinger Bands show volatility: a price near the upper band is
    "stretched" high (often overbought), near the lower band is
    "stretched" low (often oversold).
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def evaluate_df(ticker, df):
    """
    Runs the full signal logic on a single ticker's price history:
    1. Base signal: SMA20/SMA50 crossover + RSI (same as before).
    2. Confirmation check: MACD momentum, Bollinger Band position, and
       Volume surge - like a trader cross-checking multiple parts of a
       chart before acting on it.
    A signal is only reported if the base signal fires AND at least
    MIN_CONFIRMATIONS of the 3 confirming indicators agree.
    """
    if df is None or df.empty or len(df) < max(SMA_SLOW, MACD_SLOW, BOLLINGER_PERIOD) + 2:
        return None

    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(SMA_FAST).mean()
    df["SMA_slow"] = df["Close"].rolling(SMA_SLOW).mean()
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["MACD"], df["MACD_signal"] = compute_macd(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = compute_bollinger_bands(
        df["Close"], BOLLINGER_PERIOD, BOLLINGER_STD_DEV
    )
    df["Vol_avg"] = df["Volume"].rolling(VOLUME_AVG_PERIOD).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = (prev["SMA_fast"] <= prev["SMA_slow"]) and (latest["SMA_fast"] > latest["SMA_slow"])
    crossed_down = (prev["SMA_fast"] >= prev["SMA_slow"]) and (latest["SMA_fast"] < latest["SMA_slow"])

    base_signal = None
    if crossed_up and latest["RSI"] < RSI_OVERBOUGHT:
        base_signal = "BUY"
    elif crossed_down or latest["RSI"] > RSI_OVERBOUGHT:
        base_signal = "SELL"

    if base_signal is None:
        return None

    # ---- Confirmation checks ----
    confirmations = []

    # 1. MACD momentum agrees with the direction of the base signal
    macd_bullish = latest["MACD"] > latest["MACD_signal"]
    if (base_signal == "BUY" and macd_bullish) or (base_signal == "SELL" and not macd_bullish):
        confirmations.append("MACD")

    # 2. Bollinger Band position agrees (price stretched in the signal's direction)
    if base_signal == "BUY" and latest["Close"] <= latest["BB_lower"] * 1.02:
        confirmations.append("Bollinger")
    elif base_signal == "SELL" and latest["Close"] >= latest["BB_upper"] * 0.98:
        confirmations.append("Bollinger")

    # 3. Volume surge - move is backed by real trading activity
    volume_surge = (
        pd.notna(latest["Vol_avg"]) and latest["Vol_avg"] > 0
        and latest["Volume"] >= latest["Vol_avg"] * VOLUME_SURGE_MULTIPLIER
    )
    if volume_surge:
        confirmations.append("Volume")

    if len(confirmations) < MIN_CONFIRMATIONS:
        return None

    confirmed_by = ", ".join(confirmations)
    reason = "RSI overbought" if base_signal == "SELL" and latest["RSI"] > RSI_OVERBOUGHT else (
        "SMA crossed down" if base_signal == "SELL" else "SMA crossed up"
    )

    if base_signal == "BUY":
        suggested_sell = latest["Close"] * (1 + DEFAULT_PROFIT_TARGET_PCT / 100)
        return (
            f"{ticker}: BUY signal ({reason}) — buy price {latest['Close']:.2f}, "
            f"suggested sell price {suggested_sell:.2f} (+{DEFAULT_PROFIT_TARGET_PCT}%) — "
            f"RSI {latest['RSI']:.1f} — confirmed by: {confirmed_by}"
        )

    return (
        f"{ticker}: {base_signal} signal ({reason}) — price {latest['Close']:.2f}, "
        f"RSI {latest['RSI']:.1f} — confirmed by: {confirmed_by}"
    )


def check_tickers_in_batches(tickers):
    """
    Downloads price history for many tickers at once, in batches, and runs
    the signal logic on each one. Batching is much faster and more reliable
    than downloading 500 tickers one at a time.
    """
    results = []
    total = len(tickers)

    for start in range(0, total, BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        print(f"Checking tickers {start + 1}-{min(start + BATCH_SIZE, total)} of {total}...")

        try:
            data = yf.download(
                batch,
                period="6mo",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"Batch download failed ({e}), skipping this batch.")
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[ticker] if ticker in data.columns.get_level_values(0) else None

                if df is None or df.empty:
                    print(f"{ticker}: no data")
                    continue

                result = evaluate_df(ticker, df)
                results.append((ticker, result))
                if result:
                    print(result)
            except Exception as e:
                print(f"{ticker}: error processing ({e})")

        if start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    return results


def load_alert_state():
    """Loads the remembered alert history so we don't repeat the same alert every day."""
    try:
        with open(ALERT_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_alert_state(state):
    """Saves alert history back to disk. The workflow commits this file to your repo."""
    try:
        with open(ALERT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Could not save {ALERT_STATE_PATH} ({e}).")


def load_sector_lookup():
    """
    Builds a Symbol -> Industry lookup from the NIFTY 500 CSV, so we can
    check how concentrated your holdings are in any one sector.
    """
    try:
        df = pd.read_csv(NIFTY_500_CSV_PATH)
        return dict(zip(df["Symbol"].astype(str).str.strip(), df["Industry"].astype(str).str.strip()))
    except Exception as e:
        print(f"Could not load sector info from {NIFTY_500_CSV_PATH} ({e}).")
        return {}


def check_sector_concentration(holding_tickers):
    """
    Warns if too many of your holdings are in the same industry.
    holding_tickers is a list of tickers like ['SBIN.NS', 'HDFCBANK.NS'].
    """
    if not holding_tickers:
        return []

    sector_lookup = load_sector_lookup()
    if not sector_lookup:
        return []

    sector_counts = {}
    for ticker in holding_tickers:
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        sector = sector_lookup.get(symbol, "Unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    total = len(holding_tickers)
    warnings = []
    for sector, count in sector_counts.items():
        pct = (count / total) * 100
        if pct >= SECTOR_CONCENTRATION_WARNING_PCT and sector != "Unknown":
            warnings.append(
                f"{pct:.0f}% of your holdings are in {sector} ({count} of {total} stocks) "
                f"- consider diversifying"
            )
    return warnings


def check_holdings():
    """
    Reads your holdings.csv (stocks you've actually bought) and checks
    today's price against your buy price. Returns a list of readable
    status lines. Also returns the list of holding tickers (for the
    sector concentration check) and any sector warnings.

    Behavior:
    - Reports your live profit/loss on every holding, every run.
    - First time a holding reaches the profit target: alerts you to sell
      HALF the position and let the rest ride (partial profit booking).
      After that, it won't repeat this specific alert again.
    - Once partial profit is booked, the remaining half is tracked with a
      trailing stop: if the price falls TRAILING_STOP_PCT from its peak,
      you get one final alert to exit the rest.
    - If a holding drops to your stop-loss (manual or default), you're
      alerted once. It won't repeat this specific alert again.
    """
    try:
        df = pd.read_csv(HOLDINGS_CSV_PATH)
    except Exception as e:
        print(f"Could not read {HOLDINGS_CSV_PATH} ({e}). Skipping holdings check.")
        return [], [], []

    if df.empty:
        return [], [], []

    state = load_alert_state()
    updates = []
    holding_tickers = []

    for _, row in df.iterrows():
        ticker = str(row["Symbol"]).strip()
        holding_tickers.append(ticker)
        try:
            buy_price = float(row["BuyPrice"])
        except (ValueError, TypeError):
            continue

        target = row.get("TargetPrice")
        stop_loss = row.get("StopLoss")
        target = float(target) if pd.notna(target) and str(target).strip() != "" else None
        stop_loss = float(stop_loss) if pd.notna(stop_loss) and str(stop_loss).strip() != "" else None

        using_default_target = target is None
        if using_default_target:
            target = buy_price * (1 + DEFAULT_PROFIT_TARGET_PCT / 100)

        using_default_stop = stop_loss is None
        if using_default_stop:
            stop_loss = buy_price * (1 - DEFAULT_STOP_LOSS_PCT / 100)

        try:
            data = yf.download(ticker, period="5d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                print(f"{ticker}: no price data found for holdings check")
                continue
            current_price = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"{ticker}: error fetching price for holdings check ({e})")
            continue

        change_pct = ((current_price - buy_price) / buy_price) * 100
        direction = "up" if change_pct >= 0 else "down"

        line = (
            f"{ticker}: bought at {buy_price:.2f}, now {current_price:.2f} "
            f"({direction} {abs(change_pct):.1f}%)"
        )

        ticker_state = state.get(ticker, {})
        alert_note = None

        # ---- Stop-loss check (only before any partial profit is booked) ----
        if not ticker_state.get("partial_sold") and current_price <= stop_loss:
            if not ticker_state.get("stop_loss_alerted"):
                label = "STOP-LOSS HIT" if not using_default_stop else f"STOP-LOSS HIT ({DEFAULT_STOP_LOSS_PCT}% default)"
                alert_note = f" -- {label}, consider selling"
                ticker_state["stop_loss_alerted"] = True
            # if already alerted before, stay quiet (no repeat spam)

        # ---- Profit target / partial booking check ----
        elif not ticker_state.get("partial_sold") and current_price >= target:
            label = "PROFIT TARGET REACHED" if using_default_target else "TARGET REACHED"
            alert_note = (
                f" -- {label} ({abs(change_pct):.1f}% gain) - sell HALF your position "
                f"now, let the rest ride with a trailing stop"
            )
            ticker_state["partial_sold"] = True
            ticker_state["trailing_peak"] = current_price
            ticker_state["stop_loss_alerted"] = False  # reset, no longer relevant post-partial-sell

        # ---- Trailing stop on the remaining half, after partial booking ----
        elif ticker_state.get("partial_sold") and not ticker_state.get("trailing_stop_alerted"):
            peak = max(ticker_state.get("trailing_peak", current_price), current_price)
            ticker_state["trailing_peak"] = peak
            drawdown_pct = ((peak - current_price) / peak) * 100
            if drawdown_pct >= TRAILING_STOP_PCT:
                alert_note = (
                    f" -- TRAILING STOP HIT (fell {drawdown_pct:.1f}% from peak {peak:.2f}) "
                    f"- consider selling your remaining position"
                )
                ticker_state["trailing_stop_alerted"] = True

        if alert_note:
            line += alert_note

        state[ticker] = ticker_state
        updates.append(line)
        print(line)

    save_alert_state(state)

    sector_warnings = check_sector_concentration(holding_tickers)
    for w in sector_warnings:
        print(f"Sector check: {w}")

    return updates, holding_tickers, sector_warnings



def send_email(sections):
    body = "Your daily stock watch update:\n\n" + "\n\n".join(sections)
    body += "\n\nReminder: rule-based signal only, not financial advice."
    msg = MIMEText(body)
    msg["Subject"] = "Stock Alert: Daily Update"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)


def main():
    tickers = get_nifty500_tickers() if USE_NIFTY_500 else MANUAL_TICKERS

    results = check_tickers_in_batches(tickers)
    alerts = [r for _, r in results if r]

    print("\n--- Your Holdings ---")
    holdings_updates, holding_tickers, sector_warnings = check_holdings()

    # Combine signal alerts, holdings updates, and sector warnings into one
    # email so you get your full picture in one place.
    email_sections = []
    if alerts:
        email_sections.append("SIGNALS TODAY:\n" + "\n".join(alerts))
    if holdings_updates:
        email_sections.append("YOUR HOLDINGS:\n" + "\n".join(holdings_updates))
    if sector_warnings:
        email_sections.append("DIVERSIFICATION CHECK:\n" + "\n".join(sector_warnings))

    if email_sections and EMAIL_FROM and EMAIL_PASSWORD and EMAIL_TO:
        send_email(email_sections)
        print(f"\nEmail sent with {len(alerts)} signal(s), {len(holdings_updates)} holdings update(s), "
              f"{len(sector_warnings)} sector warning(s).")
    elif email_sections:
        print("\nUpdates found but email isn't configured yet (see README.txt).")
    else:
        print("\nNothing to report today — no email sent.")


if __name__ == "__main__":
    main()
