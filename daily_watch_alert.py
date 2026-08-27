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
import smtplib
from email.mime.text import MIMEText
import requests
import io
import yfinance as yf
import pandas as pd

# ---- STOCK UNIVERSE ----
# Set to True to automatically watch all ~500 stocks in the NIFTY 500 index
# (a broad list covering large, mid, and small companies on the NSE).
# Set to False to use your own manual list in MANUAL_TICKERS instead.
USE_NIFTY_500 = True

# Only used if USE_NIFTY_500 is False
MANUAL_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# How many tickers to download from Yahoo Finance at once. Batching avoids
# rate-limit errors and timeouts that can happen if you request 500 at once.
BATCH_SIZE = 50
# Small pause between batches, in seconds, to be polite to Yahoo Finance's servers.
BATCH_DELAY = 2

SMA_FAST = 20
SMA_SLOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

# Email settings are read from environment variables (kept secret, set in
# GitHub Actions "Secrets" — see README.txt). Never type your password
# directly into this file.
EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD")  # Gmail "App Password"
EMAIL_TO = os.environ.get("ALERT_EMAIL_TO")

# NSE's official NIFTY 500 constituent list (updated periodically by NSE itself)
NIFTY_500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


def get_nifty500_tickers():
    """
    Downloads the current NIFTY 500 list directly from NSE and converts the
    symbols into Yahoo Finance format (e.g. RELIANCE -> RELIANCE.NS).
    Falls back to the manual list if the download fails for any reason
    (NSE occasionally blocks automated requests without proper headers).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(NIFTY_500_URL, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        tickers = [f"{sym}.NS" for sym in symbols]
        print(f"Loaded {len(tickers)} tickers from NIFTY 500 list.")
        return tickers
    except Exception as e:
        print(f"Could not fetch NIFTY 500 list ({e}). Falling back to manual list.")
        return MANUAL_TICKERS


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate_df(ticker, df):
    """Runs the SMA/RSI signal logic on a single ticker's price history."""
    if df is None or df.empty or len(df) < SMA_SLOW + 2:
        return None

    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(SMA_FAST).mean()
    df["SMA_slow"] = df["Close"].rolling(SMA_SLOW).mean()
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = (prev["SMA_fast"] <= prev["SMA_slow"]) and (latest["SMA_fast"] > latest["SMA_slow"])
    crossed_down = (prev["SMA_fast"] >= prev["SMA_slow"]) and (latest["SMA_fast"] < latest["SMA_slow"])

    if crossed_up and latest["RSI"] < RSI_OVERBOUGHT:
        return f"{ticker}: BUY signal — price {latest['Close']:.2f}, RSI {latest['RSI']:.1f}"
    elif crossed_down or latest["RSI"] > RSI_OVERBOUGHT:
        reason = "RSI overbought" if latest["RSI"] > RSI_OVERBOUGHT else "SMA crossed down"
        return f"{ticker}: SELL signal ({reason}) — price {latest['Close']:.2f}, RSI {latest['RSI']:.1f}"
    return None


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
                print(result or f"{ticker}: no signal today")
            except Exception as e:
                print(f"{ticker}: error processing ({e})")

        if start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    return results


def send_email(alerts):
    body = "Your daily stock watch triggered these signals:\n\n" + "\n".join(alerts)
    body += "\n\nReminder: rule-based signal only, not financial advice."
    msg = MIMEText(body)
    msg["Subject"] = f"Stock Alert: {len(alerts)} signal(s) today"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)


def main():
    tickers = get_nifty500_tickers() if USE_NIFTY_500 else MANUAL_TICKERS

    results = check_tickers_in_batches(tickers)
    alerts = [r for _, r in results if r]

    if alerts and EMAIL_FROM and EMAIL_PASSWORD and EMAIL_TO:
        send_email(alerts)
        print(f"\nEmail sent with {len(alerts)} alert(s).")
    elif alerts:
        print("\nSignals found but email isn't configured yet (see README.txt).")
    else:
        print("\nNo signals today — no email sent.")


if __name__ == "__main__":
    main()
