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
import smtplib
from email.mime.text import MIMEText
import yfinance as yf
import pandas as pd

# ---- EDIT THIS LIST with your stocks ----
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def check_ticker(ticker):
    df = yf.download(ticker, period="6mo", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < SMA_SLOW + 2:
        return None

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
    alerts = []
    for ticker in TICKERS:
        result = check_ticker(ticker)
        if result:
            alerts.append(result)
        print(result or f"{ticker}: no signal today")

    if alerts and EMAIL_FROM and EMAIL_PASSWORD and EMAIL_TO:
        send_email(alerts)
        print(f"\nEmail sent with {len(alerts)} alert(s).")
    elif alerts:
        print("\nSignals found but email isn't configured yet (see README.txt).")
    else:
        print("\nNo signals today — no email sent.")


if __name__ == "__main__":
    main()
