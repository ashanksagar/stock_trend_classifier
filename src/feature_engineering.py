import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_technical_indicators(self):
        df = self.df

        # Daily returns
        df["daily_return"] = df["Close"].pct_change()

        # Moving averages
        df["5_day_ma"] = df["Close"].rolling(window=5).mean()
        df["10_day_ma"] = df["Close"].rolling(window=10).mean()

        # Volatility
        df["volatility_10"] = df["daily_return"].rolling(window=10).std()

        # Volume change
        df["volume_change"] = df["Volume"].pct_change()

        # RSI (14-day)
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD (12 EMA - 26 EMA)
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands (20-day)
        df["20_day_ma"] = df["Close"].rolling(window=20).mean()
        df["20_day_std"] = df["Close"].rolling(window=20).std()
        df["BB_upper"] = df["20_day_ma"] + (2 * df["20_day_std"])
        df["BB_lower"] = df["20_day_ma"] - (2 * df["20_day_std"])

        self.df = df

    def add_target(self):
        self.df["Target"] = (self.df["Close"].shift(-1) > self.df["Close"]).astype(int)

    def get_featured_data(self):
        self.add_technical_indicators()
        self.add_target()
        return self.df.dropna().reset_index(drop=True)
