import yfinance as yf
import pandas as pd
import os

class StockDataLoader:
    def __init__(self, ticker, start_date, end_date, data_dir="data"):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.data = None

    def _flatten_and_clean_columns(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
        df.columns = [col.replace(f"_{self.ticker}", "") for col in df.columns]
        return df

    def download(self, save_csv=True):
        print(f"Downloading {self.ticker} from {self.start_date} to {self.end_date}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval="1d")

        if df.empty:
            print(f"No data returned for {self.ticker}.")
            return None

        df = self._flatten_and_clean_columns(df)
        df.reset_index(inplace=True)
        df["Ticker"] = self.ticker
        self.data = df

        if save_csv:
            os.makedirs(self.data_dir, exist_ok=True)
            path = os.path.join(self.data_dir, f"{self.ticker}.csv")
            df.to_csv(path, index=False)
            print(f"Saved to {path}")

        return df

    def load_from_csv(self):
        path = os.path.join(self.data_dir, f"{self.ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved CSV found at {path}. Please download first.")
        self.data = pd.read_csv(path, parse_dates=["Date"])
        print(f"Loaded {self.ticker} data from {path}")
        return self.data

    def get_data(self):
        if self.data is None:
            raise ValueError("No data loaded. Use `download()` or `load_from_csv()` first.")
        return self.data
