from datetime import date
import os
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.train_model_optuna import ModelTrainer
from src.backtester import Backtester
import pandas as pd

# Top 50 S&P tickers (custom curated list)
tickers = [
    "MSFT", "NVDA", "AAPL", "AMZN", "GOOG", "GOOGL", "META", "AVGO", "BRK.B", "TSLA",
    "WMT", "JPM", "LLY", "V", "MA", "ORCL", "NFLX", "XOM", "COST", "PG", "JNJ", "HD",
    "ABBV", "BAC", "PLTR", "KO", "PM", "UNH", "TMUS", "IBM", "GE", "CSCO", "CRM",
    "CVX", "WFC", "ABT", "LIN", "MCD", "DIS", "INTU", "MS", "AXP", "NOW", "T", "MRK",
    "ACN", "AMD", "RTX", "GS", "ISRG"
]

start_date = "2024-01-01"
end_date = str(date.today())
model_dir = "models"

for ticker in tickers:
    print(f"Retraining model for: {ticker}")

    try:
        # Step 1: Fetch latest stock data
        loader = StockDataLoader(ticker, start_date, end_date)
        df = loader.download(save_csv=True)

        if df is None or df.empty:
            print(f"No data for {ticker}. Skipping.")
            continue

        # Step 2: Feature engineering
        fe = FeatureEngineer(df)
        featured_df = fe.get_featured_data()

        if featured_df.empty or len(featured_df) < 30:
            print(f"Not enough usable data for {ticker}. Skipping.")
            continue

        # Step 3: Retrain model using best Optuna setup
        trainer = ModelTrainer(featured_df, model_dir=model_dir)
        model = trainer.train_and_evaluate()

        # Step 4: Evaluate performance
        backtester = Backtester(model, featured_df)
        metrics = backtester.run()

        print(f"Backtest for {ticker}:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

    except Exception as e:
        print(f"Error with {ticker}: {e}")
