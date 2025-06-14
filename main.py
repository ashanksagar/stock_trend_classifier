from datetime import date
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.train_model_optuna import ModelTrainer
from src.backtester import Backtester
import os


# List of tickers to train
tickers = [
    "MSFT", "NVDA", "AAPL", "AMZN", "GOOG", "GOOGL", "META", "AVGO", "BRK.B", "TSLA",
    "WMT", "JPM", "LLY", "V", "MA", "ORCL", "NFLX", "XOM", "COST", "PG",
    "JNJ", "HD", "ABBV", "BAC", "PLTR", "KO", "PM", "UNH", "TMUS", "IBM",
    "GE", "CSCO", "CRM", "CVX", "WFC", "ABT", "LIN", "MCD", "DIS", "INTU",
    "MS", "AXP", "NOW", "T", "MRK", "ACN", "AMD", "RTX", "GS", "ISRG"
]

# Date range (2024 to today)
start_date = "2024-01-01"
end_date = str(date.today())

# Where models will be saved
model_dir = "models"

for ticker in tickers:
    print(f"\nTraining model for: {ticker}")

    # Check if model already exists
    model_path = os.path.join(model_dir, f"xgb_{ticker}.pkl")
    if os.path.exists(model_path):
        print(f"Model already exists for {ticker}, skipping.")
        continue

    try:
        # Step 1: Load data
        loader = StockDataLoader(ticker, start_date, end_date)
        df = loader.download(save_csv=True)

        if df is None or df.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        # Step 2: Feature engineering
        fe = FeatureEngineer(df)
        featured_df = fe.get_featured_data()

        if featured_df.empty or len(featured_df) < 30:
            print(f"Not enough usable data for {ticker}. Skipping.")
            continue

        # Step 3: Train model
        trainer = ModelTrainer(featured_df)
        model = trainer.train_and_evaluate()

        # Step 4: Backtest
        backtester = Backtester(model, featured_df)
        metrics = backtester.run()

        print(f"Backtest Results for {ticker}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error training model for {ticker}: {e}")
