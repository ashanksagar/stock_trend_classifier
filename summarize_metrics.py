import os
import pandas as pd
from src.backtester import Backtester
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.predict import StockPredictor

tickers = [f.replace("xgb_", "").replace(".pkl", "") for f in os.listdir("models") if f.endswith(".pkl")]

summary = []

for ticker in tickers:
    try:
        print(f"Analyzing {ticker}...")
        loader = StockDataLoader(ticker, "2024-01-01", str(pd.Timestamp.today().date()))
        df = loader.load_from_csv()

        fe = FeatureEngineer(df)
        featured_df = fe.get_featured_data()

        if featured_df.empty or len(featured_df) < 30:
            continue

        model = StockPredictor(ticker).model
        backtester = Backtester(model, featured_df)
        metrics = backtester.run()

        summary.append({
            "Ticker": ticker,
            **metrics
        })

    except Exception as e:
        print(f"Error on {ticker}: {e}")

# Save summary
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values("Sharpe Ratio", ascending=False)
summary_df.to_csv("models/summary_metrics.csv", index=False)
print("✅ Summary saved to models/summary_metrics.csv")
