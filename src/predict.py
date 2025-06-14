import joblib
import os

class StockPredictor:
    def __init__(self, ticker, model_dir="models"):
        model_path = os.path.join(model_dir, f"xgb_{ticker}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model for '{ticker}' not found at {model_path}. Please train it first.")
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

    def predict(self, recent_data_df):
        columns_to_drop = ["Date", "Target", "Ticker", "MA7", "MA14", "Prediction"]
        for col in columns_to_drop:
            if col in recent_data_df.columns:
                recent_data_df = recent_data_df.drop(columns=col)

        predictions = self.model.predict(recent_data_df)
        probabilities = self.model.predict_proba(recent_data_df)
        return predictions, probabilities

