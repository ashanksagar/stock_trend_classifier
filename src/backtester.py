import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class Backtester:
    def __init__(self, model, featured_df, test_split_ratio=0.2, transaction_cost=0.002, confidence_threshold=0.7):
        """
        model: trained sklearn model
        featured_df: DataFrame with features + Target column
        test_split_ratio: % of data to use for backtesting (default last 20%)
        transaction_cost: percent cost per trade (default 0.2%)
        confidence_threshold: minimum confidence to trigger trade (default 70%)
        """
        self.model = model
        self.df = featured_df.copy()
        self.transaction_cost = transaction_cost
        self.test_split_ratio = test_split_ratio
        self.confidence_threshold = confidence_threshold
        self._prepare()

    def _prepare(self):
        # Use only the test set
        test_size = int(len(self.df) * self.test_split_ratio)
        self.df = self.df.iloc[-test_size:].copy()

        # Strip columns not used for features
        drop_cols = ["Date", "Ticker", "Target", "MA7", "MA14", "Prediction"]
        drop_cols = [col for col in drop_cols if col in self.df.columns]
        self.features = self.df.drop(columns=drop_cols)

        if isinstance(self.features.columns, pd.MultiIndex):
            self.features.columns = [
                "_".join(str(c) for c in col if c and str(c).strip() != "") for c in self.features.columns
            ]

        self.true_labels = self.df["Target"].values
        self.prices = self.df["Close"].values
        self.dates = self.df["Date"].values

    def run(self):
        probs = self.model.predict_proba(self.features)
        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        # Only act on high-confidence "Up" predictions
        returns = np.zeros(len(predictions))
        for i in range(len(predictions) - 1):
            if predictions[i] == 1 and confidence[i] >= self.confidence_threshold:
                raw_return = (self.prices[i + 1] - self.prices[i]) / self.prices[i]
                returns[i + 1] = (1 + raw_return) * (1 - self.transaction_cost) ** 2 - 1

        accuracy = accuracy_score(self.true_labels, predictions)
        cumulative_return = (returns + 1).prod() - 1
        avg_daily_return = returns.mean()
        std_daily_return = returns.std()
        sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

        # Baseline: hold every day
        baseline_returns = np.zeros(len(self.prices))
        for i in range(len(self.prices) - 1):
            baseline_returns[i + 1] = (self.prices[i + 1] - self.prices[i]) / self.prices[i]
        baseline_return = (baseline_returns + 1).prod() - 1

        return {
            "Accuracy": round(accuracy, 4),
            "Cumulative Return": round(cumulative_return * 100, 2),
            "Baseline Return": round(baseline_return * 100, 2),
            "Sharpe Ratio": round(sharpe, 4),
            "Improvement Over Baseline (%)": round((cumulative_return - baseline_return) * 100, 2)
        }
