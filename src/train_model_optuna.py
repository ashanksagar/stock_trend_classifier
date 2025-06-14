import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import optuna
from src.backtester import Backtester


class ModelTrainer:
    def __init__(self, df, model_dir="models"):
        self.df = df
        self.ticker = df["Ticker"].iloc[0]
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def split_data(self):
        X = self.df.drop(columns=[col for col in ["Date", "Target", "Ticker"] if col in self.df.columns])
        y = self.df["Target"]
        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    def flatten_columns(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c).strip() for c in col if c and str(c).strip() != "")
                for col in df.columns
            ]
        return df

    def strip_ticker_suffixes(self, df):
        df.columns = [
            col.replace(f"_{self.ticker}", "") if isinstance(col, str) else col
            for col in df.columns
        ]
        return df

    def objective(self, trial):
        X_train, X_test, y_train, y_test = self.split_data()
        X_train = self.strip_ticker_suffixes(self.flatten_columns(X_train))
        X_test = self.strip_ticker_suffixes(self.flatten_columns(X_test))

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Backtest on full featured_df to get Sharpe Ratio
        backtester = Backtester(model, self.df)
        metrics = backtester.run()

        return metrics["Sharpe Ratio"]

    def train_and_evaluate(self):
        print(f"Starting Optuna hyperparameter search for {self.ticker}...")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, show_progress_bar=True)

        print(f"Best trial for {self.ticker}:")
        print(study.best_trial.params)

        # Retrain final model with best parameters
        X_train, X_test, y_train, y_test = self.split_data()
        X_train = self.strip_ticker_suffixes(self.flatten_columns(X_train))
        X_test = self.strip_ticker_suffixes(self.flatten_columns(X_test))

        model = XGBClassifier(
            **study.best_trial.params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print("Classification Report:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        model_path = os.path.join(self.model_dir, f"xgb_{self.ticker}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Save metrics
        result_path = os.path.join(self.model_dir, "model_results.csv")
        log_df = pd.DataFrame([{
            "Ticker": self.ticker,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4)
        }])
        # Overwrite existing entry for this ticker
        if os.path.exists(result_path):
            existing = pd.read_csv(result_path)
            existing = existing[existing["Ticker"] != self.ticker]
            updated = pd.concat([existing, log_df], ignore_index=True)
            updated.to_csv(result_path, index=False)
        else:
            log_df.to_csv(result_path, index=False)


        # Save feature importances
        importances = model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        importance_path = os.path.join(self.model_dir, f"importance_{self.ticker}.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")

        return model
