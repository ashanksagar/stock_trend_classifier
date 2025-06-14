import streamlit as st
from datetime import date
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.train_model_optuna import ModelTrainer
from src.predict import StockPredictor
from src.backtester import Backtester
from datetime import datetime
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Price Trend Classifier", layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Stock Price Trend Classifier (S&P 500 Top 50)")
st.markdown("""
 By: Ashank Sagar | Updated June 2025
<div style='margin-bottom: 30px;'></div>
""", unsafe_allow_html=True)
st.markdown("Predict whether a stock will go **Up or Down** tomorrow based on historical trends. This app uses machine")
st.markdown("learning (XGBoost classifier) trained on technical indicators like RSI, MACD, moving averages, and more.")

model_dir = "models"
trained_models = [f for f in os.listdir(model_dir) if f.startswith("xgb_") and f.endswith(".pkl")]
available_tickers = sorted([f.replace("xgb_", "").replace(".pkl", "") for f in trained_models])

if not available_tickers:
    st.error("No trained models found. Please train models first.")
    st.stop()

left_col, right_col = st.columns([3, 2])

with left_col:
    ticker = st.selectbox("Select a Stock Ticker:", available_tickers)
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=date.today())

    if st.button("Run Prediction"):
        try:
            with st.spinner("📱 Fetching and processing data..."):
                loader = StockDataLoader(ticker, str(start_date), str(end_date))
                if end_date > datetime.today().date():
                    st.warning("End date cannot be in the future. Resetting to today's date.")
                    end_date = datetime.today().date()
                df = loader.download(save_csv=False)

                if df is None or df.empty:
                    st.error(f"No data found for {ticker}. Try a different date range.")
                    st.stop()

                fe = FeatureEngineer(df)
                featured_df = fe.get_featured_data()

                if featured_df.empty or len(featured_df) < 10:
                    st.warning("Not enough data after feature engineering. Select a longer date range.")
                    st.stop()

                featured_df["Date"] = pd.to_datetime(featured_df["Date"])
                featured_df = featured_df.sort_values("Date")

                featured_df["MA7"] = featured_df["Close"].rolling(window=7, min_periods=1).mean()
                featured_df["MA14"] = featured_df["Close"].rolling(window=14, min_periods=1).mean()

                drop_cols = [col for col in ["Date", "Ticker", "Target", "MA7", "MA14", "Prediction"] if col in featured_df.columns]
                predict_df = featured_df.drop(columns=drop_cols)

                if isinstance(predict_df.columns, pd.MultiIndex):
                    predict_df.columns = predict_df.columns.get_level_values(0)
                predict_df.columns = [col if isinstance(col, str) else col[0] for col in predict_df.columns]
                predict_df.columns = [col.replace(f"_{ticker}", "") for col in predict_df.columns]

                predictions, probs = StockPredictor(ticker).predict(predict_df)
                pred_df = featured_df.iloc[-len(predictions):].copy()
                pred_df["Prediction"] = predictions

                recent_pred = predictions[-1]
                recent_conf = max(probs[-1]) * 100
                direction = "⬆️ Up" if recent_pred == 1 else "🔻 Down"
                confidence = f"{recent_conf:.2f}%"

                st.success(f"Prediction for next day: **{direction}**")
                st.info(f"Model confidence: {confidence}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=featured_df["Date"], y=featured_df["Close"], mode="lines", name="Close Price"))
                fig.add_trace(go.Scatter(x=featured_df["Date"], y=featured_df["MA7"], mode="lines", name="7-day MA"))
                fig.add_trace(go.Scatter(x=featured_df["Date"], y=featured_df["MA14"], mode="lines", name="14-day MA"))
                fig.add_trace(go.Scatter(x=pred_df.loc[pred_df["Prediction"] == 1, "Date"],
                                         y=pred_df.loc[pred_df["Prediction"] == 1, "Close"],
                                         mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"), name="Model: Up",
                                         visible="legendonly"))
                fig.add_trace(go.Scatter(x=pred_df.loc[pred_df["Prediction"] == 0, "Date"],
                                         y=pred_df.loc[pred_df["Prediction"] == 0, "Close"],
                                         mode="markers", marker=dict(color="red", size=8, symbol="triangle-down"), name="Model: Down",
                                         visible="legendonly"))
                fig.update_layout(title=f"{ticker} Price + Model Predictions", xaxis_title="Date", yaxis_title="Price (USD)", height=500, legend_title="Legend")
                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance chart directly under main chart
                importance_path = os.path.join(model_dir, f"importance_{ticker}.csv")
                if os.path.exists(importance_path):
                    df_imp = pd.read_csv(importance_path)
                    bar_fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                                     title=f"Feature Importance for {ticker}", height=400)
                    st.plotly_chart(bar_fig, use_container_width=True, key="feature_importance_chart")
                else:
                    st.warning("Feature importance not found. Retrain model to generate.")

        except Exception as e:
            st.error(f"Something went wrong:\n\n{e}")

with right_col:
    st.markdown("<h3 style='text-align: center;'>Model Leaderboard (Accuracy)</h3>", unsafe_allow_html=True)

    results_path = os.path.join(model_dir, "model_results.csv")
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        df_results_sorted = (
            df_results.drop_duplicates("Ticker", keep="last")
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df_results_sorted, use_container_width=True, height=400)

        if 'featured_df' in locals():
            st.markdown("<h3 style='text-align: center;'>Backtest Results</h3>", unsafe_allow_html=True)
            backtester = Backtester(StockPredictor(ticker).model, featured_df)
            metrics = backtester.run()

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
            col2.metric("Cumulative Return", f"{metrics['Cumulative Return']:.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

            col4, col5 = st.columns(2)
            col4.metric("Baseline Return", f"{metrics['Baseline Return']:.2f}%")
            col5.metric("Improvement Over Baseline", f"{metrics['Improvement Over Baseline (%)']:.2f}%")

            # Confidence chart under backtest in right column
            if 'probs' in locals():
                last_50 = probs[-50:]
                conf_df = pd.DataFrame({
                    "Confidence": [max(p) for p in last_50],
                    "Prediction": [int(p[1] > p[0]) for p in last_50]
                })
                conf_fig = px.line(conf_df, y="Confidence", color="Prediction",
                                   title="Confidence on Last 50 Predictions",
                                   labels={"Prediction": "Predicted Class"}, height=400)
                st.plotly_chart(conf_fig, use_container_width=True, key="confidence_chart")

    else:
        st.warning("No model results found yet. Train models to generate leaderboard.")
