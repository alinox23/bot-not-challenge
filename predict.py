import pandas as pd
from src.dataPreprocessor import DataPreprocessor
from src.model import BotDetector


def run_prediction():
    preprocessor = DataPreprocessor()
    detector = BotDetector()

    try:
        preprocessor.load_scaler("data/scaler.pkl")
        detector.load_model("data/bot_detector_v1.pkl")
    except Exception as e:
        print(f"Error loading : {e}")
        return

    new_account = {
        'followers_count': [10],
        'friends_count': [4000],
        'statuses_count': [100],
        'account_age_days': [5],
        'avg_daily_posts': [2]
    }
    df_new = pd.DataFrame(new_account)

    data_scaled = preprocessor.scaler.transform(df_new)

    prediction = detector.model.predict(data_scaled)

    result = "BOT" if prediction[0] == 1 else "HUMAIN"
    print(f"\nVerdict : {result}")


if __name__ == "__main__":
    run_prediction()