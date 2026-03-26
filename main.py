from src.generate_data import DATA_PATH, FILE_NAME
import os
from src import dataPreprocessor as dp
import visualizer as visualizer
from src.model import BotDetector

def main():
    path = os.path.join(DATA_PATH, FILE_NAME)

    preprocessor=dp.DataPreprocessor(path)
    df=preprocessor.load_data()

    visualizer.Visualizer.plot_relationships(df)

    x_train,x_test,y_train,y_test=preprocessor.prepare_for_training()
    preprocessor.save_scaler("data/scaler.pkl")
    feature_names=df.drop(columns=['is_bot']).columns

    detector=BotDetector()
    detector.train(x_train,y_train)
    detector.evaluate(x_test,y_test)
    detector.save_model("data/bot_detector_v1.pkl")

    print("\n--- Importance des caractéristiques ---")
    print(detector.get_feature_importance(feature_names))


if __name__=="__main__":
    main()