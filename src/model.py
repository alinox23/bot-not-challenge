from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

class BotDetector:
    def __init__(self,n_estimators:int=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)

    def evaluate(self,x_test,y_test):
        prediction = self.model.predict(x_test)
        print("\n--- Rapport de Classification ---")
        print(classification_report(y_test,prediction))

        print("\n Confusion Matrix (false positive)")
        print(confusion_matrix(y_test,prediction))

    def get_feature_importance(self, feature_names):
        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)

    def save_model(self, filename: str):
        joblib.dump(self.model, filename)
        print(f"Model saved in: {filename} ")

    def load_model(self, filename: str):
        self.model = joblib.load(filename)
        print(f"model loaded from: {filename} ")