import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib

class DataPreprocessor:
    def __init__(self, file_path: str=None):
        self.file_path=file_path
        self.scaler = StandardScaler()
        self.df=None

    def load_data(self)->pd.DataFrame:
        self.df=pd.read_csv(self.file_path)
        return self.df

    def prepare_for_training(self,target_column:str='is_bot')->Tuple:
        x=self.df.drop(columns=[target_column])
        y=self.df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        return x_train_scaled, x_test_scaled, y_train,y_test

    def save_scaler(self, filename: str):
        joblib.dump(self.scaler, filename)
        print(f"Scaler saved in: {filename}")

    def load_scaler(self, filename: str):
        self.scaler = joblib.load(filename)
        print(f"Scaler loaded from: {filename} ")

