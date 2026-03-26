import pandas as pd
import numpy as np
import os
from typing import Final

DATA_PATH: Final[str] = "../data"
FILE_NAME: Final[str] = "bot_data.csv"


class BotDataGenerator:

    def __init__(self, sample_size: int = 500):
        self.sample_size = sample_size

    def generate(self) -> pd.DataFrame:
        humans = self._create_profiles(is_bot=False)
        bots = self._create_profiles(is_bot=True)

        df = pd.concat([humans, bots]).sample(frac=1).reset_index(drop=True)
        return df

    def _create_profiles(self, is_bot: bool) -> pd.DataFrame:
        n = self.sample_size
        if not is_bot:
            return pd.DataFrame({
                'followers_count': np.random.randint(100, 5000, n),
                'friends_count': np.random.randint(100, 2000, n),
                'statuses_count': np.random.randint(500, 20000, n),
                'account_age_days': np.random.randint(365, 3000, n),
                'avg_daily_posts': np.random.uniform(0.1, 5, n),
                'is_bot': 0
            })
        else:
            return pd.DataFrame({
                'followers_count': np.random.randint(0, 100, n),
                'friends_count': np.random.randint(1000, 5000, n),
                'statuses_count': np.random.randint(5000, 50000, n),
                'account_age_days': np.random.randint(1, 100, n),
                'avg_daily_posts': np.random.uniform(20, 100, n),
                'is_bot': 1
            })

    def save_to_csv(self, df: pd.DataFrame):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        full_path = os.path.join(DATA_PATH, FILE_NAME)
        df.to_csv(full_path, index=False)


if __name__ == '__main__':
    generator = BotDataGenerator(sample_size=500)
    data = generator.generate()
    generator.save_to_csv(data)