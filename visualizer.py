import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def plot_relationships(df:pd.DataFrame):
        sns.set_theme(style="ticks")
        plot=sns.pairplot(df,hue="is_bot",diag_kind="kde",palette="husl")
        plot.fig.suptitle("Humains (0) vs Bots (1)",y=1.02)
        plt.show()

