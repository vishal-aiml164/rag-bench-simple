import json, os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    for metric in [c for c in df.columns if c not in ('exp','k')]:
        plt.figure()
        sub = df.pivot(index='k', columns='exp', values=metric)
        sub.plot()
        plt.title(metric)
        plt.xlabel('k')
        plt.ylabel(metric)
        out = os.path.splitext(csv_path)[0] + f"_{metric}.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
