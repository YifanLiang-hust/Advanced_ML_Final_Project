import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F


def plot_distribution(output_dir, id_scores, ood_scores, out_dataset, score=None):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    id_scores = id_scores.flatten()
    ood_scores = ood_scores.flatten()
    data = {
        "ID": [id_score for id_score in id_scores],
        "OOD": [ood_score for ood_score in ood_scores]
    }
    # data = pd.DataFrame({
    #     "score": [-1 * score for score in id_scores.flatten()] + [-1 * score for score in ood_scores.flatten()],
    #     "type": ["ID"] * len(id_scores) + ["OOD"] * len(ood_scores)
    # })
    sns.displot(data, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
    if score is not None:
        # if not os.path.exists(os.path.join(output_dir,f"{out_dataset}")):
        #     os.makedirs(os.path.join(output_dir,f"{out_dataset}"))
        # plt.savefig(os.path.join(output_dir,f"{out_dataset}",f"{out_dataset}_{t}_{score}.png"), bbox_inches='tight')
        output_path = os.path.join(output_dir, f"{out_dataset}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, f"{out_dataset}_{score}.png")
        print(f"Saving figure to: {file_path}")
        plt.savefig(file_path, bbox_inches='tight')
    else:
        output_path = os.path.join(output_dir, f"{out_dataset}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, f"{out_dataset}.png")
        print(f"Saving figure to: {file_path}")
        plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
