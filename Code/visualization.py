# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from topologylayer.nn import AlphaLayer
from topologylayer.util.process import remove_zero_bars

sns.set_theme(context="talk", style="ticks",
              rc={"axes.spines.right": False, "axes.spines.top": False},
              )


def plot_persistence(output):
    output = torch.tensor(output).type(torch.float)
    dgminfo = AlphaLayer(maxdim=1)(output)

    dgms, _ = dgminfo
    df = pd.DataFrame()

    for dim in range(len(dgms)):
        dgm = dgms[dim]
        dgm = dgm.detach().numpy()
        dgm = remove_zero_bars(dgm)

        inf_ids = dgm[:, 1] == np.inf
        dgm[inf_ids, 1] = -1

        df = df.append(
            pd.DataFrame(data={"birth": dgm[:, 0],
                               "death": dgm[:, 1],
                               "dim": np.repeat(dim, dgm.shape[0]),
                               "inf": inf_ids}))

    # set infinite values to max in diagram
    df.loc[df.inf, "death"] = 1.3 * df["death"].max()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axline((0, 0), slope=1, linewidth=1, color='black', zorder=0)
    g = sns.scatterplot(data=df, x="birth", y="death", s=100, ax=ax,
                        hue="dim", style="inf", markers=["o", "^"],
                        linewidth=0, alpha=0.9
                        )
    ax.set_aspect('equal')
    ax.set_xmargin(0.2)
    xlim = (None, max(df["death"].max(), 1.5*df["birth"].max()))
    ax.set_xlim(xlim)
    sns.despine(offset=10, trim=True)
    # plt.show()
    return fig


def plot_paper(Y, colors=None, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = plt.gcf()

    if colors is not None:
        g = sns.scatterplot(x=Y[:, 0],
                            y=Y[:, 1],
                            s=130, ax=ax, hue=colors,
                            palette="husl", alpha=0.7,
                            linewidth=0)
        ax.get_legend().remove()
    else:
        g = sns.scatterplot(x=Y[:, 0],
                            y=Y[:, 1],
                            s=130,
                            ax=ax,
                            alpha=0.7,
                            linewidth=0)
    plt.axis('image')
    ax.set_title(title)
    sns.despine(offset=5, trim=True, ax=ax)
    ticks = ax.get_xticks() if len(ax.get_xticks()) <= len(
        ax.get_yticks()) else ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    sns.despine(offset=5, trim=True, ax=ax)
    plt.tight_layout()
    return fig
