"""Various statistics on the eyeballvul benchmark."""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from eyeballvul import EyeballvulRevision, get_commits, get_revision


def repo_size_histogram():
    commits = get_commits()
    sizes = sorted([cast(EyeballvulRevision, get_revision(commit)).size for commit in commits])
    df = pd.DataFrame({"sizes": sizes})
    df["cumulative_sum"] = df["sizes"].cumsum()
    sns.set_theme()

    ax = sns.displot(
        data=df["sizes"],
        kde=True,
        log_scale=True,
        legend=False,
        stat="density",
    )
    ax.set(
        xlabel="size of the repository in bytes according to linguist", ylabel="number of commits"
    )
    plt.show()

    ax2 = sns.lineplot(data=df, x="sizes", y="cumulative_sum")
    ax2.set_xscale("log")
    plt.show()
