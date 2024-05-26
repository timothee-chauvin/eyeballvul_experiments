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
    fig, ax = plt.subplots()

    sns.histplot(data=df["sizes"], kde=True, log_scale=True, ax=ax)

    ax2 = ax.twinx()

    sns.lineplot(data=df, x="sizes", y="cumulative_sum", ax=ax2)

    ax.set_xscale("log")

    # Left y axis
    ax.set_xlabel("size of the repository in bytes according to linguist")
    ax.set_ylabel("number of commits")

    # Right y axis
    ax2.set_ylabel("cumulative sum of sizes")

    # Remove the horizontal lines for the left y axis
    ax.grid(which="both", axis="y", linestyle="")

    plt.show()
