"""Various statistics on the eyeballvul benchmark."""

from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from eyeballvul import EyeballvulRevision, get_commits, get_revision

from eyeballvul_experiments.config.config_loader import Config


def repo_size_histogram():
    commits = get_commits()
    sizes = sorted([cast(EyeballvulRevision, get_revision(commit)).size for commit in commits])
    df = pd.DataFrame({"sizes": sizes})
    df["log_sizes"] = np.log10(df["sizes"])
    df["cumulative_sum"] = df["sizes"].cumsum()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["log_sizes"],
            name="Number of commits",
            nbinsx=100,
            opacity=0.75,
            marker=dict(color="rgb(138, 146, 251)", line=dict(color="white", width=0.5)),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["log_sizes"],
            y=df["cumulative_sum"],
            mode="lines",
            name="Cumulative sum of sizes",
            yaxis="y2",
            marker=dict(color="rgb(255, 0, 0)"),
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Size of the repository in bytes according to linguist",
            tickmode="array",
            tickvals=np.log10(
                [10**i for i in range(int(df["log_sizes"].min()), int(df["log_sizes"].max()) + 1)]
            ),
            ticktext=[
                f"{10**i:.0e}"
                for i in range(int(df["log_sizes"].min()), int(df["log_sizes"].max()) + 1)
            ],
            type="linear",
        ),
        yaxis=dict(
            title="Number of commits",
            gridcolor="rgba(138, 146, 251, 0.15)",
            griddash="dash",
        ),
        yaxis2=dict(
            title="Cumulative sum of sizes",
            overlaying="y",
            side="right",
            range=[0, df["cumulative_sum"].max()],
            gridcolor="rgba(255, 0, 0, 0.15)",
            griddash="dash",
        ),
        template="plotly_white",
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0)", bordercolor="rgba(255, 255, 255, 0)"
        ),
    )

    fig.write_image(Config.paths.plots / "repo_size_histogram.png")
