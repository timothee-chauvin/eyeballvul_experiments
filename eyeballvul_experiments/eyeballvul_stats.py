"""Various statistics on the eyeballvul benchmark."""

import json
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from eyeballvul import EyeballvulRevision, get_commits, get_revision, get_vulns

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


def fraction_of_benchmark_covered_by_context_window(
    context_windows: list[int],
):
    """
    Save the following dictionary to Config.paths.results / "fraction_of_benchmark_covered_by_context_window.json":

    {
      context_window: {
          "commits": fraction of the number of commits for which the repo size is smaller than the context window,
          "vuln_occurrences": fraction of the number of vulnerability occurrences (i.e. vulnerability x revision) for which the repo size is smaller than the context window,
          "vulns": fraction of the number of vulnerabilities which have at least one commit for which the repo size is smaller than the context window,
          "size": fraction of the benchmark size (sum of the sizes of each commit) that is smaller than the context window,
        }
    }
    for each context window (in tokens) provided as input.

    A vulnerability occurrence is a tuple (vulnerability, commit) where commit is among `vulnerability.commits`. It is used because the same vulnerability can be associated to multiple commits, and it can be the case that not all of these commits fall on the same side of the context window threshold.

    This function assumes that 1 token = 4 bytes.
    """
    context_windows_bytes = [context_window * 4 for context_window in context_windows]
    commits = get_commits()
    commit_to_size = {
        commit: cast(EyeballvulRevision, get_revision(commit)).size for commit in commits
    }

    vulns = get_vulns()
    vuln_occurrence_sizes = [commit_to_size[commit] for vuln in vulns for commit in vuln.commits]
    vuln_smallest_commit_sizes = [
        min(commit_to_size[commit] for commit in vuln.commits) for vuln in vulns
    ]
    commit_sizes = list(commit_to_size.values())
    benchmark_size = sum(commit_sizes)
    result = {}
    for context_window, context_window_bytes in zip(context_windows, context_windows_bytes):
        commits_below = sum(size < context_window_bytes for size in commit_sizes)
        vuln_occurrences_below = sum(size < context_window_bytes for size in vuln_occurrence_sizes)
        vuln_smallest_commit_below = sum(
            size < context_window_bytes for size in vuln_smallest_commit_sizes
        )
        size_below = sum([size for size in commit_sizes if size < context_window_bytes])
        result[context_window] = {
            "commits": commits_below / len(commits),
            "vuln_occurrences": vuln_occurrences_below / len(vuln_occurrence_sizes),
            "vulns": vuln_smallest_commit_below / len(vulns),
            "size": size_below / benchmark_size,
        }
    with open(
        Config.paths.results / "fraction_of_benchmark_covered_by_context_window.json", "w"
    ) as f:
        json.dump(result, f, indent=2)
        f.write("\n")
