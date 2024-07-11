import json
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from confidenceinterval import f1_score, precision_score, recall_score
from cvss import CVSS3
from eyeballvul import EyeballvulItem, EyeballvulScore, get_vulns
from plotly.subplots import make_subplots
from tqdm import tqdm

from eyeballvul_experiments.attempt import Attempt
from eyeballvul_experiments.config.config_loader import Config

# https://github.com/plotly/plotly.py/issues/3469
pio.kaleido.scope.mathjax = None


def get_scores_with_hash(
    attempt: Attempt, instruction_template_hash: str, scoring_model: str
) -> list[EyeballvulScore]:
    return [
        score
        for score in attempt.scores
        if score.instruction_template_hash == instruction_template_hash
        and score.scoring_model == scoring_model
    ]


def plot_overall_performance(
    instruction_template_hash: str,
    scoring_model: str,
    model_order: list[str],
    model_names: dict[str, str],
    color_map: dict[str, str],
):
    results: dict[str, dict] = {}
    reconstructed_classifications: dict[str, dict] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        results.setdefault(
            attempt.model,
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "precision_ci_low": 0.0,
                "precision_ci_upp": 0.0,
                "recall_ci_low": 0.0,
                "recall_ci_upp": 0.0,
                "f1_ci_low": 0.0,
                "f1_ci_upp": 0.0,
            },
        )
        # Only keep the first score.
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        score = scores[0]
        results[attempt.model]["tp"] += score.stats.tp
        results[attempt.model]["fp"] += score.stats.fp
        results[attempt.model]["fn"] += score.stats.fn

    for model in results:
        reconstructed_classifications.setdefault(model, {"y_true": [], "y_pred": []})

        # tp
        reconstructed_classifications[model]["y_true"].extend([1] * results[model]["tp"])
        reconstructed_classifications[model]["y_pred"].extend([1] * results[model]["tp"])

        # fp
        reconstructed_classifications[model]["y_true"].extend([0] * results[model]["fp"])
        reconstructed_classifications[model]["y_pred"].extend([1] * results[model]["fp"])

        # fn
        reconstructed_classifications[model]["y_true"].extend([1] * results[model]["fn"])
        reconstructed_classifications[model]["y_pred"].extend([0] * results[model]["fn"])

    for model in results:
        precision, precision_ci = precision_score(
            reconstructed_classifications[model]["y_true"],
            reconstructed_classifications[model]["y_pred"],
            confidence_level=0.95,
            method="wilson",
            average="binary",
        )
        recall, recall_ci = recall_score(
            reconstructed_classifications[model]["y_true"],
            reconstructed_classifications[model]["y_pred"],
            confidence_level=0.95,
            method="wilson",
            average="binary",
        )
        f1, f1_ci = f1_score(
            reconstructed_classifications[model]["y_true"],
            reconstructed_classifications[model]["y_pred"],
            confidence_level=0.95,
            method="wilson",
            average="binary",
        )
        # Sanity checks: verify that confidenceinterval computes the same values as we do
        classified_positive = results[model]["tp"] + results[model]["fp"]
        all_positive = results[model]["tp"] + results[model]["fn"]
        our_precision = results[model]["tp"] / classified_positive
        our_recall = results[model]["tp"] / (all_positive)
        our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

        if abs(precision - our_precision) > 1e-6:
            raise ValueError(f"Precision mismatch: {precision} != {results[model]['precision']}")
        if abs(recall - our_recall) > 1e-6:
            raise ValueError(f"Recall mismatch: {recall} != {results[model]['recall']}")
        if abs(f1 - our_f1) > 1e-6:
            raise ValueError(f"F1 mismatch: {f1} != {results[model]['f1']}")

        results[model]["precision"] = precision
        results[model]["precision_ci_low"] = precision_ci[0]
        results[model]["precision_ci_upp"] = precision_ci[1]
        results[model]["recall"] = recall
        results[model]["recall_ci_low"] = recall_ci[0]
        results[model]["recall_ci_upp"] = recall_ci[1]
        results[model]["f1"] = f1
        results[model]["f1_ci_low"] = f1_ci[0]
        results[model]["f1_ci_upp"] = f1_ci[1]

    results = {k: results[k] for k in sorted(results)}
    with open(Config.paths.results / "overall_performance.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    df = pd.DataFrame(
        {
            "model": list(results.keys()),
            "model_name": [model_names[model] for model in results],
            "precision": [results[model]["precision"] for model in results],
            "recall": [results[model]["recall"] for model in results],
            "f1": [results[model]["f1"] for model in results],
            "precision_ci_low": [results[model]["precision_ci_low"] for model in results],
            "precision_ci_upp": [results[model]["precision_ci_upp"] for model in results],
            "recall_ci_low": [results[model]["recall_ci_low"] for model in results],
            "recall_ci_upp": [results[model]["recall_ci_upp"] for model in results],
            "f1_ci_low": [results[model]["f1_ci_low"] for model in results],
            "f1_ci_upp": [results[model]["f1_ci_upp"] for model in results],
        }
    )
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")

    # First, plot a regular bar plot.
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df["precision"],
            name="Precision",
            marker_color="rgb(69, 126, 172)",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["precision_ci_upp"] - df["precision"],
                "arrayminus": df["precision"] - df["precision_ci_low"],
            },
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df["recall"],
            name="Recall",
            marker_color="rgb(194, 175, 240)",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["recall_ci_upp"] - df["recall"],
                "arrayminus": df["recall"] - df["recall_ci_low"],
            },
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df["f1"],
            name="F1 Score",
            marker_color="rgb(138, 146, 251)",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["f1_ci_upp"] - df["f1"],
                "arrayminus": df["f1"] - df["f1_ci_low"],
            },
        )
    )

    # Add textual values to the bars.
    for i in range(len(df)):
        for j, metric in enumerate(["precision", "recall", "f1"]):
            fig.add_annotation(
                x=df["model_name"].iloc[i],
                y=0,
                text=f"{df[metric].iloc[i]:.1%}",
                showarrow=False,
                font=dict(color="white", size=18),
                xanchor="center",
                yanchor="bottom",
                xshift=(-1 + j) * 25,
                textangle=90,
            )

    fig.update_layout(
        yaxis=dict(
            tickformat=",.0%",
        ),
        template="plotly_white",
        barmode="group",
        font_size=20,
        margin=dict(t=5, l=5, r=5, b=5),
        legend=dict(orientation="v", yanchor="bottom", y=0.7, xanchor="center", x=0.7),
        width=800,
        height=800,
    )
    fig.write_image(Config.paths.plots / "overall_performance.pdf")

    # Then plot a Pareto efficiency plot.
    traces = []
    for model in df["model"]:
        trace = go.Scatter(
            x=df.loc[df["model"] == model, "precision"],
            y=df.loc[df["model"] == model, "recall"],
            mode="markers",
            name=model_names[model],
            marker=dict(color=color_map[model], size=8),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df.loc[df["model"] == model, "precision_ci_upp"]
                - df.loc[df["model"] == model, "precision"],
                arrayminus=df.loc[df["model"] == model, "precision"]
                - df.loc[df["model"] == model, "precision_ci_low"],
                width=0,
                thickness=1,
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=df.loc[df["model"] == model, "recall_ci_upp"]
                - df.loc[df["model"] == model, "recall"],
                arrayminus=df.loc[df["model"] == model, "recall"]
                - df.loc[df["model"] == model, "recall_ci_low"],
                width=0,
                thickness=1,
            ),
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickformat=",.0%",
            title="Precision",
        ),
        yaxis=dict(
            tickformat=",.0%",
            title="Recall",
        ),
        font_size=20,
        margin=dict(t=5, l=5, r=5, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        width=800,
        height=800,
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    fig.write_image(Config.paths.plots / "pareto_efficiency.pdf")


def average_number_of_chunks_by_model(instruction_template_hash: str, scoring_model: str):
    """Among attempts that have been scored, count the average number of chunks given to models (the
    number of times they've been run)."""
    total_chunks_by_model: dict[str, list[int]] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        total_chunks_by_model.setdefault(attempt.model, [0, 0])
        if get_scores_with_hash(attempt, instruction_template_hash, scoring_model):
            total_chunks_by_model[attempt.model][0] += len(attempt.chunk_hashes)
            total_chunks_by_model[attempt.model][1] += 1
    results = {model: values[0] / values[1] for model, values in total_chunks_by_model.items()}
    results = {k: results[k] for k in sorted(results)}
    with open(Config.paths.results / "average_chunks_by_model.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def plot_performance_before_after_training_cutoff(
    instruction_template_hash: str,
    scoring_model: str,
    model_order: list[str],
    model_names: dict[str, str],
    color_map: dict[str, str],
    cutoff_dates: dict[str, str],
):
    results: dict[str, dict] = {}
    reconstructed_classifications: dict[str, dict] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        results.setdefault(attempt.model, {})
        for key in "before", "after":
            results[attempt.model].setdefault(
                key,
                {
                    "tp": 0,
                    "fp": 0.0,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "precision_ci_low": 0.0,
                    "precision_ci_upp": 0.0,
                    "recall_ci_low": 0.0,
                    "recall_ci_upp": 0.0,
                    "f1_ci_low": 0.0,
                    "f1_ci_upp": 0.0,
                },
            )
        # Only keep the first score.
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        score = scores[0]
        stats = score.stats_with_cutoff(
            cutoff_date=datetime.fromisoformat(cutoff_dates[attempt.model])
        )
        results[attempt.model]["before"]["fp"] += (
            stats.fp
            * (stats.before.tp + stats.before.fn)
            / (stats.before.tp + stats.before.fn + stats.after.tp + stats.after.fn)
        )
        results[attempt.model]["before"]["tp"] += stats.before.tp
        results[attempt.model]["before"]["fn"] += stats.before.fn
        results[attempt.model]["after"]["fp"] += (
            stats.fp
            * (stats.after.tp + stats.after.fn)
            / (stats.before.tp + stats.before.fn + stats.after.tp + stats.after.fn)
        )
        results[attempt.model]["after"]["tp"] += stats.after.tp
        results[attempt.model]["after"]["fn"] += stats.after.fn

    for model in results:
        reconstructed_classifications.setdefault(
            model,
            {
                "before": {"y_true": [], "y_pred": []},
                "after": {"y_true": [], "y_pred": []},
            },
        )

        for key in ["before", "after"]:
            results[model][key]["fp"] = round(results[model][key]["fp"])
            # tp
            reconstructed_classifications[model][key]["y_true"].extend(
                [1] * results[model][key]["tp"]
            )
            reconstructed_classifications[model][key]["y_pred"].extend(
                [1] * results[model][key]["tp"]
            )

            # fp
            reconstructed_classifications[model][key]["y_true"].extend(
                [0] * results[model][key]["fp"]
            )
            reconstructed_classifications[model][key]["y_pred"].extend(
                [1] * results[model][key]["fp"]
            )

            # fn
            reconstructed_classifications[model][key]["y_true"].extend(
                [1] * results[model][key]["fn"]
            )
            reconstructed_classifications[model][key]["y_pred"].extend(
                [0] * results[model][key]["fn"]
            )

    for model in results:
        for key in ["before", "after"]:
            precision, precision_ci = precision_score(
                reconstructed_classifications[model][key]["y_true"],
                reconstructed_classifications[model][key]["y_pred"],
                confidence_level=0.95,
                method="wilson",
                average="binary",
            )
            recall, recall_ci = recall_score(
                reconstructed_classifications[model][key]["y_true"],
                reconstructed_classifications[model][key]["y_pred"],
                confidence_level=0.95,
                method="wilson",
                average="binary",
            )
            f1, f1_ci = f1_score(
                reconstructed_classifications[model][key]["y_true"],
                reconstructed_classifications[model][key]["y_pred"],
                confidence_level=0.95,
                method="wilson",
                average="binary",
            )
            # Sanity checks: verify that confidenceinterval computes the same values as we do
            classified_positive = results[model][key]["tp"] + results[model][key]["fp"]
            all_positive = results[model][key]["tp"] + results[model][key]["fn"]
            our_precision = results[model][key]["tp"] / classified_positive
            our_recall = results[model][key]["tp"] / (all_positive)
            our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

            if abs(precision - our_precision) > 1e-6:
                raise ValueError(
                    f"Precision mismatch: {precision} != {results[model]['precision']}"
                )
            if abs(recall - our_recall) > 1e-6:
                raise ValueError(f"Recall mismatch: {recall} != {results[model]['recall']}")
            if abs(f1 - our_f1) > 1e-6:
                raise ValueError(f"F1 mismatch: {f1} != {results[model]['f1']}")

            results[model][key]["precision"] = precision
            results[model][key]["precision_ci_low"] = precision_ci[0]
            results[model][key]["precision_ci_upp"] = precision_ci[1]
            results[model][key]["recall"] = recall
            results[model][key]["recall_ci_low"] = recall_ci[0]
            results[model][key]["recall_ci_upp"] = recall_ci[1]
            results[model][key]["f1"] = f1
            results[model][key]["f1_ci_low"] = f1_ci[0]
            results[model][key]["f1_ci_upp"] = f1_ci[1]

    results = {k: results[k] for k in sorted(results)}
    with open(Config.paths.results / "performance_before_after_training_cutoff.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    df_before = pd.DataFrame(
        {
            "model": list(results.keys()),
            "model_name": [model_names[model] for model in results],
            "precision": [results[model]["before"]["precision"] for model in results],
            "recall": [results[model]["before"]["recall"] for model in results],
            "precision_ci_low": [results[model]["before"]["precision_ci_low"] for model in results],
            "precision_ci_upp": [results[model]["before"]["precision_ci_upp"] for model in results],
            "recall_ci_low": [results[model]["before"]["recall_ci_low"] for model in results],
            "recall_ci_upp": [results[model]["before"]["recall_ci_upp"] for model in results],
        }
    )
    df_before["model"] = pd.Categorical(df_before["model"], categories=model_order, ordered=True)
    df_before = df_before.sort_values("model")

    df_after = pd.DataFrame(
        {
            "model": list(results.keys()),
            "model_name": [model_names[model] for model in results],
            "precision": [results[model]["after"]["precision"] for model in results],
            "recall": [results[model]["after"]["recall"] for model in results],
            "precision_ci_low": [results[model]["after"]["precision_ci_low"] for model in results],
            "precision_ci_upp": [results[model]["after"]["precision_ci_upp"] for model in results],
            "recall_ci_low": [results[model]["after"]["recall_ci_low"] for model in results],
            "recall_ci_upp": [results[model]["after"]["recall_ci_upp"] for model in results],
        }
    )
    df_after["model"] = pd.Categorical(df_after["model"], categories=model_order, ordered=True)
    df_after = df_after.sort_values("model")

    traces = []
    for model in df_before["model"]:
        trace_before = go.Scatter(
            x=df_before.loc[df_before["model"] == model, "precision"],
            y=df_before.loc[df_before["model"] == model, "recall"],
            mode="markers",
            name=f"{model_names[model]} (before)",
            marker=dict(color=color_map[model], symbol="circle-open", size=8),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df_before.loc[df_before["model"] == model, "precision_ci_upp"]
                - df_before.loc[df_before["model"] == model, "precision"],
                arrayminus=df_before.loc[df_before["model"] == model, "precision"]
                - df_before.loc[df_before["model"] == model, "precision_ci_low"],
                width=0,
                thickness=1,
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_before.loc[df_before["model"] == model, "recall_ci_upp"]
                - df_before.loc[df_before["model"] == model, "recall"],
                arrayminus=df_before.loc[df_before["model"] == model, "recall"]
                - df_before.loc[df_before["model"] == model, "recall_ci_low"],
                width=0,
                thickness=1,
            ),
        )
        trace_after = go.Scatter(
            x=df_after.loc[df_after["model"] == model, "precision"],
            y=df_after.loc[df_after["model"] == model, "recall"],
            mode="markers",
            name=f"{model_names[model]} (after)",
            marker=dict(color=color_map[model], symbol="circle", size=8),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df_after.loc[df_after["model"] == model, "precision_ci_upp"]
                - df_after.loc[df_after["model"] == model, "precision"],
                arrayminus=df_after.loc[df_after["model"] == model, "precision"]
                - df_after.loc[df_after["model"] == model, "precision_ci_low"],
                width=0,
                thickness=1,
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_after.loc[df_after["model"] == model, "recall_ci_upp"]
                - df_after.loc[df_after["model"] == model, "recall"],
                arrayminus=df_after.loc[df_after["model"] == model, "recall"]
                - df_after.loc[df_after["model"] == model, "recall_ci_low"],
                width=0,
                thickness=1,
            ),
        )
        traces.extend([trace_before, trace_after])

    fig = go.Figure(data=traces)
    for model in df_before["model"]:
        x0 = df_after.loc[df_after["model"] == model, "precision"].values[0]
        y0 = df_after.loc[df_after["model"] == model, "recall"].values[0]
        x1 = df_before.loc[df_before["model"] == model, "precision"].values[0]
        y1 = df_before.loc[df_before["model"] == model, "recall"].values[0]

        fig.add_annotation(
            x=x0,
            y=y0,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=1,
            arrowcolor=color_map[model],
        )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickformat=",.0%",
            title="Precision",
        ),
        yaxis=dict(
            tickformat=",.0%",
            title="Recall",
        ),
        font_size=14,
        legend=dict(title="Model (Period)"),
        margin=dict(t=5, l=5, r=5, b=5),
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    fig.write_image(Config.paths.plots / "pareto_efficiency_before_after_training_cutoff.pdf")


def plot_cwes_found(instruction_template_hash: str, scoring_model: str, top_n: int):
    cwe_occurrences: dict[str, float] = {}
    cwe_descriptions: dict[str, str] = {
        "CWE-79": "XSS",
        "CWE-78": "OS Command Injection",
        "CWE-22": "Path Traversal",
        "CWE-20": "Improper Input Validation",
        "CWE-94": "Code Injection",
        "CWE-798": "Use of Hard-coded Credentials",
        "CWE-77": "Command Injection",
        "CWE-502": "Deserialization of Untrusted Data",
        "CWE-611": "XXE (XML External Entity)",
        "CWE-1321": "Prototype Pollution",
        "CWE-89": "SQL Injection",
    }
    cwe_top_25_ranks = {
        "CWE-79": "2",
        "CWE-78": "5",
        "CWE-22": "8",
        "CWE-20": "6",
        "CWE-94": "23",
        "CWE-798": "18",
        "CWE-502": "15",
        "CWE-1321": "> 25",
        "CWE-77": "16",
        "CWE-611": "> 25",
    }
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        score = scores[0]
        selected_leads = [lead for lead in attempt.leads if lead.classification == "very promising"]
        for lead_index in score.mapping.keys():
            cwe = selected_leads[lead_index].cwe
            if cwe is None:
                continue
            cwe_occurrences[cwe] = cwe_occurrences.get(cwe, 0) + 1
    total = sum(cwe_occurrences.values())
    for cwe in cwe_occurrences:
        cwe_occurrences[cwe] = cwe_occurrences[cwe] / total
    fig = go.Figure()
    df = pd.DataFrame(
        {
            "cwe": list(cwe_occurrences.keys()),
            "occurrences": list(cwe_occurrences.values()),
            "description": [cwe_descriptions.get(cwe, "") for cwe in cwe_occurrences.keys()],
        }
    )
    df = df.sort_values("occurrences", ascending=False)
    df = df.head(top_n)
    df = df.sort_values("occurrences", ascending=True)
    fig.add_trace(
        go.Bar(
            y=df["cwe"],
            x=df["occurrences"],
            text=df["description"],
            textposition="outside",
            orientation="h",
            marker_color="rgb(138, 146, 251)",
        )
    )
    # Add vertical dotted line before the top 25 ranks
    fig.add_vline(
        x=0.22,
        line_dash="dot",
        line_color="gray",
        line_width=1,
    )
    fig.add_annotation(
        x=0.93,
        y=1.05,
        xref="paper",
        yref="paper",
        text="Rank in\nCWE Top 25",
        showarrow=False,
        font=dict(size=14),
        xanchor="left",
        align="left",
    )
    # Add textual values to the bars.
    for _, row in df.iterrows():
        fig.add_annotation(
            x=0,
            y=row["cwe"],
            text=f"{row['occurrences']:.1%}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=14, color="white"),
            align="left",
        )
        # Add rank in top 25 on the right of the figure
        fig.add_annotation(
            x=0.23,
            y=row["cwe"],
            text=cwe_top_25_ranks[row["cwe"]],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=14),
            align="left",
        )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, l=5, r=100, b=5),
        xaxis={"title": "Frequency among true positives", "tickformat": ",.0%"},
        font_size=16,
    )
    fig.write_image(Config.paths.plots / "cwes_found.pdf")


def get_severity_category(severity):
    if severity == 0:
        return "None"
    elif severity < 4:
        return "Low"
    elif severity < 7:
        return "Medium"
    elif severity < 9:
        return "High"
    else:
        return "Critical"


def get_severity_score(vuln: EyeballvulItem) -> float | None:
    v3_severities = [severity for severity in vuln.severity or [] if severity["type"] == "CVSS_V3"]
    if v3_severities:
        return CVSS3(v3_severities[0]["score"]).scores()[0]
    return None


def get_severity_stats(vulns: list[EyeballvulItem]) -> dict[str, Any]:
    cve_severity_counts: dict[float, int] = {}
    no_severity = 0
    for vuln in tqdm(vulns):
        severity_score = get_severity_score(vuln)
        if severity_score:
            cve_severity_counts[severity_score] = cve_severity_counts.get(severity_score, 0) + 1
        else:
            no_severity += 1
    average = sum(value * count for value, count in cve_severity_counts.items()) / sum(
        cve_severity_counts.values()
    )
    fraction_critical = sum(
        count for value, count in cve_severity_counts.items() if value >= 9
    ) / sum(cve_severity_counts.values())
    cve_severity_fractions = {
        value: count / sum(cve_severity_counts.values())
        for value, count in cve_severity_counts.items()
    }
    return {
        "no_severity": no_severity,
        "average": average,
        "fraction_critical": fraction_critical,
        "values": dict(sorted(cve_severity_fractions.items())),
    }


def plot_cve_severities(instruction_template_hash: str, scoring_model: str):
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    cve_ids: set[str] = set()
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        score = scores[0]
        cve_ids.update(set(score.mapping.values()))
    tp_vulns = [get_vulns(id=cve_id)[0] for cve_id in cve_ids]
    tp_stats = get_severity_stats(tp_vulns)
    tp_cve_severities = tp_stats["values"]
    all_vulns = get_vulns()
    all_stats = get_severity_stats(all_vulns)
    all_cve_severities = all_stats["values"]
    results = {
        "tp": tp_stats,
        "all": all_stats,
    }

    with open(Config.paths.results / "cve_severities.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    # Plot the distribution with a bin size of 0.5.

    severity_colors = {
        "None": "gray",
        "Low": "green",
        "Medium": "yellow",
        "High": "orange",
        "Critical": "red",
    }

    tp_cve_severities_05: dict[float, int] = {}
    for severity_score in tp_cve_severities:
        bin = int(severity_score * 2) / 2
        tp_cve_severities_05[bin] = (
            tp_cve_severities_05.get(bin, 0) + tp_cve_severities[severity_score]
        )
    all_cve_severities_05: dict[float, int] = {}
    for severity_score in all_cve_severities:
        bin = int(severity_score * 2) / 2
        all_cve_severities_05[bin] = (
            all_cve_severities_05.get(bin, 0) + all_cve_severities[severity_score]
        )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    tp_df = pd.DataFrame(
        {
            "severity": list(tp_cve_severities_05.keys()),
            "occurrences": list(tp_cve_severities_05.values()),
        }
    )
    tp_df = tp_df.sort_values("severity", ascending=True)
    all_df = pd.DataFrame(
        {
            "severity": list(all_cve_severities_05.keys()),
            "occurrences": list(all_cve_severities_05.values()),
        }
    )
    all_df = all_df.sort_values("severity", ascending=True)

    n_tp = len(tp_vulns)
    n_all = len(all_vulns)

    for category, color in severity_colors.items():
        tp_category = tp_df[tp_df["severity"].apply(get_severity_category) == category]
        all_category = all_df[all_df["severity"].apply(get_severity_category) == category]
        if not tp_category.empty:
            fig.add_trace(
                go.Bar(
                    x=tp_category["severity"],
                    y=tp_category["occurrences"],
                    name=category,
                    marker_color=color,
                    legendgroup="tp",
                    legendgrouptitle_text=f"True Positives (n={n_tp})",
                ),
                row=1,
                col=1,
            )
        if not all_category.empty:
            fig.add_trace(
                go.Bar(
                    x=all_category["severity"],
                    y=all_category["occurrences"],
                    name=category,
                    marker_color=color,
                    opacity=0.8,
                    marker={
                        "color": color,
                        "opacity": 0.8,
                        "pattern": {
                            "shape": "/",
                            "bgcolor": "white",
                            "fgcolor": color,
                            "size": 4,
                            "solidity": 0.7,
                        },
                    },
                    legendgroup="all",
                    legendgrouptitle_text=f"All Vulnerabilities (n={n_all})",
                ),
                row=2,
                col=1,
            )
    nticks = len(set(tp_cve_severities_05.keys() | set(all_cve_severities_05.keys()))) + 1
    fig.update_layout(template="plotly_white", margin=dict(t=5, l=5, r=5, b=5), font_size=14)

    fig.update_xaxes(title_text="Base Severity (CVSS v3)", nticks=nticks, row=2, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=1, tickformat=",.0%", range=[0, 0.22])
    fig.update_yaxes(title_text="Rate", row=2, col=1, tickformat=",.0%", range=[0, 0.22])

    fig.write_image(Config.paths.plots / "cve_severities.pdf")


def plot_costs(
    instruction_template_hash: str,
    scoring_model: str,
    model_order: list[str],
    model_names: dict[str, str],
):
    """Plot the cost per vulnerability of each model, in terms of inference cost and false
    positives."""
    results: dict[str, dict] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        results.setdefault(
            attempt.model,
            {
                "inference_cost": 0.0,
                "fp": 0,
                "tp": 0,
                "fp_per_tp": 0.0,
                "inference_cost_per_tp": 0.0,
            },
        )
        # Only keep the first score.
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        score = scores[0]
        results[attempt.model]["inference_cost"] += sum(
            response.usage.cost for response in attempt.responses
        )
        results[attempt.model]["fp"] += score.stats.fp
        results[attempt.model]["tp"] += score.stats.tp

    for model in results:
        results[model]["fp_per_tp"] = results[model]["fp"] / results[model]["tp"]
        results[model]["inference_cost_per_tp"] = (
            results[model]["inference_cost"] / results[model]["tp"]
        )

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.reindex(model_order)
    df["model_name"] = df.index.map(model_names)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df["inference_cost_per_tp"],
            marker_color="rgb(69, 126, 172)",
            text=[f"${x:.2f}" for x in df["inference_cost_per_tp"]],
            textposition="outside",
            textfont=dict(size=14),
            name="Inference cost per true positive",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["model_name"],
            y=df["fp_per_tp"],
            marker_color="rgb(194, 175, 240)",
            text=df["fp_per_tp"].round(1),
            textposition="outside",
            textfont=dict(size=14),
            name="False positives per true positive",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        margin=dict(t=5, l=5, r=5, b=5),
        font_size=16,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_yaxes(row=1, col=1, tickprefix="$")
    fig.update_yaxes(row=1, col=1, range=[0, df["inference_cost_per_tp"].max() * 1.4])
    fig.update_yaxes(row=2, col=1, range=[0, df["fp_per_tp"].max() * 1.3])

    fig.write_image(Config.paths.plots / "costs.pdf")


if __name__ == "__main__":
    instruction_template_hash = "245ace12b6361954d0a2"
    scoring_model = "claude-3-5-sonnet-20240620"
    model_order = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gemini/gemini-1.5-pro",
    ]

    model_names = {
        "claude-3-haiku-20240307": "Claude 3 Haiku",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet",
        "claude-3-opus-20240229": "Claude 3 Opus",
        "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
        "gpt-4o-2024-05-13": "GPT-4o",
        "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
        "gemini/gemini-1.5-pro": "Gemini 1.5 Pro",
    }

    color_map = {
        "claude-3-haiku-20240307": "rgb(252, 244, 10)",
        "claude-3-sonnet-20240229": "rgb(255, 139, 15)",
        "claude-3-opus-20240229": "rgb(255, 0, 0)",
        "claude-3-5-sonnet-20240620": "rgb(255, 0, 225)",
        "gpt-4o-2024-05-13": "rgb(15, 212, 40)",
        "gpt-4-turbo-2024-04-09": "rgb(7, 99, 19)",
        "gemini/gemini-1.5-pro": "rgb(2, 14, 150)",
    }

    training_data_cutoffs = {
        "claude-3-haiku-20240307": "2023-09-01",
        "claude-3-sonnet-20240229": "2023-09-01",
        "claude-3-opus-20240229": "2023-09-01",
        "claude-3-5-sonnet-20240620": "2024-05-01",
        "gpt-4o-2024-05-13": "2023-11-01",
        "gpt-4-turbo-2024-04-09": "2024-01-01",
        "gemini/gemini-1.5-pro": "2023-12-01",
    }
    plot_overall_performance(
        instruction_template_hash, scoring_model, model_order, model_names, color_map
    )
    plot_cwes_found(instruction_template_hash, scoring_model, top_n=10)
    plot_performance_before_after_training_cutoff(
        instruction_template_hash,
        scoring_model,
        model_order,
        model_names,
        color_map,
        training_data_cutoffs,
    )
    average_number_of_chunks_by_model(instruction_template_hash, scoring_model)
    plot_cve_severities(instruction_template_hash, scoring_model)
    plot_costs(instruction_template_hash, scoring_model, model_order, model_names)
