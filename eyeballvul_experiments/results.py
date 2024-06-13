import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from confidenceinterval import f1_score, precision_score, recall_score
from eyeballvul import EyeballvulScore

from eyeballvul_experiments.attempt import Attempt
from eyeballvul_experiments.config.config_loader import Config


def get_scores_with_hash(attempt: Attempt, instruction_template_hash: str) -> list[EyeballvulScore]:
    return [
        score
        for score in attempt.scores
        if score.instruction_template_hash == instruction_template_hash
    ]


def plot_overall_performance(
    instruction_template_hash: str, model_order: list[str], color_map: dict[str, str]
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
        scores = get_scores_with_hash(attempt, instruction_template_hash)
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
            x=df["model"],
            y=df["precision"],
            name="Precision",
            marker_color="blue",
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
            x=df["model"],
            y=df["recall"],
            name="Recall",
            marker_color="green",
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
            x=df["model"],
            y=df["f1"],
            name="F1 Score",
            marker_color="yellow",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["f1_ci_upp"] - df["f1"],
                "arrayminus": df["f1"] - df["f1_ci_low"],
            },
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        width=800,
        height=600,
        font=dict(size=12),
    )
    fig.write_image(Config.paths.plots / "overall_performance.png")

    # Then plot a Pareto efficiency plot.
    traces = []
    for model in df["model"]:
        trace = go.Scatter(
            x=df.loc[df["model"] == model, "precision"],
            y=df.loc[df["model"] == model, "recall"],
            mode="markers",
            name=model,
            marker=dict(color=color_map[model]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df.loc[df["model"] == model, "precision_ci_upp"]
                - df.loc[df["model"] == model, "precision"],
                arrayminus=df.loc[df["model"] == model, "precision"]
                - df.loc[df["model"] == model, "precision_ci_low"],
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=df.loc[df["model"] == model, "recall_ci_upp"]
                - df.loc[df["model"] == model, "recall"],
                arrayminus=df.loc[df["model"] == model, "recall"]
                - df.loc[df["model"] == model, "recall_ci_low"],
            ),
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Precision",
        yaxis_title="Recall",
        width=800,
        height=600,
        font=dict(size=12),
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    fig.write_image(Config.paths.plots / "pareto_efficiency.png")


def plot_performance_before_after_training_cutoff(
    instruction_template_hash: str,
    model_order: list[str],
    color_map: dict[str, str],
    cutoff_dates: dict[str, str],
):
    results: dict[str, dict] = {}
    reconstructed_classifications: dict[str, dict] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        results.setdefault(attempt.model, {"fp": 0})
        for key in "before", "after":
            results[attempt.model].setdefault(
                key,
                {
                    "tp": 0,
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
        scores = get_scores_with_hash(attempt, instruction_template_hash)
        if not scores:
            continue
        score = scores[0]
        stats = score.stats_with_cutoff(
            cutoff_date=datetime.fromisoformat(cutoff_dates[attempt.model])
        )
        results[attempt.model]["fp"] += stats.fp
        results[attempt.model]["before"]["tp"] += stats.before.tp
        results[attempt.model]["before"]["fn"] += stats.before.fn
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
            # tp
            reconstructed_classifications[model][key]["y_true"].extend(
                [1] * results[model][key]["tp"]
            )
            reconstructed_classifications[model][key]["y_pred"].extend(
                [1] * results[model][key]["tp"]
            )

            # fp
            reconstructed_classifications[model][key]["y_true"].extend([0] * results[model]["fp"])
            reconstructed_classifications[model][key]["y_pred"].extend([1] * results[model]["fp"])

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
            classified_positive = results[model][key]["tp"] + results[model]["fp"]
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
            name=f"{model} (before)",
            marker=dict(color=color_map[model], symbol="circle-open"),
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
            name=f"{model} (after)",
            marker=dict(color=color_map[model], symbol="circle"),
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
        xaxis_title="Precision",
        yaxis_title="Recall",
        width=800,
        height=600,
        font=dict(size=12),
        legend=dict(title="Model (Period)"),
    )

    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")

    fig.write_image(Config.paths.plots / "pareto_efficiency_before_after_training_cutoff.png")


def plot_cwes_found(instruction_template_hash: str, top_n: int):
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
    }
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        scores = get_scores_with_hash(attempt, instruction_template_hash)
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
    fig.update_layout(
        template="plotly_white",
        xaxis={"title": "Frequency among true positives"},
    )
    fig.write_image(Config.paths.plots / "cwes_found.png")


if __name__ == "__main__":
    instruction_template_hash = "245ace12b6361954d0a2"
    model_order = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gemini/gemini-1.5-pro",
    ]

    color_map = {
        "claude-3-haiku-20240307": "rgb(252, 244, 10)",
        "claude-3-sonnet-20240229": "rgb(255, 139, 15)",
        "claude-3-opus-20240229": "rgb(255, 0, 0)",
        "gpt-4o-2024-05-13": "rgb(15, 212, 40)",
        "gpt-4-turbo-2024-04-09": "rgb(7, 99, 19)",
        "gemini/gemini-1.5-pro": "rgb(2, 14, 150)",
    }

    training_data_cutoffs = {
        "claude-3-haiku-20240307": "2023-09-01",
        "claude-3-sonnet-20240229": "2023-09-01",
        "claude-3-opus-20240229": "2023-09-01",
        "gpt-4o-2024-05-13": "2023-11-01",
        "gpt-4-turbo-2024-04-09": "2024-01-01",
        "gemini/gemini-1.5-pro": "2023-12-01",
    }
    plot_overall_performance(instruction_template_hash, model_order, color_map)
    plot_cwes_found(instruction_template_hash, top_n=10)
    plot_performance_before_after_training_cutoff(
        instruction_template_hash, model_order, color_map, training_data_cutoffs
    )
