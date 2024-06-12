import json

import pandas as pd
import plotly.graph_objects as go
from eyeballvul import EyeballvulScore
from statsmodels.stats.proportion import proportion_confint

from eyeballvul_experiments.attempt import Attempt
from eyeballvul_experiments.config.config_loader import Config


def get_scores_with_hash(attempt: Attempt, instruction_template_hash: str) -> list[EyeballvulScore]:
    return [
        score
        for score in attempt.scores
        if score.instruction_template_hash == instruction_template_hash
    ]


def plot_true_positive_rates(instruction_template_hash: str, model_order: list[str]):
    results: dict[str, dict] = {}
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        results.setdefault(
            attempt.model, {"tp": 0, "fp": 0, "rate": 0.0, "ci_low": 0.0, "ci_upp": 0.0}
        )
        for score in get_scores_with_hash(attempt, instruction_template_hash):
            results[attempt.model]["tp"] += score.stats.tp
            results[attempt.model]["fp"] += score.stats.fp
    for model in results:
        total = results[model]["tp"] + results[model]["fp"]
        rate = results[model]["tp"] / total
        results[model]["rate"] = rate
        ci_low, ci_upp = proportion_confint(
            results[model]["tp"], total, alpha=0.05, method="wilson"
        )
        results[model]["ci_low"] = ci_low
        results[model]["ci_upp"] = ci_upp

    with open(Config.paths.results / "true_positive_rates.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    df = pd.DataFrame(
        {
            "model": list(results.keys()),
            "true positive rate": [results[model]["rate"] for model in results],
            "low": [results[model]["ci_low"] for model in results],
            "high": [results[model]["ci_upp"] for model in results],
        }
    )
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["model"],
            y=df["true positive rate"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": df["high"] - df["true positive rate"],
                "arrayminus": df["true positive rate"] - df["low"],
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        yaxis={"title": "True Positive Rate"},
    )
    fig.write_image(Config.paths.plots / "true_positive_rates.png")


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
    print(plot_true_positive_rates(instruction_template_hash, model_order))
