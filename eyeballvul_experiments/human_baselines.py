import json
import random
from typing import Any

import numpy as np
from eyeballvul import EyeballvulScore, get_vulns

from eyeballvul_experiments.attempt import Attempt
from eyeballvul_experiments.config.config_loader import Config

# The few shot examples used in the scoring prompt are taken from these projects.
# They must not be included in the human agreement baseline.
few_shot_examples_taken_from_projects = [
    "https://github.com/bootboxjs/bootbox",
    "https://github.com/apostrophecms/sanitize-html",
    "https://github.com/benbusby/whoogle-search",
    "https://github.com/mde/ejs",
    "https://github.com/free5gc/free5gc",
    "https://github.com/demon1a/discord-recon",
]


def get_scores_with_hash(attempt: Attempt, instruction_template_hash: str) -> list[EyeballvulScore]:
    return [
        score
        for score in attempt.scores
        if score.instruction_template_hash == instruction_template_hash
    ]


def generate_random_sample_human_agreement(instruction_template_hash: str, n: int):
    """
    Generate a random sample of `n` vulnerability submissions.

    Store that sample, along with the LLM scorer scores, in Config.paths.human_baselines /
    "sample.json".

    Store a presentation of that sample for human scorers (without LLM scores) in
    Config.paths.human_baselines / "sample_for_humans.json".
    """
    samples = []
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in attempt_filenames:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        if attempt.repo_url in few_shot_examples_taken_from_projects:
            continue
        scores = get_scores_with_hash(attempt, instruction_template_hash)
        if not scores:
            continue
        # Only select the first score for each attempt.
        score = scores[0]
        selected_leads = [lead for lead in attempt.leads if lead.classification == "very promising"]
        vulns = get_vulns(commit=attempt.commit)
        for i, lead in enumerate(selected_leads):
            label = int(i in score.mapping)
            samples.append(
                {
                    "attempt": attempt.get_hash(),
                    "lead_index": i,
                    "score": label,
                    "lead": lead.model_dump(mode="json"),
                    "vulns": {v.id: v.details for v in vulns},
                }
            )
    random.Random(0).shuffle(samples)  # nosec
    subset = samples[:n]
    with open(Config.paths.human_baselines / "sample.json", "w") as f:
        json.dump(
            {
                "instruction_template_hash": instruction_template_hash,
                "sample": [
                    {k: sample[k] for k in ["attempt", "lead_index", "score"]} for sample in subset
                ],
            },
            f,
            indent=2,
        )
        f.write("\n")
    presentation_for_humans = {}
    for i, sample in enumerate(subset):
        presentation_for_humans[i + 1] = {k: sample[k] for k in ["attempt", "lead", "vulns"]}
    with open(Config.paths.human_baselines / "sample_for_humans.json", "w") as f:
        json.dump(presentation_for_humans, f, indent=2)
        f.write("\n")


def get_llm_score() -> list[int]:
    with open(Config.paths.human_baselines / "sample.json") as f:
        data = json.load(f)
    return [el["score"] for el in data["sample"]]


def get_human_score(filename: str) -> list[int]:
    with open(Config.paths.human_baselines / filename) as f:
        data = json.load(f)
    maximum = max([int(k) for k in data.keys()])
    return [data[str(i + 1)]["score"] for i in range(maximum)]


def cohen_kappa(score1: list[int], score2: list[int]) -> tuple[list, float]:
    n = len(score1)
    # Create contingency table
    table = np.zeros((2, 2))
    for i in range(n):
        table[score1[i], score2[i]] += 1
    print(table)

    # Calculate observed agreement
    po = (table[0, 0] + table[1, 1]) / n

    # Calculate expected agreement
    pe = (
        (table[0, 0] + table[0, 1]) * (table[0, 0] + table[1, 0])
        + (table[1, 0] + table[1, 1]) * (table[0, 1] + table[1, 1])
    ) / (n * n)

    # Compute kappa
    kappa = (po - pe) / (1 - pe)

    return table.tolist(), kappa


def cohen_kappas(filenames: list[str]):
    llm_score = get_llm_score()
    results: dict[str, tuple[list, float]] = {}
    for filename in filenames:
        score = get_human_score(filename)
        results[filename] = cohen_kappa(llm_score, score)
    average = sum([res[1] for res in results.values()]) / len(results)
    with open(Config.paths.results / "cohen_kappa.json", "w") as f:
        json.dump({"individual": results, "average": average}, f, indent=2)
        f.write("\n")


def average_confidence(filenames: list[str]):
    llm_score = get_llm_score()
    results: dict[str, dict[str, Any]] = {"individual": {}, "average": {}}
    for filename in filenames:
        with open(Config.paths.human_baselines / filename) as f:
            data = json.load(f)
        maximum = max([int(k) for k in data.keys()])
        score = [data[str(i + 1)]["score"] for i in range(maximum)]
        confidence = [data[str(i + 1)]["confidence"] for i in range(maximum)]
        average_confidence = sum(confidence) / len(confidence)
        disagreement_indices = [i for i in range(maximum) if score[i] != llm_score[i]]
        confidence_on_disagreement = [confidence[i] for i in disagreement_indices]
        average_confidence_on_disagreement = sum(confidence_on_disagreement) / len(
            confidence_on_disagreement
        )
        results["individual"][filename] = {
            "confidence": average_confidence,
            "confidence_on disagreement": average_confidence_on_disagreement,
        }
    results["average"] = {
        "confidence": sum(item["confidence"] for item in results["individual"].values())
        / len(results["individual"]),
        "confidence_on_disagreement": sum(
            item["confidence_on disagreement"] for item in results["individual"].values()
        )
        / len(results["individual"]),
    }
    with open(Config.paths.results / "confidence.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    instruction_template_hash = "245ace12b6361954d0a2"
    generate_random_sample_human_agreement(instruction_template_hash, 100)
    cohen_kappas(["score_a.json", "score_b.json", "score_c.json"])
    average_confidence(["score_a.json", "score_b.json", "score_c.json"])
