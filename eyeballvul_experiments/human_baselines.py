import json
import random
from typing import Any

import numpy as np
from eyeballvul import EyeballvulScore, get_vulns
from tqdm import tqdm

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


def get_scores_with_hash(
    attempt: Attempt, instruction_template_hash: str, scoring_model: str
) -> list[EyeballvulScore]:
    return [
        score
        for score in attempt.scores
        if score.instruction_template_hash == instruction_template_hash
        and score.scoring_model == scoring_model
    ]


def generate_random_sample_human_agreement(
    instruction_template_hash: str, scoring_model: str, n: int
):
    """
    Generate a random sample of `n` vulnerability submissions.

    Store that sample in Config.paths.human_baselines / "sample.json".

    Store a presentation of that sample for human scorers in Config.paths.human_baselines /
    "sample_for_humans.json".
    """
    samples = []
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    for attempt_filename in tqdm(attempt_filenames):
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
        if attempt.repo_url in few_shot_examples_taken_from_projects:
            continue
        scores = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)
        if not scores:
            continue
        selected_leads = [lead for lead in attempt.leads if lead.classification == "very promising"]
        vulns = get_vulns(commit=attempt.commit)
        for i, lead in enumerate(selected_leads):
            samples.append(
                {
                    "attempt": attempt.get_hash(),
                    "lead_index": i,
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
                "sample": [{k: sample[k] for k in ["attempt", "lead_index"]} for sample in subset],
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


def get_llm_score(
    instruction_template_hash: str, sample_element: dict[str, int | str], scoring_model: str
) -> int:
    with open(Config.paths.attempts / f"{sample_element['attempt']}.json") as f:
        attempt = Attempt.model_validate_json(f.read())
    # Only considering the first score
    full_score = get_scores_with_hash(attempt, instruction_template_hash, scoring_model)[0]
    return int(sample_element["lead_index"] in full_score.mapping)


def get_llm_scores(instruction_template_hash: str, scoring_model: str) -> list[int]:
    with open(Config.paths.human_baselines / "sample.json") as f:
        data = json.load(f)
    return [get_llm_score(instruction_template_hash, el, scoring_model) for el in data["sample"]]


def export_llm_scores(instruction_template_hash: str, scoring_model: str):
    llm_scores = get_llm_scores(instruction_template_hash, scoring_model)
    with open(
        Config.paths.human_baselines
        / f"llm_scores_{instruction_template_hash}_{scoring_model.replace('/', '_')}.json",
        "w",
    ) as f:
        json.dump(
            {
                "instruction_template_hash": instruction_template_hash,
                "scoring_model": scoring_model,
                "scores": {i + 1: score for i, score in enumerate(llm_scores)},
            },
            f,
            indent=2,
        )
        f.write("\n")


def get_human_scores(filename: str) -> list[int]:
    with open(Config.paths.human_baselines / filename) as f:
        data = json.load(f)
    maximum = max([int(k) for k in data.keys()])
    return [data[str(i + 1)]["score"] for i in range(maximum)]


def cohen_kappa(llm_score: list[int], human_score: list[int]) -> dict[str, Any]:
    n = len(llm_score)
    # Create contingency table
    table = np.zeros((2, 2))
    disagreements: dict[str, list[int]] = {"llm-positive": [], "llm-negative": []}
    for i in range(n):
        table[llm_score[i], human_score[i]] += 1
        if llm_score[i] == 1 and human_score[i] == 0:
            disagreements["llm-positive"].append(i + 1)
        elif llm_score[i] == 0 and human_score[i] == 1:
            disagreements["llm-negative"].append(i + 1)
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

    return {
        "contingency_table": table.tolist(),
        "kappa": kappa,
        "disagreements": disagreements,
    }


def cohen_kappas(instruction_template_hash: str, filenames: list[str], scoring_model: str):
    llm_scores = get_llm_scores(instruction_template_hash, scoring_model)
    results: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        human_scores = get_human_scores(filename)
        results[filename] = cohen_kappa(llm_scores, human_scores)
    average = sum([res["kappa"] for res in results.values()]) / len(results)
    all_llm_positive = [i + 1 for i, score in enumerate(llm_scores) if score == 1]
    with open(Config.paths.results / "cohen_kappa.json", "w") as f:
        json.dump(
            {"individual": results, "average": average, "all_llm_positive": all_llm_positive},
            f,
            indent=2,
        )
        f.write("\n")


def average_confidence(instruction_template_hash: str, filenames: list[str], scoring_model: str):
    llm_scores = get_llm_scores(instruction_template_hash, scoring_model)
    results: dict[str, dict[str, Any]] = {"individual": {}, "average": {}}
    for filename in filenames:
        with open(Config.paths.human_baselines / filename) as f:
            data = json.load(f)
        maximum = max([int(k) for k in data.keys()])
        scores = [data[str(i + 1)]["score"] for i in range(maximum)]
        confidence = [data[str(i + 1)]["confidence"] for i in range(maximum)]
        average_confidence = sum(confidence) / len(confidence)
        disagreement_indices = [i for i in range(maximum) if scores[i] != llm_scores[i]]
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


def generate_random_cves(n: int):
    vulns = get_vulns()
    subset = random.Random(0).sample(vulns, n)  # nosec
    presentation = [
        {"id": v.id, "repo_url": v.repo_url, "details": v.details, "cwes": v.cwes, "rating": -1}
        for v in subset
    ]
    with open(Config.paths.human_baselines / "random_cves.json", "w") as f:
        json.dump(presentation, f, indent=2)
        f.write("\n")


def stats_from_scored_cves(filename: str):
    with open(Config.paths.human_baselines / filename) as f:
        data = json.load(f)
    histogram = {i: 0 for i in range(1, 6)}
    for item in data:
        histogram[item["rating"]] += 1
    average = sum(k * v for k, v in histogram.items()) / sum(histogram.values())

    # For each rating, the number of items above this rating
    n_above = {}
    for i in range(1, 6):
        n_above[i] = sum(histogram[j] for j in range(i, 6))
    with open(Config.paths.human_baselines / "random_cve_stats.json", "w") as f:
        json.dump(
            {"histogram": histogram, "number_above": n_above, "average": average}, f, indent=2
        )
        f.write("\n")


if __name__ == "__main__":
    instruction_template_hash = "245ace12b6361954d0a2"
    scoring_model = "claude-3-5-sonnet-20240620"
    # generate_random_sample_human_agreement(instruction_template_hash, scoring_model, 100)
    generate_random_cves(n=100)
    export_llm_scores(instruction_template_hash, scoring_model)
    cohen_kappas(
        instruction_template_hash, ["score_a.json", "score_b.json", "score_c.json"], scoring_model
    )
    average_confidence(
        instruction_template_hash, ["score_a.json", "score_b.json", "score_c.json"], scoring_model
    )
    stats_from_scored_cves("random_cves_scored.json")
