import asyncio
import json
import logging
import mimetypes
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from eyeballvul import EyeballvulRevision, get_revisions
from eyeballvul.score import instruction_template_hash
from typeguard import typechecked

from eyeballvul_experiments.attempt import Attempt, SimpleResponse
from eyeballvul_experiments.chunk import Chunk, File
from eyeballvul_experiments.config.config_loader import Config
from eyeballvul_experiments.llm_gateway.gateway_interface import (
    APIConnectionError,
    ContextWindowExceededError,
    Usage,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s-%(asctime)s - %(message)s")

# Avoid seeing the info logs from these libraries
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RepositoryTooLargeError(Exception):
    def __init__(self, message, repo_size):
        super().__init__(message)
        self.repo_size = repo_size


class RepositoryEmptyError(Exception):
    pass


@typechecked
def keep_file_based_on_filename(filename: Path) -> bool:
    """
    Should the file be included in the chunk?

    First stage of filtering, based purely on the filename.
    """
    if any(str(filename).lower().endswith(extension) for extension in Config.exclude_extensions):
        return False
    if any(part.startswith(".") for part in filename.parts):
        return False
    return True


@typechecked
def keep_file_based_on_contents(full_filename: Path, contents: str) -> bool:
    """
    Should the file be included in the chunk?

    Second stage of filtering, based on the contents of the file.
    """
    if len(contents) > Config.exclude_files_above:
        return False
    mime_type, _ = mimetypes.guess_type(full_filename)
    if mime_type is None or not mime_type.startswith("text/"):
        return False
    return True


@typechecked
def get_files_in_repo(repo_dir: Path) -> list[File]:
    """Return a list of `File` objects extracted and filtered from the repository at `repo_dir`."""
    filenames_unfiltered: list[Path] = [
        file.relative_to(repo_dir) for file in repo_dir.rglob("*") if file.is_file()
    ]
    files: list[File] = []
    for filename in filenames_unfiltered:
        if not keep_file_based_on_filename(filename):
            logging.debug(f"Not keeping {filename} based on filename")
            continue
        try:
            with open(repo_dir / filename) as f:
                contents = f.read()
        except UnicodeDecodeError:
            logging.debug(f"Not keeping {filename} based on UnicodeDecodeError")
            continue
        if not keep_file_based_on_contents(repo_dir / filename, contents):
            logging.debug(f"Not keeping {filename} based on contents")
            continue
        logging.debug(f"Keeping {filename}")
        files.append(File(filename=str(filename), contents=contents))
    return files


@typechecked
async def query_model_one_chunk(model: str, chunk: Chunk) -> tuple[str, Usage]:
    """
    Run `model` on the given `chunk`, and return a tuple of:

    - the response from the model
    - the Usage object extracted from the response
    """
    logging.info(f"Querying model {model} on chunk {chunk.get_hash()}...")
    safety_settings = None
    if "gemini" in model:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    response = await Config.gateway.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": Config.instruction_template.format(
                    cwe_list=Config.cwe_list, chunk=chunk.full_str()
                ),
            }
        ],
        num_retries=3,
        safety_settings=safety_settings,
    )
    return (response.choices[0].message.content, response.usage)


@typechecked
def create_chunk_initial_guess(
    files: list[File], revision: EyeballvulRevision, max_context_bytes: int, included_repo_size: int
) -> Chunk:
    """
    Create the smallest chunk from the given `files`, taken in order, that is bigger than
    `max_context_bytes` (or contains all the files).

    `included_repo_size` is the total size of all files in the repository, in bytes (not the one provided by linguist, but the one here, after filtering out files). It is used to compute the fraction of the repository included in the chunk. This function will be called with shrinking lists of `files`, so this number must be provided each time.
    """
    chunk = Chunk(repo_url=revision.repo_url, commit=revision.commit, repo_size=included_repo_size)
    for file in files:
        if len(chunk) > max_context_bytes:
            break
        chunk.files.append(file)
    logging.info(f"Initial chunk guess: {len(chunk)} bytes, {len(chunk.files)} files")
    return chunk


@typechecked
async def query_model(
    model: str, revision: EyeballvulRevision, repo_dir: Path, max_size_bytes: int
) -> Attempt:
    """
    Run `model` on the repository at `revision` (with the commit already checked out in `repo_dir`),
    and return a new `Attempt` object (not yet scored), unless the repository exceeds
    `max_size_bytes`.

    The model may be queried multiple times on chunks of the repository at that commit if the repository exceeds the model's context length.

    This method works the following way:
    - the files at this commit are obtained
    - chunks are built incrementally, by trying to completely fill the context window.
        - if the context window is not exceeded, the response is extracted. We move on to the next chunk.
        - if the context window is exceeded, the chunk is reduced by at least one file and at most 5% of its size. This is tried again until it fits.
    """
    attempt = Attempt(
        commit=revision.commit,
        repo_url=revision.repo_url,
        model=model,
    )
    files = get_files_in_repo(repo_dir)
    included_repo_size = sum(len(file.contents) for file in files)
    if included_repo_size == 0:
        raise RepositoryEmptyError("Repository is empty")
    if included_repo_size > max_size_bytes:
        raise RepositoryTooLargeError("Repository too large", included_repo_size)
    while files:
        chunk = create_chunk_initial_guess(files, revision, max_size_bytes, included_repo_size)
        files = files[len(chunk.files) :]
        move_to_next_chunk = False
        while not move_to_next_chunk:
            try:
                response, usage = await query_model_one_chunk(model, chunk)
                chunk.log()
                attempt.chunk_hashes.append(chunk.get_hash())
                attempt.responses.append(
                    SimpleResponse(content=response, date=datetime.now(), usage=usage)
                )
                move_to_next_chunk = True
            except ContextWindowExceededError:
                removed_files: list[File] = chunk.shrink()
                if len(chunk.files) == 0:
                    # This means that the first file in the chunk was too large.
                    # Ignore this file.
                    logging.warning(
                        f"Ignoring file {removed_files[0].filename} at runtime because it is too large (size: {len(removed_files[0].contents)} bytes)."
                    )
                    files = removed_files[1:] + files
                    move_to_next_chunk = True
                else:
                    files = removed_files + files
    return attempt


@typechecked
async def do_attempt(
    model: str, revision: EyeballvulRevision, repo_dir: Path, max_size_bytes: int
) -> tuple[float, dict[str, int]]:
    try:
        attempt = await query_model(model, revision, repo_dir, max_size_bytes)
        attempt.log()  # in case something goes wrong later...
        attempt.parse()
        await attempt.add_score()
        attempt.log()
        return (attempt.cost(), {})
    except RepositoryEmptyError:
        logging.warning(f"Skipping revision {revision.commit} with {model} because it is empty.")
        return (0.0, {revision.commit: 0})
    except RepositoryTooLargeError as e:
        logging.warning(
            f"Skipping revision {revision.commit} with {model} because it is too large (size {e.repo_size})."
        )
        return (0.0, {revision.commit: e.repo_size})
    except APIConnectionError as e:
        if "Encountered text corresponding to disallowed special token " in str(e):
            logging.warning(
                f"Skipping revision {revision.commit} with {model} because of a disallowed special token: {e}"
            )
            return (0.0, {})
        else:
            raise


@typechecked
async def handle_repo(
    repo_url: str,
    revisions: list[EyeballvulRevision],
    models: list[str],
    max_size_bytes: int,
    attempts_by_commit: dict[str, list[Attempt]],
    cache: dict[str, int],
) -> tuple[float, dict[str, int]]:
    """
    Handle all the given `revisions` of the repository at `repo_url`, for all `models`.

    Return the total cost used in new invocations of models, and a cache update dictionary (possibly
    empty).
    """
    models_by_revision: dict[str, list[str]] = {}
    for revision in revisions:
        for model in models:
            if not already_attempted(attempts_by_commit, revision.commit, model):
                models_by_revision.setdefault(revision.commit, []).append(model)
    if not models_by_revision:
        logging.info(f"No new attempts to make for {repo_url}.")
        return 0.0, {}
    sizes = [cache[revision.commit] for revision in revisions if revision.commit in cache]
    if len(sizes) == len(revisions) and all(size == 0 or size > max_size_bytes for size in sizes):
        logging.info(
            f"Skipping {repo_url} because all revisions are known to be too large or empty."
        )
        return 0.0, {}
    total_cost = 0.0
    cache_update = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
        for revision in revisions:
            if revision.commit in cache:
                if cache[revision.commit] == 0:
                    logging.info(f"Skipping revision {revision.commit} because it is empty.")
                    continue
                if cache[revision.commit] > max_size_bytes:
                    logging.info(
                        f"Skipping revision {revision.commit} because it is too large (size {cache[revision.commit]})."
                    )
                    continue
            subprocess.check_call(
                ["git", "checkout", revision.commit],
                cwd=repo_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tasks = [
                asyncio.create_task(do_attempt(model, revision, repo_dir, max_size_bytes))
                for model in models_by_revision.get(revision.commit, [])
            ]
            for task in asyncio.as_completed(tasks):
                cost, new_cache_update = await task
                total_cost += cost
                cache_update.update(new_cache_update)
    return total_cost, cache_update


@typechecked
def get_attempts_by_commit() -> dict[str, list[Attempt]]:
    """Return a dictionary of attempts, grouped by commit."""
    attempts_by_commit: dict[str, list[Attempt]] = {}
    for attempt_file in Config.paths.attempts.glob("*.json"):
        with open(attempt_file) as f:
            attempt = Attempt.model_validate_json(f.read())
        attempts_by_commit.setdefault(attempt.commit, []).append(attempt)
    return attempts_by_commit


@typechecked
def already_attempted(
    attempts_by_commit: dict[str, list[Attempt]], commit: str, model: str
) -> bool:
    """Return whether the given `commit` has already been attempted with the given `model`."""
    return any(attempt.model == model for attempt in attempts_by_commit.get(commit, []))


@typechecked
def cost_of_past_attempts(attempts_by_commit: dict[str, list[Attempt]]) -> float:
    """Return the total cost of all past attempts."""
    return sum(attempt.cost() for attempts in attempts_by_commit.values() for attempt in attempts)


@typechecked
def read_cache() -> dict[str, int]:
    """Read the cache of repository sizes."""
    try:
        with open(Config.paths.cache / "cache.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@typechecked
def write_cache(cache: dict[str, int]) -> None:
    """Write the cache of repository sizes."""
    with open(Config.paths.cache / "cache.json", "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


async def run_models_on_revisions(
    revisions: list[EyeballvulRevision],
    models: list[str],
    max_size_bytes: int,
    cost_limit: float,
    attempts_by_commit: dict[str, list[Attempt]],
    cache: dict[str, int],
    current_total_cost: float,
):
    revisions_by_repo: dict[str, list[EyeballvulRevision]] = {}
    for revision in revisions:
        revisions_by_repo.setdefault(revision.repo_url, []).append(revision)
    revisions_by_repo = dict(sorted(revisions_by_repo.items()))
    total_cost = current_total_cost
    repo_url_len = len(revisions_by_repo)
    for i, (repo_url, revisions) in enumerate(revisions_by_repo.items()):
        logging.info(f"({i+1}/{repo_url_len}) Handling {repo_url}...")
        repo_cost, cache_update = await handle_repo(
            repo_url, revisions, models, max_size_bytes, attempts_by_commit, cache
        )
        total_cost += repo_cost
        if cache_update:
            logging.info("Updating cache.")
            cache.update(cache_update)
            write_cache(cache)
        logging.info(f"Total cost so far: ${total_cost:.2f}")
        if total_cost > cost_limit:
            logging.info(f"Cost limit reached: ${total_cost:.2f}")
            return


async def run_models():
    models = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gemini/gemini-1.5-pro",
    ]
    max_size_bytes = 600_000
    cost_limit = 2000

    cache = read_cache()

    attempts_by_commit = get_attempts_by_commit()
    total_cost = cost_of_past_attempts(attempts_by_commit)
    logging.info(f"Total cost of past attempts: ${total_cost:.2f}")

    # First, process all revisions that have at least one vuln past the cutoff date.
    cutoff_date = "2023-09-01"
    revisions_after = [
        revision for revision in get_revisions(after=cutoff_date) if revision.size < max_size_bytes
    ]
    await run_models_on_revisions(
        revisions=revisions_after,
        models=models,
        max_size_bytes=max_size_bytes,
        cost_limit=cost_limit,
        attempts_by_commit=attempts_by_commit,
        cache=cache,
        current_total_cost=total_cost,
    )

    # Then process all other revisions in alphabetical order of commit hash.
    other_revisions = [
        revision
        for revision in sorted(get_revisions(), key=lambda r: r.commit)
        if revision.size < max_size_bytes and revision not in revisions_after
    ]
    limit = 700
    other_revisions = other_revisions[:limit]
    await run_models_on_revisions(
        revisions=other_revisions,
        models=models,
        max_size_bytes=max_size_bytes,
        cost_limit=cost_limit,
        attempts_by_commit=get_attempts_by_commit(),
        cache=cache,
        current_total_cost=total_cost,
    )


async def re_score_attempt(
    semaphore, task_id: str, attempt_filename: str, scoring_model: str, add_if_exists: bool = False
):
    """Using a semaphore to avoid being rate-limited."""
    async with semaphore:
        with open(Config.paths.attempts / attempt_filename) as f:
            attempt = Attempt.model_validate_json(f.read())
            if not add_if_exists and any(
                score.instruction_template_hash == instruction_template_hash
                and score.scoring_model == scoring_model
                for score in attempt.scores
            ):
                logging.info(
                    f"({task_id}) Skipping {attempt.get_hash()} because it has already been scored."
                )
            else:
                logging.info(f"({task_id}) Scoring {attempt.get_hash()}...")
                await attempt.add_score()
                attempt.log()


async def re_score_all(scoring_model: str, add_if_exists: bool = False):
    """Re-score all attempts, except those that have already been scored with the current
    instruction template hash if `add_if_exists` is False."""
    attempt_filenames = [attempt.name for attempt in Config.paths.attempts.iterdir()]
    total = len(attempt_filenames)
    max_concurrent_tasks = 5
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        asyncio.create_task(
            re_score_attempt(semaphore, f"{i+1}/{total}", filename, scoring_model, add_if_exists)
        )
        for i, filename in enumerate(attempt_filenames)
    ]
    await asyncio.gather(*tasks)


async def main():
    await run_models()


if __name__ == "__main__":
    asyncio.run(main())
