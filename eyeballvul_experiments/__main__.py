import logging
import mimetypes
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from eyeballvul import EyeballvulRevision, get_commits, get_revision
from litellm import completion, model_cost
from litellm.exceptions import ContextWindowExceededError
from typeguard import typechecked

from eyeballvul_experiments.attempt import Attempt, Response, Usage
from eyeballvul_experiments.chunk import Chunk, File
from eyeballvul_experiments.config.config_loader import Config

logging.basicConfig(level=logging.INFO)


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
            logging.info(f"Not keeping {filename} based on filename")
            continue
        try:
            with open(repo_dir / filename) as f:
                contents = f.read()
        except UnicodeDecodeError:
            logging.info(f"Not keeping {filename} based on UnicodeDecodeError")
            continue
        if not keep_file_based_on_contents(repo_dir / filename, contents):
            logging.info(f"Not keeping {filename} based on contents")
            continue
        logging.info(f"Keeping {filename}")
        files.append(File(filename=str(filename), contents=contents))
    return files


def fake_completion(chunk: Chunk):
    if len(chunk) > 128000 * 4:
        raise ValueError()
    return ["vul1", "vul2"], (len(chunk), 0)


def parse_response(response: str) -> list[str]:
    """
    Parse the model's response into a list of leads.

    Raise ValueError if the response can't be parsed.
    """
    return []  # TODO


@typechecked
def query_model_one_chunk(model: str, chunk: Chunk) -> tuple[str, Usage]:
    """
    Run `model` on the given `chunk`, and return a tuple of:

    - the response from the model
    - a tuple (input tokens used, output tokens used)
    """
    logging.info(f"Querying model {model} on chunk {chunk.get_hash()}...")
    response = completion(
        model=model,
        messages=[
            {"role": "user", "content": Config.instruction_template.format(chunk=chunk.full_str())}
        ],
    )
    usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
    return (response.choices[0].message.content, usage)


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


def query_model(model: str, revision: EyeballvulRevision, repo_dir: Path) -> Attempt:
    """
    Run `model` on the repository at `commit`, and return a new `Attempt` object (not yet scored).

    The model may be queried multiple times on chunks of the repository at that commit if the repository exceeds the model's context length.

    This method works the following way:
    - the repository is checked out at the given commit
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
    max_context_bytes = int(model_cost[model]["max_input_tokens"] * 4)
    subprocess.check_call(["git", "checkout", revision.commit], cwd=repo_dir)
    files = get_files_in_repo(repo_dir)
    included_repo_size = sum(len(file.contents) for file in files)
    while files:
        chunk = create_chunk_initial_guess(files, revision, max_context_bytes, included_repo_size)
        files = files[len(chunk.files) :]
        move_to_next_chunk = False
        while not move_to_next_chunk:
            try:
                response, usage = query_model_one_chunk(model, chunk)
                chunk.log()
                attempt.chunk_hashes.append(chunk.get_hash())
                attempt.responses.append(Response(content=response, date=datetime.now()))
                attempt.update_usage_and_cost(usage)
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
def handle_repo(repo_url: str, revisions: list[EyeballvulRevision], model: str) -> None:
    """Handle all the given `revisions` of the repository at `repo_url`."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
        for revision in revisions:
            attempt = query_model(model, revision, repo_dir)
            attempt.parse()
            attempt.add_score()
            attempt.log()


model = "gpt-4o"
project = "https://github.com/parisneo/lollms-webui"
revisions = [get_revision(commit) for commit in get_commits(project=project)][:1]
handle_repo(project, revisions, model)
