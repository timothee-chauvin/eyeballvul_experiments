import hashlib

from typeguard import typechecked


@typechecked
def get_str_weak_hash(s: str) -> str:
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:32]
