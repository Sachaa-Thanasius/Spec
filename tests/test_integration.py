# ruff: noqa: RUF012
from typing import Annotated

import spec

DEFAULT = "1.0.0"
VERSION = (0, 1, 0)

ALL_TAGS = {
    "ruby",
    "python",
    "js",
    "c++",
    "system",
}

HOOK_TYPES = (
    "commit-msg",
    "post-checkout",
    "post-commit",
    "post-merge",
    "post-rewrite",
    "pre-commit",
    "pre-merge-commit",
    "pre-push",
    "pre-rebase",
    "prepare-commit-msg",
)
STAGES = (*HOOK_TYPES, "manual")


_STAGES = {
    "commit": "pre-commit",
    "merge-commit": "pre-merge-commit",
    "push": "pre-push",
}


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def check_min_version(version: str) -> bool:
    return _parse_version(version) <= VERSION


check_type_tag = ALL_TAGS.__contains__


class ManifestHook(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(check_min_version)] = "0"
    id: str
    name: str
    entry: str
    language: Annotated[str, spec.validate(check_type_tag)]
    alias: str = ""
    files: str = ""
    exclude: str = "^$"
    types: Annotated[list[str], spec.validate(check_type_tag)] = ["file"]
    types_or: Annotated[list[str], spec.validate(check_type_tag)] = []
    additional_dependencies: list[str] = []
    args: list[str] = []
    fail_fast: bool = False
    pass_filenames: bool = True
    description: str = DEFAULT
    log_file: str = ""
    require_serial: bool = False
    stages: Annotated[
        list[str],
        spec.validate(lambda lst: {_STAGES.get(v, v) for v in lst}.issubset(STAGES)),
    ] = []
    verbose: bool = False


# def test_manifest_hook() -> None:
#     dct = {
#         "id": "fake-hook",
#         "name": "fake-hook",
#         "entry": "fake-hook",
#         "language": "system",
#         "stages": ["commit-msg", "push", "commit", "merge-commit"],
#     }
#     processed = ManifestHook(dct)

#     assert processed.stages == [
#         "commit-msg",
#         "pre-push",
#         "pre-commit",
#         "pre-merge-commit",
#     ]
