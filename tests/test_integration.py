# This test is a modified section of pre-commit's configuration schema, for the sake of emulating a real-world use case.
# See pre_commit/clientlib.py.

from typing import Annotated

import pytest

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


def check_max_version(version: str) -> bool:
    return _parse_version(version) <= VERSION


check_type_tag = ALL_TAGS.__contains__


def check_valid_stage(value: str) -> bool:
    return _STAGES.get(value, value) in STAGES


class ManifestHook(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(check_max_version)] = "0"
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
    stages: Annotated[list[str], spec.validate(check_valid_stage)] = []
    verbose: bool = False


# @pytest.mark.xfail(reason="Something's currently wrong when validating a list.")
def test_manifest_hook() -> None:
    dct = {
        "id": "fake-hook",
        "name": "fake-hook",
        "entry": "fake-hook",
        "language": "system",
        "stages": ["commit-msg", "push", "commit", "merge-commit"],
    }
    processed = ManifestHook(dct)

    assert processed.stages == [
        "commit-msg",
        "pre-push",
        "pre-commit",
        "pre-merge-commit",
    ]
