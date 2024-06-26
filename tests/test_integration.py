# This test is a modified section of pre-commit's configuration schema, for the sake of emulating a real-world use case.
# See pre_commit/clientlib.py.

import re
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


def _check_max_version(version: str) -> bool:
    return _parse_version(version) <= VERSION


_check_type_tag = ALL_TAGS.__contains__


def _transform_stage(stage: str) -> str:
    return _STAGES.get(stage, stage)


def _transform_stages(stages: list[str]) -> list[str]:
    return [_transform_stage(stage) for stage in stages]


def _is_subset_of_valid_stages(potential_stages: list[str]) -> bool:
    return {_transform_stage(stage) for stage in potential_stages}.issubset(STAGES)


def _is_in_hook_types(hook_types: list[str]) -> bool:
    return set(HOOK_TYPES).issubset(hook_types)


def _is_valid_regex(value: bytes | str) -> bool:
    try:
        re.compile(value)
    except re.error:
        return False
    else:
        return True


class ManifestHook(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(_check_max_version)] = "0"
    id: str
    name: str
    entry: str
    language: Annotated[str, spec.validate(_check_type_tag)]
    alias: str = ""
    files: str = ""
    exclude: str = "^$"
    types: Annotated[list[str], spec.validate(_check_type_tag)] = ["file"]
    types_or: Annotated[list[str], spec.validate(_check_type_tag)] = []
    additional_dependencies: list[str] = []
    args: list[str] = []
    fail_fast: bool = False
    pass_filenames: bool = True
    description: str = DEFAULT
    log_file: str = ""
    require_serial: bool = False
    stages: Annotated[list[str], spec.validate(_is_subset_of_valid_stages).hook(_transform_stages)] = []
    verbose: bool = False


class ConfigSchema(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(_check_max_version)] = "0"
    # repos
    default_install_hook_types: Annotated[list[str], spec.validate(_is_in_hook_types)] = ["pre-commit"]
    # default_stages: list[str] = STAGES
    files: Annotated[str, spec.validate(_is_valid_regex)] = ""
    exclude: Annotated[str, spec.validate(_is_valid_regex)] = "^$"
    fail_fast: bool = False


def test_manifest_hook() -> None:
    dct = {
        "id": "fake-hook",
        "name": "fake-hook",
        "entry": "fake-hook",
        "language": "system",
        "stages": ["commit-msg", "push", "commit", "merge-commit"],
    }
    processed = ManifestHook(dct)

    assert processed.stages == ["commit-msg", "pre-push", "pre-commit", "pre-merge-commit"]


@pytest.mark.parametrize(
    "manifest_obj",
    [
        [
            {
                "id": "a",
                "name": "b",
                "entry": "c",
                "language": "python",
                "files": r"\.py$",
            }
        ],
        [
            {
                "id": "a",
                "name": "b",
                "entry": "c",
                "language": "python",
                "language_version": "python3.4",
                "files": r"\.py$",
            }
        ],
        # A regression in 0.13.5: always_run and files are permissible
        [
            {
                "id": "a",
                "name": "b",
                "entry": "c",
                "language": "python",
                "files": "",
                "always_run": True,
            }
        ],
    ],
)
def test_valid_manifests(manifest_obj: list[dict[str, str]] | list[dict[str, str | bool]]) -> None:
    _ = [ManifestHook(item) for item in manifest_obj]
