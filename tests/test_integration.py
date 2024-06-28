# This test is a modified section of pre-commit's configuration schema, for the sake of emulating a real-world use case.
# See pre_commit/clientlib.py.

import re
import shlex
import sys
from typing import Annotated, Any

import pytest

import spec

DEFAULT = "1.0.0"
VERSION = "0.1.0"

ALL_TAGS = {
    ".tgz",
    ".py",
    ".js",
}

LANGUAGE_NAMES = {
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

CONFIG_FILE = ".pre-commit-config.yaml"


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def _is_le_max_version(version: str) -> bool:
    return _parse_version(version) <= _parse_version(VERSION)


def _migrate_stage(stage: str) -> str:
    return _STAGES.get(stage, stage)


def _migrate_stages(stages: list[str]) -> list[str]:
    return [_migrate_stage(stage) for stage in stages]


def _is_subset_of_valid_stages(potential_stages: list[str]) -> bool:
    return set(_migrate_stages(potential_stages)).issubset(STAGES)


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
    minimum_pre_commit_version: Annotated[str, spec.validate(_is_le_max_version)] = "0"
    id: str
    name: str
    entry: str
    language: Annotated[str, spec.validate(LANGUAGE_NAMES.__contains__)]
    alias: str = ""
    files: str = ""
    exclude: str = "^$"
    types: Annotated[list[str], spec.validate(ALL_TAGS.__contains__).default(lambda: ["file"])]
    types_or: Annotated[list[str], spec.validate(ALL_TAGS.__contains__).default(list)]
    additional_dependencies: Annotated[list[str], spec.default(list)]
    args: Annotated[list[str], spec.default(list)]
    always_run: bool = False
    fail_fast: bool = False
    pass_filenames: bool = True
    description: str = ""
    language_version: str = DEFAULT
    log_file: str = ""
    require_serial: bool = False
    stages: Annotated[list[str] | None, spec.validate(_is_subset_of_valid_stages).hook(_migrate_stages)] = None
    verbose: bool = False


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
def test_valid_manifests(manifest_obj: list[dict[str, Any]]) -> None:
    _ = [ManifestHook(item) for item in manifest_obj]


# =========================================================================


def on_meta_extras(ek: set[str], ak: set[str], data: dict[str, Any]) -> None:
    if "entry" in ek:
        msg = "'entry' cannot be overriden."
        raise spec.FailedValidation(msg)


def _entry(name: str) -> str:
    return f"{shlex.quote(sys.executable)} -m pre_commit.meta_hooks.{name}"


_check_hooks_apply_name = "check-hooks-apply"
_check_useless_excludes_name = "check-useless-excludes"
_identity_name = "identity"

_config_file_pattern = f"^{re.escape(CONFIG_FILE)}$"


class BaseMetaHook(ManifestHook, extras_policy=on_meta_extras):
    language: Annotated[str, spec.validate("system".__eq__)] = "system"


class CheckHooksApplyMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_check_hooks_apply_name.__eq__)] = _check_hooks_apply_name
    files: Annotated[str, spec.validate(_config_file_pattern.__eq__)]
    entry: Annotated[str, spec.validate(_entry("check_hooks_apply").__eq__)]


class CheckUselessExcludesMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_check_useless_excludes_name.__eq__)] = _check_useless_excludes_name
    files: Annotated[str, spec.validate(_config_file_pattern.__eq__)]
    entry: Annotated[str, spec.validate(_entry("check-useless-excludes").__eq__)]


class IdentityMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_identity_name.__eq__)] = _identity_name
    verbose: bool = True
    entry: Annotated[str, spec.validate(_entry("check-useless-excludes").__eq__)]


class MetaHook(spec.TransparentModel[CheckHooksApplyMetaHook | CheckUselessExcludesMetaHook | IdentityMetaHook]):
    pass


class ConfigHook(ManifestHook): ...


class LocalHook(ManifestHook): ...


class DefaultLanguageVersion(spec.Model, extras_policy="deny"):
    pass


class ConfigSchema(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(_is_le_max_version)] = "0"
    # repos
    default_install_hook_types: Annotated[list[str], spec.validate(_is_in_hook_types).default(lambda: ["pre-commit"])]
    default_language_version: DefaultLanguageVersion
    default_stages: Annotated[list[str], spec.default(lambda: list(STAGES))]
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
