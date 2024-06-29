"""This integration test is an attempt at satisfying pre-commit's configuration schemas for config files.

This is purely for the sake of emulating a real-world use case. Some code is copied from pre-commit/clientlib.py,
and the schema comes from there as well. All rights to Anthony Sottile for those.
"""
# ruff: noqa: T201, T203

import re
import shlex
import sys
from typing import Annotated, Any, Literal

import pytest

import spec

# region -------- Constants --------


PRE_COMMIT_VERSION = "0.1.0"
CONFIG_FILE = ".pre-commit-config.yaml"
CONFIG_FILE_REGEX = f"^{re.escape(CONFIG_FILE)}$"

ALL_FILE_TAGS = (
    ".js",
    ".py",
    ".tgz",
)

LANGUAGE_NAMES = (
    "conda",
    "dotnet",
    "python",
    "ruby",
    "system",
)

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
VALID_STAGES = (*HOOK_TYPES, "manual")


_OLD_STAGES = {
    "commit": "pre-commit",
    "merge-commit": "pre-merge-commit",
    "push": "pre-push",
}


# endregion


# region -------- Helpers --------


is_language = LANGUAGE_NAMES.__contains__
is_file_tag = ALL_FILE_TAGS.__contains__


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


_PARSED_VERSION = _parse_version(PRE_COMMIT_VERSION)


def is_le_max_version(version: str) -> bool:
    return _parse_version(version) <= _PARSED_VERSION


def _migrate_stage(stage: str) -> str:
    return _OLD_STAGES.get(stage, stage)


def migrate_stages(stages: list[str]) -> list[str]:
    return [_migrate_stage(stage) for stage in stages]


def is_subset_of_stages(potential_stages: list[str]) -> bool:
    return set(migrate_stages(potential_stages)).issubset(VALID_STAGES)


def is_in_hook_types(hook_types: list[str]) -> bool:
    return set(HOOK_TYPES).issubset(hook_types)


def is_valid_regex(value: bytes | str) -> bool:
    try:
        re.compile(value)
    except re.error:
        return False
    else:
        return True


def is_sensible_top_level_regex(value: str) -> bool:
    if not is_valid_regex(value):
        return False

    if "/*" in value:
        print("This top-level field is a regex, not a glob -- matching '/*' probably isn't what you want here")

    for fwd_slash_expr in (r"[\\/]", r"[\/]", r"[/\\]"):
        if fwd_slash_expr in value:
            print(
                "pre-commit normalizes the slashes in this top-level field to forward slashes, so you "
                rf"can use / instead of {fwd_slash_expr}."
            )

    return True


def meta_entry(name: str) -> str:
    return f"{shlex.quote(sys.executable)} -m pre_commit.meta_hooks.{name}"


# endregion


# region -------- Hooks --------


class ManifestHook(spec.Model):
    minimum_pre_commit_version: Annotated[str, spec.validate(is_le_max_version)] = "0"
    id: str
    name: str
    entry: str
    language: Annotated[str, spec.validate(is_language)]
    alias: str = ""
    files: str = ""
    exclude: str = "^$"
    types: Annotated[list[str], spec.validate(is_file_tag).default(lambda: ["file"])]
    types_or: Annotated[list[str], spec.validate(is_file_tag).default(list)]
    additional_dependencies: Annotated[list[str], spec.default(list)]
    args: Annotated[list[str], spec.default(list)]
    always_run: bool = False
    fail_fast: bool = False
    pass_filenames: bool = True
    description: str = ""
    language_version: str = PRE_COMMIT_VERSION
    log_file: str = ""
    require_serial: bool = False
    stages: Annotated[list[str] | None, spec.validate(is_subset_of_stages).hook(migrate_stages)] = None
    verbose: bool = False


_lang_meta_constraint = "system"


class BaseMetaHook(ManifestHook):
    language: Annotated[str, spec.validate(_lang_meta_constraint.__eq__)] = _lang_meta_constraint


_name_cha_meta_constraint = "check-hooks-apply"
_entry_cha_meta_constraint = meta_entry("check_hooks_apply")


class CheckHooksApplyMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_name_cha_meta_constraint.__eq__)] = _name_cha_meta_constraint
    files: Annotated[str, spec.validate(CONFIG_FILE_REGEX.__eq__)] = CONFIG_FILE_REGEX
    entry: Annotated[str, spec.validate(_entry_cha_meta_constraint.__eq__)] = _entry_cha_meta_constraint


_name_cue_meta_constraint = "check-useless-excludes"
_entry_cue_meta_constraint = meta_entry("check_useless_excludes")


class CheckUselessExcludesMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_name_cue_meta_constraint.__eq__)] = _name_cue_meta_constraint
    files: Annotated[str, spec.validate(CONFIG_FILE_REGEX.__eq__)] = CONFIG_FILE_REGEX
    entry: Annotated[str, spec.validate(_entry_cue_meta_constraint.__eq__)] = _entry_cue_meta_constraint


_name_id_meta_constraint = "identity"
_entry_id_meta_constraint = meta_entry("identity")


class IdentityMetaHook(BaseMetaHook):
    name: Annotated[str, spec.validate(_name_id_meta_constraint.__eq__)] = _name_id_meta_constraint
    verbose: bool = True
    entry: Annotated[str, spec.validate(_entry_id_meta_constraint.__eq__)] = _entry_id_meta_constraint


class MetaHook(spec.TransparentModel[CheckHooksApplyMetaHook | CheckUselessExcludesMetaHook | IdentityMetaHook]):
    pass


class ConfigHook(ManifestHook):
    # NOTE: Excluding id, stages, files, and exclude, copy and keep in sync everything from ManifestHook but optional
    # and without defaults? Breaks Liskov, hence the pyright: ignores, but this isn't meant to be normal class
    # inheritance anyway. Not sure how else to transfer annotations/attributes in a way a type-checker will understand.
    # TODO: Examine Transparent for hints at a better way?
    minimum_pre_commit_version: Annotated[str | None, spec.validate(is_le_max_version)] = None  # pyright: ignore
    name: str | None = None  # pyright: ignore
    entry: str | None = None  # pyright: ignore
    language: Annotated[str | None, spec.validate(is_language)] = None  # pyright: ignore
    alias: str | None = None  # pyright: ignore
    types: Annotated[list[str] | None, spec.validate(is_file_tag)] = None  # pyright: ignore
    types_or: Annotated[list[str] | None, spec.validate(is_file_tag)] = None  # pyright: ignore
    additional_dependencies: list[str] | None = None  # pyright: ignore
    args: list[str] | None = None  # pyright: ignore
    always_run: bool | None = None  # pyright: ignore
    fail_fast: bool | None = None  # pyright: ignore
    pass_filenames: bool | None = None  # pyright: ignore
    description: str | None = None  # pyright: ignore
    language_version: str | None = None  # pyright: ignore
    log_file: str | None = None  # pyright: ignore
    require_serial: bool | None = None  # pyright: ignore
    stages: Annotated[list[str] | None, spec.validate(is_subset_of_stages).hook(migrate_stages)] = None
    verbose: bool | None = None  # pyright: ignore

    # NOTE: Ensure these match the corresponding members in ManifestHook, with the only change being the validator.
    files: Annotated[str, spec.validate(is_sensible_top_level_regex)] = ""
    exclude: Annotated[str, spec.validate(is_sensible_top_level_regex)] = "^$"


class LocalHook(ManifestHook):
    # NOTE: Ensure these match the corresponding members in ManifestHook, with the only change being the validator.
    files: Annotated[str, spec.validate(is_sensible_top_level_regex)] = ""
    exclude: Annotated[str, spec.validate(is_sensible_top_level_regex)] = "^$"


class AnyHook(spec.TransparentModel[MetaHook | ConfigHook | LocalHook]):
    pass


# endregion


def warn_extras_in_repo_config(ek: set[str], ak: set[str], dct: dict[str, Any]) -> None:
    print(f"Unexpected key(s) present on {dct['repo']}: {', '.join(ek)}")


def warn_if_mutable_rev(rev: str) -> str:
    if "." not in rev and not re.match(r"^[a-fA-F0-9]+$", rev):
        print(
            "Some field in some repo "
            "appears to be a mutable reference "
            "(moving tag / branch).  Mutable references are never "
            "updated after first install and are not supported. "
            "See https://pre-commit.com/#using-the-latest-version-for-a-repository "
            "for more details. "
            "Hint: `pre-commit autoupdate` often fixes this.",
        )
    return rev


class MetaHookRepositoryConfig(spec.Model):
    repo: Literal["meta"]
    hooks: list[MetaHook]


class LocalHookRepositoryConfig(spec.Model):
    repo: Literal["local"]
    hooks: list[LocalHook]


class ConfigHookRepositoryConfig(spec.Model):
    repo: str
    hooks: list[ConfigHook]
    rev: Annotated[str | None, spec.hook(warn_if_mutable_rev)] = None


class RepositoryConfig(
    spec.TransparentModel[MetaHookRepositoryConfig | LocalHookRepositoryConfig | ConfigHookRepositoryConfig]
):
    # TODO: Find a way to restrict the specific type of AnyHook based on the value of repo. Maybe __post_init__ would
    # work, even if that can't stop the validation early?
    # TODO: Restrict rev as well to only be present if not a local or meta hook, and otherwise be absent.
    pass


class DefaultLanguageVersion(spec.Model, allow_extras=False):
    # NOTE: All the language names would have to all be listed out, which isn't as nice as the cfgv setup for working
    # with one source of truth. Not sure how to get around that in a way that type-checkers can understand, not without
    # doing metaclass shenanigans (see David Beazeley's sly for inspiration?).
    #
    # Currently only listing out the sample names in LANGUAGE_NAMES.

    conda: str = PRE_COMMIT_VERSION
    dotnet: str = PRE_COMMIT_VERSION
    python: str = PRE_COMMIT_VERSION
    ruby: str = PRE_COMMIT_VERSION
    system: str = PRE_COMMIT_VERSION


def warn_extras_in_config_root(ek: set[str], ak: set[str], dct: dict[str, Any]) -> None:
    print(f"Unexpected key(s) present at root: {', '.join(ek)}")


class ConfigSchema(spec.Model, allow_extras=warn_extras_in_config_root):
    minimum_pre_commit_version: Annotated[str, spec.validate(is_le_max_version)] = "0"
    repos: list[RepositoryConfig]
    default_install_hook_types: Annotated[list[str], spec.validate(is_in_hook_types).default(lambda: ["pre-commit"])]
    default_language_version: DefaultLanguageVersion
    default_stages: Annotated[list[str], spec.default(lambda: list(VALID_STAGES))]
    files: Annotated[str, spec.validate(is_valid_regex)] = ""
    exclude: Annotated[str, spec.validate(is_valid_regex)] = "^$"
    fail_fast: bool = False
    ci: dict[str, Any] | None = None


# region -------- Tests --------


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


@pytest.mark.parametrize(
    ("manifest_obj", "expected_stages"),
    [
        (
            {
                "id": "fake-hook",
                "name": "fake-hook",
                "entry": "fake-hook",
                "language": "system",
            },
            None,
        ),
        (
            {
                "id": "fake-hook",
                "name": "fake-hook",
                "entry": "fake-hook",
                "language": "system",
                "stages": ["commit-msg", "push", "commit", "merge-commit"],
            },
            ["commit-msg", "pre-push", "pre-commit", "pre-merge-commit"],
        ),
    ],
)
def test_manifest_hook(manifest_obj: dict[str, Any], expected_stages: object) -> None:
    processed = ManifestHook(manifest_obj)

    from pprint import pprint

    pprint(processed.to_dict())

    # FIXME: Case 1; processed.stages is False for some reason.
    assert processed.stages == expected_stages


# endregion
