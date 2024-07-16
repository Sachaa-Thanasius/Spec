# ruff: noqa: UP007
import datetime
import enum
import typing

import pytest

from spec import Item, _generate_data_type_repr, _prettify_item_type
from spec._helpers import resolve_annotation


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        (Item(typ=int, key=""), "int"),
        (
            Item(typ=list, key="", internal_items=[Item(typ=int, key="")._to_internal()]),  # pyright: ignore [reportUnknownArgumentType]
            "list[int]",
        ),
    ],
)
def test_prettify_type(input_value: Item, expected_result: str) -> None:
    assert _prettify_item_type(input_value._to_internal()) == expected_result


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        (1, "int"),
        ([1], "list[int]"),
        ([1, ""], "list[int | str]"),
        ({1: 1}, "dict[int, int]"),
        ({"a": 1, "b": "c"}, "dict[str, int | str]"),
    ],
)
def test_generate_type_repr_from_data(input_value: object, expected_result: str) -> None:
    assert _generate_data_type_repr(input_value) == expected_result


# The tests for resolve_annotation are modified from discord.py, published at https://github.com/Rapptz/discord.py/blob/v2.4.0/tests/test_utils.py#L190-L273.
# It was made available under MIT and is copyright 2015-present Rapptz.
# The license in its original form may be found at https://github.com/Rapptz/discord.py/blob/v2.4.0/LICENSE
# and is also included in this repository's `LICENSE` file.
@pytest.mark.parametrize(
    ("annotation", "resolved"),
    [
        (datetime.datetime, datetime.datetime),
        ("datetime.datetime", datetime.datetime),
        (
            'typing.Union[typing.Literal["a"], typing.Literal["b"]]',
            typing.Union[typing.Literal["a"], typing.Literal["b"]],  # noqa: PYI030
        ),
        (
            "typing.Union[typing.Union[int, str], typing.Union[bool, dict]]",
            typing.Union[int, str, bool, dict],  # pyright: ignore [reportMissingTypeArgument]
        ),
    ],
)
def test_resolve_annotation(annotation: typing.Any, resolved: typing.Any) -> None:
    assert resolved == resolve_annotation(annotation, globals(), locals(), None)


@pytest.mark.parametrize(
    ("annotation", "resolved", "check_cache"),
    [
        (datetime.datetime, datetime.datetime, False),
        ("datetime.datetime", datetime.datetime, True),
        (
            'typing.Union[typing.Literal["a"], typing.Literal["b"]]',
            typing.Union[typing.Literal["a"], typing.Literal["b"]],  # noqa: PYI030
            True,
        ),
        (
            "typing.Union[typing.Union[int, str], typing.Union[bool, dict]]",
            typing.Union[int, str, bool, dict],  # pyright: ignore [reportMissingTypeArgument]
            True,
        ),
    ],
)
def test_resolve_annotation_with_cache(annotation: typing.Any, resolved: typing.Any, check_cache: bool) -> None:
    cache: dict[str, typing.Any] = {}

    assert resolved == resolve_annotation(annotation, globals(), locals(), cache)

    if check_cache:
        assert len(cache) == 1

        cached_item = cache[annotation]

        latest = resolve_annotation(annotation, globals(), locals(), cache)

        assert latest is cached_item
        assert typing.get_origin(latest) is typing.get_origin(resolved)
    else:
        assert len(cache) == 0


def test_resolve_annotation_optional_normalisation() -> None:
    value = resolve_annotation("typing.Union[None, int]", globals(), locals(), None)
    assert value.__args__ == (int, type(None))


@pytest.mark.parametrize(
    ("annotation", "resolved"),
    [
        ("int | None", typing.Optional[int]),
        (int | None, typing.Optional[int]),
        ("str | int", typing.Union[str, int]),
        (str | int, typing.Union[str, int]),
        ("str | int | None", typing.Optional[typing.Union[str, int]]),
        (str | int | None, typing.Optional[typing.Union[str, int]]),
    ],
)
def test_resolve_annotation_310(annotation: typing.Any, resolved: typing.Any) -> None:
    assert resolved == resolve_annotation(annotation, globals(), locals(), None)


@pytest.mark.parametrize(
    ("annotation", "resolved"),
    [
        ("int | None", typing.Optional[int]),
        ("str | int", typing.Union[str, int]),
        ("str | int | None", typing.Optional[typing.Union[str, int]]),
    ],
)
def test_resolve_annotation_with_cache_310(annotation: typing.Any, resolved: typing.Any) -> None:
    cache: dict[str, typing.Any] = {}

    assert resolved == resolve_annotation(annotation, globals(), locals(), cache)
    assert typing.get_origin(resolved) is typing.Union

    assert len(cache) == 1

    cached_item = cache[annotation]

    latest = resolve_annotation(annotation, globals(), locals(), cache)
    assert latest is cached_item
    assert typing.get_origin(latest) is typing.get_origin(resolved)


@pytest.mark.parametrize(
    ("annotation", "resolved"),
    [
        (None, type(None)),
        (typing.NewType("_MyInt", int), int),
    ],
)
def test_resolve_annotation_various(annotation: typing.Any, resolved: typing.Any) -> None:
    assert resolved == resolve_annotation(annotation, globals(), locals())


class _Color(enum.Enum):
    RED = 1


@pytest.mark.parametrize(
    ("annotation", "resolved"),
    [
        (
            "typing.Literal[None, 2, True, 'hello', b'world', _Color.RED]",
            typing.Literal[None, 2, True, "hello", b"world", _Color.RED],
        ),
    ],
)
def test_resolve_annotation_valid_literal(annotation: typing.Any, resolved: typing.Any) -> None:
    assert resolved == resolve_annotation(annotation, globals(), locals())


@pytest.mark.parametrize(
    "annotation",
    [
        typing.Literal[object()],  # type: ignore
        typing.Literal[type],  # type: ignore
        typing.Literal[_Color],  # type: ignore
        typing.Literal[range(10)],  # type: ignore
    ],
)
def test_resolve_annotation_invalid_literal(annotation: typing.Any) -> None:
    with pytest.raises(TypeError) as exc_info:
        resolve_annotation(annotation, globals(), locals())

    assert exc_info.value.args[0] == "Literal arguments must be of type str, bytes, int, bool, Enum, or NoneType."
