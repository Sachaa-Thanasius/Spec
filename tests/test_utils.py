import pytest

from spec.item import Item
from spec.util import generate_type_from_data, pretty_type


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        (Item(_ty=int, _key=""), "int"),
        (
            Item(_ty=list, _key="", _internal_items=[Item(_ty=int, _key="")._to_internal()]),  # pyright: ignore [reportUnknownArgumentType]
            "list[int]",
        ),
    ],
)
def test_pretty_type(input_value: Item, expected_result: str) -> None:
    assert pretty_type(input_value._to_internal()) == expected_result


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
def test_generate_type_from_data(input_value: object, expected_result: str) -> None:
    assert generate_type_from_data(input_value) == expected_result
