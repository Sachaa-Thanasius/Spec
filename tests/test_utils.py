import pytest

from spec import Item, generate_type_repr_from_data, prettify_type


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        (Item(_typ=int, _key=""), "int"),
        (
            Item(_typ=list, _key="", _internal_items=[Item(_typ=int, _key="")._to_internal()]),  # pyright: ignore [reportUnknownArgumentType]
            "list[int]",
        ),
    ],
)
def test_prettify_type(input_value: Item, expected_result: str) -> None:
    assert prettify_type(input_value._to_internal()) == expected_result


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
    assert generate_type_repr_from_data(input_value) == expected_result
