import pytest

from spec import Item, _generate_type_repr_from_data, _prettify_type


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
    assert _prettify_type(input_value._to_internal()) == expected_result


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
    assert _generate_type_repr_from_data(input_value) == expected_result
