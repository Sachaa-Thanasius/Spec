import pytest

from spec import Item, _generate_type_repr_from_data, _prettify_type, _UniqueList


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([1, 2, 3, 4, 5, 5, 6, 6], id="integers"),
        pytest.param(["hello", "world", "hello2", "hello", "world"], id="strings"),
        pytest.param([int, str, float, tuple, str, int], id="builtin types"),
    ],
)
def test_unique_list(items: list[object]) -> None:
    unique_list = _UniqueList()

    for item in items:
        unique_list.append(item)

    guaranteed_unique = set(items)

    assert len(unique_list) == len(guaranteed_unique)
    assert set(unique_list) == guaranteed_unique


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
