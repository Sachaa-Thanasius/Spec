from spec.item import Item
from spec.util import generate_type_from_data, pretty_type


def test_pretty_type() -> None:
    assert pretty_type(Item(_ty=int, _key="")._to_internal()) == "int"
    assert pretty_type(Item(_ty=list, _key="", _internal_items=[Item(_ty=int, _key="")._to_internal()])._to_internal()) == "list[int]"

def test_generate_type_from_data() -> None:
    assert generate_type_from_data(1) == "int"
    assert generate_type_from_data([1]) == "list[int]"
    assert generate_type_from_data([1, ""]) == "list[int | str]"
    assert generate_type_from_data({1: 1}) == "dict[int, int]"
    assert generate_type_from_data({"a": 1, "b": "c"}) == "dict[str, int | str]"
