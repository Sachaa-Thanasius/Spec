from typing import Annotated

import pytest

import spec


class Simple(spec.Model):
    a: str
    b: float


def test_valid_instantiation() -> None:
    input_value = {"a": "hello", "b": 1.0}
    Simple(input_value)
    Simple(**input_value)


def test_invalid_instantiation() -> None:
    with pytest.raises(spec.errors.MissingArgument):
        Simple()


# ================================
class Simple1(spec.Model):
    a: int
    b: str


class Simple2(spec.Model):
    c: int


simple_raw_sample = {"a": 1, "b": "value"}
simple_model_inst = Simple1(simple_raw_sample)


def test_simple_values() -> None:
    assert simple_model_inst.a == 1
    assert simple_model_inst.b == "value"


def test_simple_eq() -> None:
    assert simple_model_inst == Simple1(simple_raw_sample)
    assert simple_model_inst != Simple2({"c": 2})


# ================================
class Inner(spec.Model):
    value: int


class Outer(spec.Model):
    inner: Inner


nested_raw_sample = {"inner": {"value": 1}}
nested_model_inst = Outer(nested_raw_sample)


def test_nested_inner_value() -> None:
    assert nested_model_inst.inner == Inner(nested_raw_sample["inner"])


def test_nested_inner_value_value() -> None:
    assert nested_model_inst.inner.value == 1


# ================================
class List(spec.Model):
    data: list[int]


list_raw_sample = {"data": [1, 2, 3]}
list_model_inst = List(list_raw_sample)


def test_list_data_value() -> None:
    assert list_model_inst.data == [1, 2, 3]


# ================================
class ListOuter(spec.Model):
    inners: list[Inner]


nested_list_raw_sample = {"inners": [{"value": 1}, {"value": 2}]}
nested_list_model_inst = ListOuter(nested_list_raw_sample)


def test_nested_list_inner_value() -> None:
    assert nested_list_model_inst.inners == [Inner(data) for data in nested_list_raw_sample["inners"]]


def test_nested_list_inner_value_value() -> None:
    for inner, data in zip(nested_list_model_inst.inners, nested_list_raw_sample["inners"], strict=False):
        assert inner.value == data["value"]


# ================================
class Dict(spec.Model):
    data: dict[str, int]


dict_raw_sample = {"data": {"a": 1, "b": 2}}
dict_model_inst = Dict(dict_raw_sample)


def test_dict_data_value() -> None:
    assert dict_model_inst.data == dict_raw_sample["data"]


# ================================
class AnnotatedUsage(spec.Model):
    a: Annotated[int, spec.rename("b")]


annotated_raw_sample = {"b": 1}
annotated_model_inst = AnnotatedUsage(annotated_raw_sample)


def test_annotated_data_value() -> None:
    assert annotated_model_inst.a == annotated_raw_sample["b"]


# ================================
class DefaultUsage(spec.Model):
    a: Annotated[int, spec.default(lambda: 0)]


default_model_inst = DefaultUsage({})


def test_default_data_value() -> None:
    assert default_model_inst.a == 0


# ================================
class OptionalModel(spec.Model):
    value: int | None


optional_not_passed_model_inst = OptionalModel({})
optional_passed_none_model_inst = OptionalModel({"value": None})
optional_passed_value_model_inst = OptionalModel({"value": 1})


# ================================
class AnnotatedOptional(spec.Model):
    value: Annotated[int | None, spec.rename("data")]


annotated_optional_raw_sample = {"data": 1}
annotated_optional_model_inst = AnnotatedOptional({"data": 1})


# ================================
class AnnotatedOptionalWithDefault(spec.Model):
    value: Annotated[int | None, spec.default(lambda: 0)]


annotated_optional_with_default_raw_sample: dict[str, object] = {}
annotated_optional_with_default_model_inst = AnnotatedOptionalWithDefault(annotated_optional_with_default_raw_sample)


# ================================
class DefaultClsAttr(spec.Model):
    value: int = 1


default_class_attr_model_inst = DefaultClsAttr({})


# ================================
class PartA(spec.Model):
    a: int


class PartB(spec.Model):
    b: str


class UntaggedPart(spec.TransparentModel[PartA | PartB]):
    pass


ExternallyTaggedPart: type[spec.TransparentModel[PartA | PartB]] = spec.transparent(
    Annotated[PartA | PartB, spec.tag("external")]
)

AdjacentlyTaggedPart: type[spec.TransparentModel[PartA | PartB]] = spec.transparent(
    Annotated[PartA | PartB, spec.tag("adjacent", tag="type", content="value")]
)

InternallyTaggedPart: type[spec.TransparentModel[PartA | PartB]] = spec.transparent(
    PartA | PartB, spec.tag("internal", tag="type")
)


# fmt: off
untagged_part_raw_sample_a          = {"a": 1}
untagged_part_raw_sample_b          = {"b": "data"}
untagged_part_model_inst_a          = UntaggedPart(untagged_part_raw_sample_a)
untagged_part_model_inst_b          = UntaggedPart(untagged_part_raw_sample_b)

tagged_part_raw_sample_a            = {"PartA": {"a": 1}}
tagged_part_raw_sample_b            = {"PartB": {"b": "data"}}
tagged_part_model_inst_a            = ExternallyTaggedPart(tagged_part_raw_sample_a)
tagged_part_model_inst_b            = ExternallyTaggedPart(tagged_part_raw_sample_b)

adjacently_tagged_part_raw_sample_a = {"type": "PartA", "value": {"a": 1}}
adjacently_tagged_part_raw_sample_b = {"type": "PartB", "value": {"b": "data"}}
adjacently_tagged_part_model_inst_a = AdjacentlyTaggedPart(adjacently_tagged_part_raw_sample_a)
adjacently_tagged_part_model_inst_b = AdjacentlyTaggedPart(adjacently_tagged_part_raw_sample_b)

internally_tagged_part_raw_sample_a = {"type": "PartA", "a": 1}
internally_tagged_part_raw_sample_b = {"type": "PartB", "b": "data"}
internally_tagged_part_model_inst_a = InternallyTaggedPart(internally_tagged_part_raw_sample_a)
internally_tagged_part_model_inst_b = InternallyTaggedPart(internally_tagged_part_raw_sample_b)
# fmt: on


@pytest.mark.parametrize(
    ("model_instance", "value_class"),
    [
        (untagged_part_model_inst_a, PartA),
        (untagged_part_model_inst_b, PartB),
        (tagged_part_model_inst_a, PartA),
        (tagged_part_model_inst_b, PartB),
        (adjacently_tagged_part_model_inst_a, PartA),
        (adjacently_tagged_part_model_inst_b, PartB),
        (internally_tagged_part_model_inst_a, PartA),
        (internally_tagged_part_model_inst_b, PartB),
    ],
)
def test_tagged_part_value_type(
    model_instance: spec.TransparentModel[PartA | PartB],
    value_class: type[spec.Model],
) -> None:
    assert isinstance(model_instance.value, value_class)


@pytest.mark.parametrize(
    ("model_instance", "name", "expected_value"),
    [
        (untagged_part_model_inst_a, "a", 1),
        (untagged_part_model_inst_b, "b", "data"),
        (tagged_part_model_inst_a, "a", 1),
        (tagged_part_model_inst_b, "b", "data"),
        (adjacently_tagged_part_model_inst_a, "a", 1),
        (adjacently_tagged_part_model_inst_b, "b", "data"),
        (internally_tagged_part_model_inst_a, "a", 1),
        (internally_tagged_part_model_inst_b, "b", "data"),
    ],
)
def test_tagged_part_value_attr(
    model_instance: spec.TransparentModel[PartA | PartB],
    name: str,
    expected_value: object,
) -> None:
    assert getattr(model_instance.value, name) == expected_value


# ================================
@pytest.mark.parametrize(
    ("renaming_scheme", "attr_name"),
    [
        (spec.Upper, "MY_FOO_VAR"),
        (spec.CamelCase, "myFooVar"),
        (spec.PascalCase, "MyFooVar"),
        (spec.KebabCase, "my-foo-var"),
        (spec.ScreamingKebabCase, "MY-FOO-VAR"),
    ],
)
def test_renaming_schemes(renaming_scheme: type[spec.RenameScheme], attr_name: str) -> None:
    class GlobalRename(spec.Model, rename=renaming_scheme):
        my_foo_var: int

    instance = GlobalRename({attr_name: 1})

    assert instance.my_foo_var == 1

    assert instance.to_dict() == {attr_name: 1}

    with pytest.raises(spec.errors.MissingRequiredKey):
        GlobalRename({"my_foo": 2})


# ================================
class WithEquals(spec.Model):
    data: Annotated[str, spec.hook(lambda v: f"===={v}====")]


def test_hooked_value() -> None:
    value = "hello world!"
    instance = WithEquals({"data": value})
    assert instance.data == f"===={value}===="


# ================================
@pytest.mark.parametrize(
    ("model_instance", "expected_value"),
    [
        (optional_not_passed_model_inst, None),
        (optional_passed_none_model_inst, None),
        (optional_passed_value_model_inst, 1),
        (annotated_optional_model_inst, 1),
        (annotated_optional_with_default_model_inst, 0),
        (default_class_attr_model_inst, DefaultClsAttr.value),
    ],
)
def test_value_value(model_instance: spec.Model, expected_value: object) -> None:
    assert model_instance.value == expected_value  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


@pytest.mark.parametrize(
    ("model_instance", "raw_sample"),
    [
        (simple_model_inst, {"a": 1, "b": "value"}),
        (nested_model_inst, {"inner": {"value": 1}}),
        (list_model_inst, {"data": [1, 2, 3]}),
        (nested_list_model_inst, {"inners": [{"value": 1}, {"value": 2}]}),
        (dict_model_inst, {"data": {"a": 1, "b": 2}}),
        (annotated_model_inst, {"b": 1}),
        (default_model_inst, {"a": 0}),
        (optional_not_passed_model_inst, {"value": None}),
        (optional_passed_none_model_inst, {"value": None}),
        (optional_passed_value_model_inst, {"value": 1}),
        (annotated_optional_model_inst, {"data": 1}),
        (annotated_optional_with_default_model_inst, {"value": 0}),
        (untagged_part_model_inst_a, {"a": 1}),
        (untagged_part_model_inst_b, {"b": "data"}),
        (tagged_part_model_inst_a, {"PartA": {"a": 1}}),
        (tagged_part_model_inst_b, {"PartB": {"b": "data"}}),
        (adjacently_tagged_part_model_inst_a, {"type": "PartA", "value": {"a": 1}}),
        (adjacently_tagged_part_model_inst_b, {"type": "PartB", "value": {"b": "data"}}),
        (internally_tagged_part_model_inst_a, {"type": "PartA", "a": 1}),
        (internally_tagged_part_model_inst_b, {"type": "PartB", "b": "data"}),
    ],
)
def test_model_to_dict(model_instance: spec.Model, raw_sample: dict[str, object]) -> None:
    assert model_instance.to_dict() == raw_sample


@pytest.mark.parametrize(
    ("model_instance", "expected_value"),
    [
        (simple_model_inst, "<Simple1 a=1 b='value'>"),
        (nested_model_inst, "<Outer inner=<Inner value=1>>"),
        (list_model_inst, "<List data=[1, 2, 3]>"),
        (nested_list_model_inst, "<ListOuter inners=[<Inner value=1>, <Inner value=2>]>"),
        (dict_model_inst, "<Dict data={'a': 1, 'b': 2}>"),
        (annotated_model_inst, "<AnnotatedUsage a=1>"),
        (default_model_inst, "<DefaultUsage a=0>"),
        (optional_not_passed_model_inst, "<OptionalModel value=None>"),
        (optional_passed_none_model_inst, "<OptionalModel value=None>"),
        (optional_passed_value_model_inst, "<OptionalModel value=1>"),
        (annotated_optional_model_inst, "<AnnotatedOptional value=1>"),
        (annotated_optional_with_default_model_inst, "<AnnotatedOptionalWithDefault value=0>"),
        (internally_tagged_part_model_inst_a, "<PartAOrPartB <PartA a=1>>"),
        (internally_tagged_part_model_inst_b, "<PartAOrPartB <PartB b='data'>>"),
    ],
)
def test_model_repr(model_instance: spec.Model, expected_value: str) -> None:
    assert repr(model_instance) == expected_value


# ================================
@pytest.mark.parametrize(
    ("model_class", "payload"),
    [
        (Simple2, {"c": "not a string"}),
        (List, {"data": [1, 2, "3"]}),
    ],
)
def test_invalid_type(model_class: type[spec.Model], payload: dict[str, object]) -> None:
    with pytest.raises(spec.errors.InvalidType):
        model_class(payload)


def test_invalid_tag() -> None:
    with pytest.raises(spec.errors.MissingTypeName):
        spec.transparent(int | str, spec.tag("external"))


class ValueValidator(spec.Model):
    x: Annotated[int, spec.validate(range(10).__contains__)]


def test_invalid_value() -> None:
    with pytest.raises(spec.errors.FailedValidation):
        ValueValidator(x=100)


def test_valid_value() -> None:
    ValueValidator(x=5)
