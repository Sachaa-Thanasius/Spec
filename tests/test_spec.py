from typing import Annotated

import pytest

import spec


def test_exception_notes() -> None:
    class Inner(spec.Model):
        c: int

    class Middle(spec.Model):
        b: Inner

    class Outer(spec.Model):
        a: Middle

    invalid_sample = {"a": {"b": {"c": "1"}}}

    with pytest.raises(spec.InvalidType) as exc_info:
        Outer(invalid_sample)

    assert exc_info.value.__notes__ == ["--------At Inner.c", "----At Middle.b", "At Outer.a"]


class Simple(spec.Model):
    a: str
    b: float


def test_valid_instantiation() -> None:
    input_value = {"a": "hello", "b": 1.0}
    Simple(input_value)
    Simple(**input_value)


def test_invalid_instantiation_args() -> None:
    with pytest.raises(spec.MissingArgument) as exc_info:
        Simple()

    assert exc_info.value.args[0] == "No data or kwargs passed to Model."

    with pytest.raises(ValueError) as exc_info:  # noqa: PT011
        Simple({"a": "hello"}, b=1.0)  # pyright: ignore [reportCallIssue]

    assert exc_info.value.args[0] == "Only data or kwargs is accepted, not both."


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
    value: int | None = None


optional_not_passed_model_inst = OptionalModel({})
optional_passed_none_model_inst = OptionalModel({"value": None})
optional_passed_value_model_inst = OptionalModel({"value": 1})


# ================================
class AnnotatedOptional(spec.Model):
    value: Annotated[int | None, spec.rename("data")]


annotated_optional_model_inst = AnnotatedOptional({"data": 1})


# ================================
class AnnotatedOptionalWithDefault(spec.Model):
    value: Annotated[int | None, spec.default(lambda: 0)]


annotated_optional_with_default_model_inst = AnnotatedOptionalWithDefault({})


# ================================
class NoDefaultModel(spec.Model):
    val: int | spec.NoDefault


def test_no_default_properties() -> None:
    NoDefaultModel({})
    NoDefaultModel({"val": 1})
    NoDefaultModel({"val": spec.NODEFAULT})

    with pytest.raises(spec.InvalidType):
        NoDefaultModel({"val": None})


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

externally_tagged_part_raw_sample_a = {"PartA": {"a": 1}}
externally_tagged_part_raw_sample_b = {"PartB": {"b": "data"}}
externally_tagged_part_model_inst_a = ExternallyTaggedPart(externally_tagged_part_raw_sample_a)
externally_tagged_part_model_inst_b = ExternallyTaggedPart(externally_tagged_part_raw_sample_b)

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
        (externally_tagged_part_model_inst_a, PartA),
        (externally_tagged_part_model_inst_b, PartB),
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
        (externally_tagged_part_model_inst_a, "a", 1),
        (externally_tagged_part_model_inst_b, "b", "data"),
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

renaming_cases = pytest.mark.parametrize(
    ("renaming_scheme", "attr_name", "internal_attr_name"),
    [
        (spec.Upper, "MY", "my"),
        (spec.PascalCase, "My", "my"),
        (spec.ScreamingKebabCase, "MY", "my"),
        (spec.Upper, "MY_FOO_VAR", "my_foo_var"),
        (spec.CamelCase, "myFooVar", "my_foo_var"),
        (spec.PascalCase, "MyFooVar", "my_foo_var"),
        (spec.KebabCase, "my-foo-var", "my_foo_var"),
        (spec.ScreamingKebabCase, "MY-FOO-VAR", "my_foo_var"),
    ],
)


@renaming_cases
def test_renaming_schemes(renaming_scheme: type[spec.RenameScheme], attr_name: str, internal_attr_name: str) -> None:
    GlobalRename = type(
        "GlobalRename",
        (spec.Model,),
        {"__annotations__": {internal_attr_name: int}},
        rename=renaming_scheme,
    )

    instance = GlobalRename({attr_name: 1})
    assert instance.to_dict() == {attr_name: 1}
    assert getattr(instance, internal_attr_name) == 1


@renaming_cases
def test_invalid_input_when_rename_active(
    renaming_scheme: type[spec.RenameScheme],
    attr_name: str,
    internal_attr_name: str,
) -> None:
    GlobalRename = type(
        "GlobalRename",
        (spec.Model,),
        {"__annotations__": {internal_attr_name: int}},
        rename=renaming_scheme,
    )

    with pytest.raises(spec.MissingRequiredKey) as exc_info:
        GlobalRename({internal_attr_name: 2})

    assert exc_info.value.args[0] == f"Missing required key 'GlobalRename.{attr_name}'."


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
        (externally_tagged_part_model_inst_a, {"PartA": {"a": 1}}),
        (externally_tagged_part_model_inst_b, {"PartB": {"b": "data"}}),
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
    ("model_class", "payload", "expected_error_msg"),
    [
        (Simple2, {"c": "a string"}, "'Simple2.c' expected type 'int' but found 'str'."),
        (List, {"data": [1, 2, "3"]}, "'List.data' expected type 'list[int]' but found 'list[int | str]'."),
    ],
)
def test_invalid_type(model_class: type[spec.Model], payload: dict[str, object], expected_error_msg: str) -> None:
    with pytest.raises(spec.InvalidType) as exc_info:
        model_class(payload)

    assert exc_info.value.args[0] == expected_error_msg


def test_invalid_tag() -> None:
    with pytest.raises(spec.MissingTypeName) as exc_info:
        spec.transparent(int | str, spec.tag("external"))

    assert exc_info.value.args[0] == "'intOrstr.value' union type is missing a type name for '<class 'int'>'."


class ValueValidator(spec.Model):
    x: Annotated[int, spec.validate(range(10).__contains__)]


def test_passing_validator() -> None:
    assert ValueValidator(x=5)


def test_failing_validator() -> None:
    with pytest.raises(spec.FailedValidationHook) as exc_info:
        ValueValidator(x=100)

    assert exc_info.value.args[0] == "'ValueValidator.x' failed validation."


def test_denied_extra_keys() -> None:
    class NoExtra(spec.Model, allow_extras=False):
        thing: str

    with pytest.raises(spec.NoExtraKeysAllowed) as exc_info:
        NoExtra({"thing": "hello", "thing2": "world"})

    assert exc_info.value.args[0] == "No extra keys are allowed in this model. Extra(s) found: {'thing2'}."
