# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius <i.like.ike101@gmail.com>
#
# SPDX-License-Identifier: MIT

# TODO: Check if type_name, tag_info, and rename already cover parts of tagging I thought were missing. Also add tests.
# TODO: Add handling for NODEFAULT, NoDefault (maybe Literal).
# TODO: Add an "omit_defaults" parameter to "Model.to_dict()".
# TODO: Consider imitating some of msgspec's, TypedDict's, and cfgv's APIs and error messages.

import enum
import sys
import types
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin as tp_get_origin,
    overload,
)

from ._typing_helpers import resolve_annotation

if sys.version_info >= (3, 13):  # pragma: >=3.13 cover
    from typing import TypeVar
else:  # pragma: <3.13 cover
    from typing_extensions import TypeVar

if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from types import get_original_bases
else:  # pragma: <3.12 cover
    from typing_extensions import get_original_bases


__all__ = (
    # Exceptions
    "SpecError",
    "MissingArgument",
    "MissingRequiredKey",
    "InvalidType",
    "FailedValidationHook",
    "UnknownUnionKey",
    "MissingTypeName",
    "NoExtraKeysAllowed",
    # Utilities
    "NoDefault",
    "NODEFAULT",
    # Item
    "Item",
    "rename",
    "default",
    "validate",
    "hook",
    "tag",
    "type_name",
    # Renaming schemes
    "to_default",
    "to_upper",
    "to_camel_case",
    "to_pascal_case",
    "to_kebab_case",
    "to_screaming_kebab_case",
    # Model
    "is_model",
    "Model",
    "TransparentModel",
    "transparent",
)

_T = TypeVar("_T")
_T_def = TypeVar("_T_def", default=Any)


# region Exceptions
# ----------------------------------------------------------------------------------------------------------------------


class SpecError(Exception):
    """Base exception class for spec."""

    def __init__(self, model_name: str, message: str):
        super().__init__(message)
        self.model_name = model_name
        self.message = message


class MissingArgument(SpecError):
    """Exception that's raised when a Model is instantiated with no arguments."""

    def __init__(self, model_name: str):
        super().__init__(model_name, "No data or kwargs passed to Model.")


class MissingRequiredKey(SpecError):
    """Exception that's raised when a Model is instantiated without a required key."""

    def __init__(self, model_name: str, key: str):
        super().__init__(model_name, f"Missing required key '{model_name}.{key}'.")


class InvalidType(SpecError):
    """Exception that's raised when the type for a given value doesn't match the expected type from the Model."""

    def __init__(self, model_name: str, key: str, expected: str, typ: str):
        super().__init__(model_name, f"'{model_name}.{key}' expected type '{expected}' but found '{typ}'.")
        self.key = key
        self.expected = expected
        self.typ = typ

    @classmethod
    def from_expected(
        cls,
        model: "Model",
        item: "_InternalItem",
        root_item: "_InternalItem",
        root_value: object,
    ) -> Self:
        return cls(type(model).__name__, item.key, _prettify_item_type(root_item), _generate_data_type_repr(root_value))


class FailedValidationHook(SpecError):
    """Exception that's raised when a validation hook for an item in a model fails."""

    def __init__(self, model_name: str, key: str):
        super().__init__(model_name, f"'{model_name}.{key}' failed validation.")
        self.key = key


class UnknownUnionKey(SpecError):
    """Exception that's raised when the provided tag for a tagged union model doesn't work to find any sub-models."""

    def __init__(self, model_name: str, key: str):
        super().__init__(model_name, f"Unknown key found `{key}`.")
        self.key = key


class MissingTypeName(SpecError):
    """Exception that's raised when a tagged union model is missing a type name."""

    def __init__(self, model_name: str, key: str, typ: str):
        super().__init__(model_name, f"'{model_name}.{key}' union type is missing a type name for '{typ}'.")
        self.key = key
        self.typ = typ


class NoExtraKeysAllowed(SpecError):
    """Exception that's raised when a model prohibits extra keys, but they're provided anyway."""

    def __init__(self, model_name: str, extra_keys: Iterable[str], allowed_keys: Iterable[str]):
        super().__init__(model_name, f"No extra keys are allowed in this model. Extra(s) found: {extra_keys}.")
        self.extra_keys = extra_keys
        self.allowed_keys = allowed_keys


# endregion


# region Utilities
# ----------------------------------------------------------------------------------------------------------------------


@contextmanager
def add_note_to_exception(note: str) -> Generator[None]:
    """Add a note to an captured exception, then let it continue propogating."""

    try:
        yield
    except Exception as exc:
        exc.add_note(note)
        raise


@contextmanager
def format_exception_notes() -> Generator[None]:
    """Make the notes of a captured exception slightly more graph-like in presentation."""

    try:
        yield
    except Exception as exc:
        if hasattr(exc, "__notes__"):
            exc.__notes__[:-1] = [f"----{note}" for note in exc.__notes__[:-1]]
        raise


class Missing(enum.Enum):
    MISSING = 0

    __repr__ = enum.global_enum_repr  # pyright: ignore [reportAssignmentType]


MISSING = Missing.MISSING
"""Singleton sentinel for internal use."""


class NoDefault(enum.Enum):
    """Type of spec.NODEFAULT singleton. Meant to be used in annotations."""

    NODEFAULT = 0

    __repr__ = enum.global_enum_repr  # pyright: ignore [reportAssignmentType]


NODEFAULT = NoDefault.NODEFAULT
"""Singleton sentinel that indicates no default value exists for a model member."""


def _get_type_name(typ: type) -> str:
    return getattr(typ, "_spec_model_type_name", typ.__name__)


def get_origin(typ: Any) -> Any:
    """Get the unsubscripted version of a type, or just the type if it doesn't support subscripting or wasn't
    subscripted.

    See typing.get_origin for more details about the base functionality.

    Examples
    --------
    Default functionality:
    >>> assert get_origin(list[str]) is list
    >>> assert get_origin(set[str]) is set

    Differing functionality from typing.get_origin:
    >>> assert get_origin(list) is list
    >>> assert get_origin(set) is set
    >>> assert get_origin(tuple) is tuple
    >>> assert get_origin(dict) is dict
    """

    return tp_get_origin(typ) or typ


def _prettify_item_type(item: "_InternalItem") -> str:
    # Reminder: item.typ is only a list when tags are involved.
    if not isinstance(item.typ, list) and (generics := item.internal_items):
        generic_str = f"[{', '.join([_prettify_item_type(generic) for generic in generics])}]"
    else:
        generic_str = ""

    if not isinstance(item.typ, list):  # pyright: ignore [reportUnknownMemberType]
        return f"{getattr(item.typ, '__name__', type(item.typ).__name__)}{generic_str}"
    else:  # noqa: RET505 # Readability.
        typ_names = [getattr(typ.typ, "__name__", type(typ.typ).__name__) for typ in item.typ]  # pyright: ignore
        return f"{_repr_as_union(typ_names)}{generic_str}"


def _repr_as_union(type_reprs: Iterable[str]) -> str:
    return " | ".join(type_reprs or ["Unknown"])


def _generate_data_type_repr(data: object) -> str:
    # Ensure uniqueness and conserve order.
    match data:
        case list() | set() | tuple():
            unique_types = tuple(dict.fromkeys(map(_generate_data_type_repr, data)))  # pyright: ignore [reportUnknownArgumentType]
            generics = (_repr_as_union(unique_types),)

        case dict():
            key_types = tuple(dict.fromkeys(map(_generate_data_type_repr, data.keys())))  # pyright: ignore [reportUnknownArgumentType]
            value_types = tuple(dict.fromkeys(map(_generate_data_type_repr, data.values())))  # pyright: ignore [reportUnknownArgumentType]

            generics = (_repr_as_union(key_types), _repr_as_union(value_types))
        case _:
            generics = ()

    generic_str = f"[{', '.join(generics)}]" if generics else ""

    return f"{type(data).__name__}{generic_str}"  # pyright: ignore [reportUnknownArgumentType]


# endregion


# region Items
# ----------------------------------------------------------------------------------------------------------------------


class _InternalItem(Generic[_T_def]):
    __slots__ = (
        "key",
        "rename",
        "typ",
        "internal_items",
        "default",
        "validate",
        "hook",
        "tag",
        "tag_info",
        "type_name",
    )

    def __init__(
        self,
        key: str,
        rename: str | None,
        typ: type[_T_def],
        internal_items: list["_InternalItem"],
        default: Callable[[], _T_def] | None,
        validate: Callable[[_T_def], bool],
        hook: Callable[[_T_def], _T_def],
        tag: Literal["untagged", "external", "internal", "adjacent"],
        tag_info: dict[str, Any],
        type_name: str | None,
    ):
        self.key: str = key
        self.rename: str | None = rename
        self.typ: type[_T_def] = typ
        self.internal_items: list[_InternalItem] = internal_items
        self.default: Callable[[], _T_def] | None = default
        self.validate: Callable[[_T_def], bool] = validate
        self.hook: Callable[[_T_def], _T_def] = hook
        self.tag: Literal["untagged", "external", "internal", "adjacent"] = tag
        self.tag_info: dict[str, Any] = tag_info
        self.type_name: str | None = type_name

    @property
    def actual_key(self) -> str:
        return self.rename or self.key

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in type(self).__slots__)})"


class Item(Generic[_T_def]):
    """Intermediate representation of a model member which accumulates state information.

    Includes info like the member's type, default value, validation hook, post-validation hook, etc.
    """

    __slots__ = (
        "_key",
        "_rename",
        "_typ",
        "_internal_items",
        "_default",
        "_modified",
        "_validate",
        "_hook",
        "_tag",
        "_tag_info",
        "_type_name",
    )

    def __init__(
        self,
        key: str | None = None,
        rename: str | None = None,
        typ: type[_T_def] | None = None,
        internal_items: list[_InternalItem] | None = None,
        default: Callable[[], _T_def] | None = None,
        modified: list[str] | Literal[MISSING] = MISSING,
        validate: Callable[[_T_def], bool] = lambda _: True,
        hook: Callable[[_T_def], _T_def] = lambda x: x,
        tag: Literal["untagged", "external", "internal", "adjacent"] = "untagged",
        tag_info: dict[str, Any] | Literal[MISSING] = MISSING,
        type_name: str | None = None,
    ):
        self._key: str | None = key
        self._rename: str | None = rename
        self._typ: type[_T_def] | None = typ
        self._internal_items: list[_InternalItem] | None = internal_items
        self._default: Callable[[], _T_def] | None = default
        self._modified: list[str] = modified if modified is not MISSING else []
        self._validate: Callable[[_T_def], bool] = validate
        self._hook: Callable[[_T_def], _T_def] = hook
        self._tag: Literal["untagged", "external", "internal", "adjacent"] = tag
        self._tag_info: dict[str, Any] = tag_info if tag_info is not MISSING else {}
        self._type_name: str | None = type_name

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in type(self).__slots__)})"

    def _to_internal(self) -> _InternalItem[_T_def]:
        assert self._key is not None
        assert self._typ is not None

        return _InternalItem(
            self._key,
            self._rename,
            self._typ,
            self._internal_items or [],
            self._default,
            self._validate,
            self._hook,
            self._tag,
            self._tag_info,
            self._type_name,
        )

    def rename(self, key: str) -> Self:
        self._rename = key
        self._modified.append("_rename")

        return self

    def default(self, default: Callable[[], _T_def]) -> Self:
        self._default = default
        self._modified.append("_default")

        return self

    def validate(self, validator: Callable[[_T_def], bool]) -> Self:
        self._validate = validator
        self._modified.append("_validate")

        return self

    def hook(self, f: Callable[[_T_def], _T_def]) -> Self:
        self._hook = f
        self._modified.append("_hook")

        return self

    @overload
    def tag(self, tag_type: Literal["untagged", "external"]) -> Self: ...

    @overload
    def tag(self, tag_type: Literal["internal"], *, tag: object) -> Self: ...

    @overload
    def tag(self, tag_type: Literal["adjacent"], *, tag: object, content: object) -> Self: ...

    def tag(self, tag_type: Literal["untagged", "external", "internal", "adjacent"], **kwargs: object) -> Self:
        self._tag = tag_type
        self._tag_info = kwargs

        self._modified.extend(["_tag", "_tag_info"])

        return self

    def type_name(self, name: str) -> Self:
        self._type_name = name
        self._modified.append("_type_name")

        return self


def rename(key: str) -> Item:
    return Item().rename(key)


def default(default: Callable[[], _T_def]) -> Item[_T_def]:
    return Item().default(default)  # pyright: ignore[reportArgumentType, reportReturnType]


def validate(validator: Callable[[_T_def], bool]) -> Item[_T_def]:
    return Item().validate(validator)  # pyright: ignore[reportArgumentType, reportReturnType]


def hook(f: Callable[[_T_def], _T_def]) -> Item[_T_def]:
    return Item().hook(f)  # pyright: ignore[reportArgumentType, reportReturnType]


@overload
def tag(tag_type: Literal["untagged", "external"]) -> Item: ...


@overload
def tag(tag_type: Literal["internal"], *, tag: object) -> Item: ...


@overload
def tag(tag_type: Literal["adjacent"], *, tag: object, content: object) -> Item: ...


def tag(tag_type: Literal["untagged", "external", "internal", "adjacent"], **kwargs: object) -> Item:
    return Item().tag(tag_type, **kwargs)


def type_name(name: str) -> Item:
    return Item().type_name(name)


# endregion


# region Main validation logic
# ----------------------------------------------------------------------------------------------------------------------


def is_model(obj: object) -> TypeGuard[type["Model"]]:
    """Return True if the object is a subclass of spec.Model."""

    return isinstance(obj, type) and issubclass(obj, Model)


def _is_list_of_internal_items(obj: object) -> TypeGuard[list[_InternalItem]]:
    return isinstance(obj, list) and all(isinstance(item, _InternalItem) for item in obj)  # pyright: ignore [reportUnknownVariableType]


def validate_value(
    item: _InternalItem,
    model: "Model",
    value: Any,
    root_item: _InternalItem | None = None,
    root_value: Any = MISSING,
) -> Any:
    root_item = root_item or item
    root_value = root_value if (root_value is not MISSING) else value

    with add_note_to_exception(f"At {type(model).__name__}.{item.key}"):
        if is_model(item.typ):
            return item.typ(value)

        origin = get_origin(item.typ)

        if _is_list_of_internal_items(item.typ):
            # Tagged case: item.typ is only a list when the model is tagged (implicitly as a union or explicitly).
            # See convert_to_item for that assignment.
            match (item.tag, value):
                # Union case.
                case ("untagged", _):
                    for arg in item.typ:
                        try:
                            value = validate_value(arg, model, value, root_item, root_value)
                        except SpecError:
                            pass
                        else:
                            break
                    else:
                        raise InvalidType.from_expected(model, item, root_item, root_value)

                case ("external", dict()):
                    value = cast(dict[str, Any], value)
                    try:
                        key, value = next(iter(value.items()))
                    except StopIteration:
                        raise UnknownUnionKey(type(model).__name__, "") from None

                    for internal_item in item.typ:
                        assert internal_item.type_name

                        if internal_item.type_name == key:
                            value = validate_value(internal_item, model, value, root_item, root_value)
                            model._spec_tag_map[internal_item.key] = key
                            break
                    else:
                        raise UnknownUnionKey(type(model).__name__, key)

                case ("adjacent", dict()):
                    value = cast(dict[str, Any], value)

                    tag_key = item.tag_info["tag"]
                    content_key = item.tag_info["content"]

                    try:
                        key, content = value[tag_key], value[content_key]
                    except KeyError:
                        raise InvalidType.from_expected(model, item, root_item, root_value) from None

                    for internal_item in item.typ:
                        assert internal_item.type_name

                        if internal_item.type_name == key:
                            value = validate_value(internal_item, model, content, root_item, root_value)
                            model._spec_tag_map[internal_item.key] = key
                            break
                    else:
                        raise UnknownUnionKey(type(model).__name__, key)

                case ("internal", dict()):
                    value = cast(dict[str, Any], value)

                    tag_key = item.tag_info["tag"]

                    try:
                        key = value[tag_key]
                    except KeyError:
                        raise MissingRequiredKey(type(model).__name__, tag_key) from None

                    for internal_item in item.typ:
                        assert internal_item.type_name

                        if internal_item.type_name == key:
                            value = validate_value(internal_item, model, value, root_item, root_value)
                            model._spec_tag_map[internal_item.key] = key
                            break
                    else:
                        raise UnknownUnionKey(type(model).__name__, key)

                case ("external" | "adjacent" | "internal", _):
                    raise InvalidType.from_expected(model, item, root_item, root_value)

        # Any case: Checking Any with isinstance won't work.
        elif isinstance(origin, type) and issubclass(origin, Any):
            pass

        # Base case.
        elif not isinstance(value, origin):
            raise InvalidType.from_expected(model, item, root_item, root_value)

        if origin in (list, set, tuple):
            internal_item = item.internal_items[0]

            list_output: list[_InternalItem] = [
                validate_value(internal_item, model, internal_value, root_item, root_value) for internal_value in value
            ]

            value = origin(list_output)

        elif origin is dict:
            internal_item_key, internal_item_value = item.internal_items

            dict_output: dict[Any, Any] = {}

            for internal_key, internal_value in value.items():
                validated_key = validate_value(internal_item_key, model, internal_key, root_item, root_value)
                validated_value = validate_value(internal_item_value, model, internal_value, root_item, root_value)

                dict_output[validated_key] = validated_value

            value = dict_output

        if not item.validate(value):
            raise FailedValidationHook(type(model).__name__, item.key)

        return item.hook(value)


def convert_to_item(klass: type, key: str, annotation: Any, existing: Item | None = None) -> Item:
    """Convert a type annotation into an instance of spec.Item.

    Parameters
    ----------
    cls: type
        The class that the type annotation came from.
    key: str
        The variable name that the type annotation was attached to.
    annotation: Any
        The annotation.
    existing: Item | None, optional
        An existing Item to use when converting the annotation. Defaults to None.

    Returns
    -------
    Item
        The corresponding Item to the annotation.
    """

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return convert_to_item(klass, key, args[0], args[1])

    if existing:
        existing._key = key
        existing._typ = annotation

        item = existing
    else:
        # Base case.
        item = Item(key=key, typ=annotation)

    if origin in {Union, types.UnionType}:
        # Provide NODEFAULT as a default if NoDefault is in the annotation and no default value was set.
        if any(x == NoDefault for x in args) and "_default" not in item._modified:
            item._default = lambda: NODEFAULT

        internal_types: list[_InternalItem] = []

        for typ in args:
            inner_item = convert_to_item(klass, key, typ)

            if item._tag != "untagged":
                if is_model(typ) and not inner_item._type_name:
                    inner_item._type_name = typ._spec_model_type_name

                if not inner_item._type_name:
                    raise MissingTypeName(klass.__name__, key, repr(typ))

            internal_types.append(inner_item._to_internal())

        item._typ = internal_types

    if existing:
        for modified in existing._modified:
            setattr(item, modified, getattr(existing, modified))

    item._internal_items = [convert_to_item(klass, key, typ)._to_internal() for typ in get_args(annotation)]

    return item


def value_to_dict(value: object, tag_map: dict[str, Any], item: _InternalItem) -> Any:
    """Normalize a value within a model for the dict version of the model.

    If the model is tagged, the value may be wrapped in a dict preemptively.
    """

    output: Any

    match value:
        case Model():
            output = value.to_dict()

        case list() | set() | tuple():
            output = type(value)(  # pyright: ignore [reportUnknownArgumentType]
                (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value)
                for inner_value in value  # pyright: ignore [reportUnknownVariableType]
                if inner_value is not NODEFAULT
            )

        case dict():
            output = {
                inner_key: (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value)
                for inner_key, inner_value in value.items()  # pyright: ignore [reportUnknownVariableType]
                if inner_value is not NODEFAULT
            }

        case _:
            output = value

    match item.tag:
        case "external":
            output = {tag_map[item.key]: output}
        case "internal":
            output[item.tag_info["tag"]] = tag_map[item.key]
        case "adjacent":
            output = {item.tag_info["tag"]: tag_map[item.key], item.tag_info["content"]: output}
        case _:
            pass

    return output


# endregion


# region Renaming schemes
# ----------------------------------------------------------------------------------------------------------------------


def to_default(key: str) -> str:
    return key


def to_upper(key: str) -> str:
    return key.upper()


def to_camel_case(key: str) -> str:
    match key.split("_"):
        case [first]:
            return first[0].lower() + first[1:]
        case [first, *rest]:
            return "".join([(first[0].lower() + first[1:]), *(word[0].upper() + word[1:] for word in rest)])
        case _:  # pragma: no cover
            return ""


def to_pascal_case(key: str) -> str:
    parts = [(word[0].upper() + word[1:]) for word in key.split("_") if word]
    return "".join(parts)


def to_kebab_case(key: str) -> str:
    return key.replace("_", "-")


def to_screaming_kebab_case(key: str) -> str:
    return to_upper(to_kebab_case(key))


_BUILTIN_RENAME_SCHEMES = {
    "default": to_default,
    "upper": to_upper,
    "camel": to_camel_case,
    "pascal": to_pascal_case,
    "kebab": to_kebab_case,
    "screaming_kebab": to_screaming_kebab_case,
}


# endregion


# region Model
# ----------------------------------------------------------------------------------------------------------------------


RenameScheme = Literal["default", "upper", "camel", "pascal", "kebab", "screaming_kebab"] | Callable[[str], str]
OnExtrasCallback: TypeAlias = Callable[[set[str], set[str], dict[str, Any]], Any]


class Model:
    _spec_model_items: ClassVar[dict[str, _InternalItem]]
    _spec_model_type_name: ClassVar[str]
    _spec_model_extras_policy: ClassVar[bool | OnExtrasCallback]

    def __init_subclass__(
        cls,
        *,
        type_name: str | None = None,
        rename: RenameScheme = to_default,
        allow_extras: bool | OnExtrasCallback = True,
    ):
        try:
            rename_scheme = _BUILTIN_RENAME_SCHEMES[rename]  # pyright: ignore [reportArgumentType]
        except KeyError:
            if not callable(rename):
                msg = (
                    "'rename' must be one of 'default', 'upper', 'camel', 'pascal', 'kebab', 'screaming_kebnab', or "
                    "a custom callable that takes a key and returns the renamed key."
                )
                raise TypeError(msg) from None
            rename_scheme = rename

        if not (isinstance(allow_extras, bool) or callable(allow_extras)):
            msg = "'allow_extras' must either be True, False, or a custom callback function."
            raise TypeError(msg)

        cls._spec_model_type_name = type_name or cls.__name__
        if isinstance(allow_extras, bool):
            cls._spec_model_extras_policy = allow_extras
        else:
            cls._spec_model_extras_policy = staticmethod(allow_extras)

        items: dict[str, _InternalItem] = {}

        # Allow inheritance of items, as well as replacment of them in subclasses.
        for base in reversed(cls.__mro__):
            if issubclass(base, Model) and (base_items := getattr(base, "_spec_model_items", None)):
                items.update(base_items)

        for key, annotation in cls.__annotations__.items():
            ann = resolve_annotation(annotation, sys.modules[cls.__module__].__dict__, None, None)
            item = convert_to_item(cls, key, ann)

            if "rename" not in item._modified and item._rename is None:
                item._rename = rename_scheme(key)

            internal_item = item._to_internal()

            if (default := getattr(cls, key, MISSING)) is not MISSING:
                internal_item.default = lambda: default

            items[internal_item.actual_key] = internal_item

        cls._spec_model_items = items

    @overload
    def __init__(self, /, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, data: dict[str, Any] | None = None, /) -> None: ...

    def __init__(self, data: dict[str, Any] | None = None, /, **kwargs: Any) -> None:
        with format_exception_notes():
            self._spec_tag_map: dict[str, str] = {}

            if data is None and not kwargs:
                raise MissingArgument(type(self).__name__)

            if data and kwargs:
                msg = "Only data or kwargs is accepted, not both."
                raise ValueError(msg)

            data = data or kwargs

            if self._spec_model_extras_policy is not True:
                allowed_keys = set(self._spec_model_items)
                extra_keys = set(data) - allowed_keys
                if extra_keys:
                    if self._spec_model_extras_policy is not False:
                        self._spec_model_extras_policy(extra_keys, allowed_keys, data)
                    else:
                        raise NoExtraKeysAllowed(type(self).__name__, extra_keys, allowed_keys)

            for key, item in self._spec_model_items.items():
                if key not in data:
                    if item.default:
                        setattr(self, item.key, item.default())
                    else:
                        raise MissingRequiredKey(type(self).__name__, key)

            for key, value in data.items():
                if not (item := self._spec_model_items.get(key)):
                    continue

                new_value = validate_value(item, self, value)

                setattr(self, item.key, new_value)

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def __repr__(self):
        items = [f"{item.key}={getattr(self, item.key, MISSING)!r}" for item in self._spec_model_items.values()]
        return f"<{type(self).__name__} {' '.join(items)}>"

    def to_dict(self) -> dict[str, Any]:
        """Return the validated model as a dict, respecting the tag scheme."""

        return {
            item.actual_key: value_to_dict(getattr(self, item.key), self._spec_tag_map, item)
            for item in self._spec_model_items.values()
            if getattr(self, item.key) is not NODEFAULT
        }


class TransparentModel(Generic[_T], Model):
    value: _T

    def __init_subclass__(cls, *, item: Item | None = None):
        typ = get_args(get_original_bases(cls)[0])[0]
        cls._spec_model_items = {"value": convert_to_item(cls, "value", typ, item)._to_internal()}

    def __init__(self, data: object):
        super().__init__({"value": data})

    def __repr__(self):
        return f"<{type(self).__name__} {self.value!r}>"

    def to_dict(self) -> dict[str, Any]:
        return value_to_dict(self.value, self._spec_tag_map, self._spec_model_items["value"])


def transparent(typ: type[_T] | Any, item: Item | None = None) -> type[TransparentModel[_T]]:
    if get_origin(typ) in {Union, types.UnionType}:
        name = "Or".join([_get_type_name(t) for t in get_args(typ)])
    else:
        name = _get_type_name(typ)

    return types.new_class(name, (TransparentModel[typ],), {"item": item})


# endregion
