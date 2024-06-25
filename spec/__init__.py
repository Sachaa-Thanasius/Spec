# SPDX-FileCopyrightText: 2023-present Zomatree <me@zomatree.live>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    overload,
)
from typing import get_origin as tp_get_origin

if sys.version_info >= (3, 13):  # pragma: >=3.13 cover
    from typing import TypeVar
else:  # pragma: <3.13 cover
    from typing_extensions import TypeVar

if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from types import get_original_bases
else:  # pragma: <3.12 cover
    from typing_extensions import get_original_bases

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self


__all__ = (
    "__version__",
    # == Exceptions
    "SpecError",
    "MissingArgument",
    "MissingRequiredKey",
    "InvalidType",
    "FailedValidation",
    "UnknownUnionKey",
    "MissingTypeName",
    # == Item
    "Item",
    "rename",
    "default",
    "validate",
    "hook",
    "tag",
    "type_name",
    # == Model
    "is_model",
    "Model",
    "TransparentModel",
    "transparent",
    # Renaming schemes
    "RenameBase",
    "Default",
    "Upper",
    "CamelCase",
    "PascalCase",
    "KebabCase",
    "ScreamingKebabCase",
    "RenameScheme",
)

__version__ = "0.0.1"

_T = TypeVar("_T")
_T_def = TypeVar("_T_def", default=Any)


# region ==== Exceptions ====


class SpecError(Exception):
    """Base exception class for Spec."""


class MissingArgument(SpecError):
    """Exception that's raised when a Model is instantiated with no arguments."""


class MissingRequiredKey(SpecError):
    """Exception that's raised when a Model is instantiated without a required key."""


class InvalidType(SpecError):
    """Exception that's raised when the type for a given value doesn't match the expected type from the Model."""

    @classmethod
    def from_expected(cls, model: Model, item: _InternalItem, root_item: _InternalItem, root_value: Any) -> Self:
        return cls(
            f"{type(model).__name__}.{item.key} expected type {prettify_type(root_item)}"
            f" but found {generate_type_repr_from_data(root_value)}"
        )


class FailedValidation(SpecError):
    """Exception that's raised when a validation hook for an item in a model fails."""


class UnknownUnionKey(SpecError):
    pass


class MissingTypeName(SpecError):
    pass


# endregion

# region ==== Utilities ====


class _UniqueList(list[_T_def]):
    def append(self, value: _T_def) -> None:
        if value not in self:
            super().append(value)


class _Missing:
    def __bool__(self) -> Literal[False]:
        return False


MISSING = _Missing()


def _is_union(typ: Any) -> bool:
    """Determine if the origin of a type is Union/UnionType."""

    origin = get_origin(typ)

    return (origin is Union) or (origin is types.UnionType)


def get_origin(obj: Any) -> Any:
    """Get the unsubscripted version of a type, or just the type if it doesn't support subscripting or wasn't
    subscripted.

    See typing.get_origin for more details about the base functionality.

    Examples
    --------
    Only giving examples with different behavior from typing.get_origin.

    >>> assert get_origin(list) is list
    >>> assert get_origin(set) is set
    >>> assert get_origin(tuple) is tuple
    >>> assert get_origin(dict) is dict
    """

    return tp_get_origin(obj) or obj


def prettify_type(item: _InternalItem) -> str:
    if not isinstance(item.typ, list) and (generics := item.internal_items):
        generic_str = f"[{', '.join([prettify_type(generic) for generic in generics])}]"
    else:
        generic_str = ""

    if not isinstance(item.typ, list):  # pyright: ignore [reportUnknownMemberType]
        return f"{getattr(item.typ, '__name__', type(item.typ).__name__)}{generic_str}"
    else:
        typ_names = [getattr(typ.typ, "__name__", type(typ.typ).__name__) for typ in item.typ]  # pyright: ignore
        return f"{repr_as_union(typ_names)}{generic_str}"


def repr_as_union(types: list[Any]) -> str:
    return " | ".join(types or ["Unknown"])


def generate_type_repr_from_data(data: Any) -> str:
    match data:
        case list() | set() | tuple():
            unique_types = _UniqueList()

            for value in data:  # pyright: ignore [reportUnknownVariableType]
                unique_types.append(generate_type_repr_from_data(value))

            generics = [repr_as_union(unique_types)]

        case dict():
            key_types = _UniqueList()
            value_types = _UniqueList()

            for key, value in data.items():  # pyright: ignore [reportUnknownVariableType]
                key_types.append(generate_type_repr_from_data(key))
                value_types.append(generate_type_repr_from_data(value))

            generics = [repr_as_union(key_types), repr_as_union(value_types)]
        case _:
            generics = []

    generic_str = f"[{', '.join(generics)}]" if generics else ""

    return f"{type(data).__name__}{generic_str}"  # pyright: ignore [reportUnknownArgumentType]


# endregion

# region ==== Items ====


@dataclass
class Item(Generic[_T_def]):
    _key: str | None = None
    _rename: str | None = None
    _typ: type[_T_def] | None = None
    _internal_items: list[_InternalItem] | None = None
    _default: Callable[[], _T_def] | None = None
    _modified: list[str] = field(default_factory=list)
    _validate: Callable[[_T_def], bool] = lambda _: True
    _hook: Callable[[_T_def], _T_def] = lambda x: x
    _tag: Literal["untagged", "external", "internal", "adjacent"] = "untagged"
    _tag_info: dict[str, Any] = field(default_factory=dict)
    _type_name: str | None = None

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
    def tag(self, tag_type: Literal["untagged"]) -> Self: ...

    @overload
    def tag(self, tag_type: Literal["external"]) -> Self: ...

    @overload
    def tag(self, tag_type: Literal["internal"], *, tag: Any) -> Self: ...

    @overload
    def tag(self, tag_type: Literal["adjacent"], *, tag: Any, content: Any) -> Self: ...

    def tag(self, tag_type: Literal["untagged", "external", "internal", "adjacent"], **kwargs: Any) -> Self:
        self._tag = tag_type
        self._tag_info = kwargs

        self._modified.extend(["_tag", "_tag_info"])

        return self

    def type_name(self, name: str) -> Self:
        self._type_name = name
        self._modified.append("_type_name")

        return self


@dataclass
class _InternalItem(Generic[_T_def]):
    key: str
    rename: str | None
    typ: type[_T_def]
    internal_items: list[_InternalItem]
    default: Callable[[], _T_def] | None
    validate: Callable[[_T_def], bool]
    hook: Callable[[_T_def], _T_def]
    tag: Literal["untagged", "external", "internal", "adjacent"]
    tag_info: dict[str, Any]
    type_name: str | None

    @property
    def actual_key(self) -> str:
        return self.rename or self.key


def rename(key: str) -> Item:
    return Item().rename(key)


def default(default: Callable[[], _T_def]) -> Item[_T_def]:
    return Item().default(default)


def validate(validator: Callable[[_T_def], bool]) -> Item[_T_def]:
    return Item().validate(validator)


def hook(f: Callable[[_T_def], _T_def]) -> Item[_T_def]:
    return Item().hook(f)


@overload
def tag(tag_type: Literal["untagged"]) -> Item[_T_def]: ...


@overload
def tag(tag_type: Literal["external"]) -> Item[_T_def]: ...


@overload
def tag(tag_type: Literal["internal"], *, tag: Any) -> Item[_T_def]: ...


@overload
def tag(tag_type: Literal["adjacent"], *, tag: Any, content: Any) -> Item[_T_def]: ...


def tag(tag_type: Literal["untagged", "external", "internal", "adjacent"], **kwargs: Any) -> Item[_T_def]:
    return Item().tag(tag_type, **kwargs)


def type_name(name: str) -> Item[_T_def]:
    return Item().type_name(name)


# endregion

# region ==== Model ====


def is_model(obj: Any) -> TypeGuard[type[Model]]:
    return isinstance(obj, type) and issubclass(obj, Model)


def _validate_item(  # noqa: PLR0912, PLR0915
    item: _InternalItem[Any],
    model: Model,
    value: Any,
    root_item: _InternalItem | None = None,
    root_value: Any = MISSING,
) -> Any:
    root_item = root_item or item
    root_value = root_value or value

    if is_model(item.typ):
        return item.typ(value)

    origin = get_origin(item.typ)

    if isinstance(item.typ, list):
        match item.tag:
            case "untagged":
                for arg in item.typ:
                    assert isinstance(arg, _InternalItem)

                    try:
                        value = _validate_item(arg, model, value, root_item, root_value)
                    except SpecError:
                        pass
                    else:
                        break
                else:
                    raise InvalidType.from_expected(model, item, root_item, root_value)

            case "external":
                if not isinstance(value, dict):
                    raise InvalidType.from_expected(model, item, root_item, root_value)

                try:
                    key, value = next(iter(value.items()))
                except StopIteration:
                    raise UnknownUnionKey("Unknown key found ``") from None

                for internal_item in item.typ:
                    assert isinstance(internal_item, _InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = _validate_item(internal_item, model, value, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

            case "adjacent":
                if not isinstance(value, dict):
                    raise InvalidType.from_expected(model, item, root_item, root_value)

                tag_key = item.tag_info["tag"]
                content_key = item.tag_info["content"]

                try:
                    key = value[tag_key]
                    content = value[content_key]
                except KeyError:
                    raise InvalidType.from_expected(model, item, root_item, root_value) from None

                for internal_item in item.typ:
                    assert isinstance(internal_item, _InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = _validate_item(internal_item, model, content, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

            case "internal":
                if not isinstance(value, dict):
                    raise InvalidType.from_expected(model, item, root_item, root_value)

                tag_key = item.tag_info["tag"]

                try:
                    key = value[tag_key]
                except KeyError:
                    raise MissingRequiredKey(f"Missing required key {type(model).__name__}.{tag_key}") from None

                for internal_item in item.typ:
                    assert isinstance(internal_item, _InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = _validate_item(internal_item, model, value, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

    elif not isinstance(value, origin):
        raise InvalidType.from_expected(model, item, root_item, root_value)

    if origin in (list, set, tuple):
        internal_item = item.internal_items[0]

        list_output: list[_InternalItem] = [
            _validate_item(internal_item, model, internal_value, root_item, root_value) for internal_value in value
        ]

        value = origin(list_output)

    elif origin is dict:
        internal_item_key, internal_item_value = item.internal_items

        dict_output: dict[Any, Any] = {}

        for internal_key, internal_value in value.items():
            validated_key = _validate_item(internal_item_key, model, internal_key, root_item, root_value)
            validated_value = _validate_item(internal_item_value, model, internal_value, root_item, root_value)

            dict_output[validated_key] = validated_value

        value = dict_output

    if not item.validate(value):
        raise FailedValidation(f"{type(model).__name__}.{item.key} failed validation")

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
        An existing Item to use in configuring the annotation item. Defaults to None. Used while recursively analyzing
        an annotation.

    Returns
    -------
    Item
        The corresponding Item to the annotation.
    """

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        # FIXME: Should this return immediately?
        item = convert_to_item(klass, key, args[0], args[1])
    elif existing:
        existing._key = key
        existing._typ = annotation

        item = existing
    else:
        # Base case.
        item = Item(_key=key, _typ=annotation)

    if _is_union(origin):
        # Provide None as a default if Optional is in the annotation and no default value was set.
        if any(x is types.NoneType for x in args) and "_default" not in item._modified:
            item._default = lambda: None

        internal_types: list[_InternalItem] = []

        for typ in args:
            inner_item = convert_to_item(klass, key, typ)

            if item._tag != "untagged":
                if is_model(typ) and not inner_item._type_name:
                    inner_item._type_name = typ._type_name

                if not inner_item._type_name:
                    raise MissingTypeName(f"{klass.__name__}.{key} union type is missing a type name for {typ}")

            internal_types.append(inner_item._to_internal())

        item._typ = internal_types

    if existing:
        for modified in existing._modified:
            setattr(item, modified, getattr(existing, modified))

    item._internal_items = [convert_to_item(klass, key, typ)._to_internal() for typ in get_args(annotation)]

    return item


def value_to_dict(value: Any, tag_map: dict[str, Any], item: _InternalItem) -> Any:
    """Normalize a value within a model for putting into a dict.

    If the model is tagged, the value may be wrapped in a dict preemptively.
    """

    match value:
        case Model():
            output = value.to_dict()

        case list() | set() | tuple():
            output = type(value)(  # pyright: ignore [reportUnknownArgumentType]
                (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value) for inner_value in value
            )

        case dict():
            output = {
                inner_key: (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value)
                for inner_key, inner_value in value.items()  # pyright: ignore
            }

        case _:
            output: Any = value

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


# region == Renaming schemes ==


class ImplementsRename(Protocol):
    def rename(self, key: str) -> str: ...


class RenameBase:
    @staticmethod
    def rename(key: str) -> str:
        raise NotImplementedError


class Default(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        return key


class Upper(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        return key.upper()


class CamelCase(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        match key.split("_"):
            case [first]:
                return first
            case [first, *rest]:
                return "".join([first, *(word[0].upper() + word[1:] for word in rest)])
            case _:
                return ""


class PascalCase(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        parts = [(word[0].upper() + word[1:]) for word in key.split("_") if word]
        return "".join(parts)


class KebabCase(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        return key.replace("_", "-")


class ScreamingKebabCase(RenameBase):
    @staticmethod
    def rename(key: str) -> str:
        return Upper.rename(KebabCase.rename(key))


RenameScheme: TypeAlias = Default | Upper | CamelCase | PascalCase | KebabCase | ScreamingKebabCase

# endregion


class Model:
    _items: ClassVar[dict[str, _InternalItem]]
    _type_name: ClassVar[str]

    def __init_subclass__(cls, *, type_name: str | None = None, rename: ImplementsRename = Default) -> None:
        items: dict[str, _InternalItem] = {}

        cls._type_name = type_name or cls.__name__

        for key, annotation in cls.__annotations__.items():
            item = convert_to_item(cls, key, annotation)

            if "rename" not in item._modified and item._rename is None:
                item._rename = rename.rename(key)

            intl_item = item._to_internal()

            if (default := getattr(cls, key, MISSING)) is not MISSING:
                intl_item.default = lambda: default

            items[intl_item.actual_key] = intl_item

        cls._items = items

    @overload
    def __init__(self, /, **kwargs: Any) -> None: ...

    @overload
    def __init__(self, data: dict[str, Any] | None = None, /) -> None: ...

    def __init__(self, data: dict[str, Any] | None = None, /, **kwargs: Any) -> None:
        self.__tag_map__: dict[str, str] = {}

        if data is None and not kwargs:
            raise MissingArgument("No data or kwargs passed to Model")

        data = data or kwargs

        for key, item in self._items.items():
            if key not in data:
                if item.default:
                    setattr(self, item.key, item.default())
                else:
                    raise MissingRequiredKey(f"Missing required key {type(self).__name__}.{key}")

        for key, value in data.items():
            if not (item := self._items.get(key)):
                continue

            new_value = _validate_item(item, self, value)

            setattr(self, item.key, new_value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        items = [f"{item.key}={getattr(self, item.key)!r}" for item in self._items.values()]

        return f"<{type(self).__name__} {' '.join(items)}>"

    def to_dict(self) -> dict[str, Any]:
        """Return the validated model as a dict, respecting the tag scheme."""

        return {
            item.actual_key: value_to_dict(getattr(self, item.key), self.__tag_map__, item)
            for item in self._items.values()
        }


class TransparentModel(Generic[_T], Model):
    value: _T

    def __init_subclass__(cls, *, item: Item | None = None) -> None:
        typ = get_args(get_original_bases(cls)[0])[0]
        cls._items = {"value": convert_to_item(cls, "value", typ, item)._to_internal()}

    def __init__(self, data: Any) -> None:
        super().__init__({"value": data})

    def to_dict(self) -> dict[str, Any]:
        return value_to_dict(self.value, self.__tag_map__, self._items["value"])

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.value!r}>"


def transparent(typ: type[_T] | Any, item: Item | None = None) -> type[TransparentModel[_T]]:
    def _get_name(v: type) -> str:
        return getattr(v, "_type_name", v.__name__)

    if _is_union(typ):  # noqa: SIM108 # Readability.
        name = "Or".join([_get_name(v) for v in get_args(typ)])
    else:
        name = _get_name(typ)

    return types.new_class(name, (TransparentModel[typ],), {"item": item})


# endregion
