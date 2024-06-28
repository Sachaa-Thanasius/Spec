# SPDX-FileCopyrightText: 2023-present Zomatree <me@zomatree.live>
#
# SPDX-License-Identifier: MIT

import sys
import types
from collections.abc import Callable, Iterable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    Union,
    get_args,
    get_origin as tp_get_origin,
    overload,
)

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
    # Exceptions
    "SpecError",
    "MissingArgument",
    "MissingRequiredKey",
    "InvalidType",
    "FailedValidation",
    "UnknownUnionKey",
    "MissingTypeName",
    "ExtraKeysDisallowed",
    # Item
    "Item",
    "rename",
    "default",
    "validate",
    "hook",
    "tag",
    "type_name",
    # Model
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

_T = TypeVar("_T")
_T_def = TypeVar("_T_def", default=Any)


# region Exceptions


class SpecError(Exception):
    """Base exception class for spec."""


class MissingArgument(SpecError):
    """Exception that's raised when a Model is instantiated with no arguments."""


class MissingRequiredKey(SpecError):
    """Exception that's raised when a Model is instantiated without a required key."""


class InvalidType(SpecError):
    """Exception that's raised when the type for a given value doesn't match the expected type from the Model."""

    @classmethod
    def from_expected(
        cls,
        model: "Model",
        item: "_InternalItem",
        root_item: "_InternalItem",
        root_value: object,
    ) -> Self:
        return cls(
            f"'{type(model).__name__}.{item.key}' expected type '{_prettify_type(root_item)}'"
            f" but found '{_generate_type_repr_from_data(root_value)}'"
        )


class FailedValidation(SpecError):
    """Exception that's raised when a validation hook for an item in a model fails."""


class UnknownUnionKey(SpecError):
    """Exception that's raised when the provided tag for a tagged model doesn't work to find any sub-models."""


class MissingTypeName(SpecError):
    """Exception that's raised when a tagged union model is missing a type name."""


class ExtraKeysDisallowed(SpecError):
    """Exception that's raised when a model prohibits extra keys but they're provided anyway."""

    def __init__(self, extra_keys: Iterable[str], allowed_keys: Iterable[str]) -> None:
        super().__init__(f"No extra keys are allowed in this model. Extra(s) found: {extra_keys}.")
        self.extra_keys = extra_keys
        self.allowed_keys = allowed_keys


# endregion

# region Utilities


class _Missing:
    def __bool__(self) -> Literal[False]:
        return False


MISSING: Any = _Missing()


def _is_union(typ: type) -> bool:
    """Determine if the origin of a type is Union/UnionType."""

    origin = get_origin(typ)

    return (origin is Union) or (origin is types.UnionType)


def _get_type_name(v: type) -> str:
    return getattr(v, "_type_name", v.__name__)


def get_origin(typ: type) -> Any:
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


def _prettify_type(item: "_InternalItem") -> str:
    # Reminder: item.typ is only a list when tags are involved.
    if not isinstance(item.typ, list) and (generics := item.internal_items):
        generic_str = f"[{', '.join([_prettify_type(generic) for generic in generics])}]"
    else:
        generic_str = ""

    if not isinstance(item.typ, list):  # pyright: ignore [reportUnknownMemberType]
        return f"{getattr(item.typ, '__name__', type(item.typ).__name__)}{generic_str}"
    else:  # noqa: RET505 # Readability.
        typ_names = [getattr(typ.typ, "__name__", type(typ.typ).__name__) for typ in item.typ]  # pyright: ignore
        return f"{_repr_as_union(typ_names)}{generic_str}"


def _repr_as_union(type_reprs: Iterable[str]) -> str:
    return " | ".join(type_reprs or ["Unknown"])


def _generate_type_repr_from_data(data: object) -> str:
    # Ensure uniqueness and conserve order.
    match data:
        case list() | set() | tuple():
            unique_types = tuple(dict.fromkeys(map(_generate_type_repr_from_data, data)))  # pyright: ignore [reportUnknownArgumentType]
            generics = (_repr_as_union(unique_types),)

        case dict():
            key_types = tuple(dict.fromkeys(map(_generate_type_repr_from_data, data.keys())))  # pyright: ignore [reportUnknownArgumentType]
            value_types = tuple(dict.fromkeys(map(_generate_type_repr_from_data, data.values())))  # pyright: ignore [reportUnknownArgumentType]

            generics = (_repr_as_union(key_types), _repr_as_union(value_types))
        case _:
            generics = ()

    generic_str = f"[{', '.join(generics)}]" if generics else ""

    return f"{type(data).__name__}{generic_str}"  # pyright: ignore [reportUnknownArgumentType]


# endregion

# region Items


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
    ) -> None:
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
        modified: list[str] = MISSING,
        validate: Callable[[_T_def], bool] = lambda _: True,
        hook: Callable[[_T_def], _T_def] = lambda x: x,
        tag: Literal["untagged", "external", "internal", "adjacent"] = "untagged",
        tag_info: dict[str, Any] = MISSING,
        type_name: str | None = None,
    ) -> None:
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
def tag(tag_type: Literal["untagged"]) -> Item: ...


@overload
def tag(tag_type: Literal["external"]) -> Item: ...


@overload
def tag(tag_type: Literal["internal"], *, tag: object) -> Item: ...


@overload
def tag(tag_type: Literal["adjacent"], *, tag: object, content: object) -> Item: ...


def tag(tag_type: Literal["untagged", "external", "internal", "adjacent"], **kwargs: object) -> Item:
    return Item().tag(tag_type, **kwargs)


def type_name(name: str) -> Item:
    return Item().type_name(name)


# endregion

# region Model


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
    root_value = root_value or value

    if is_model(item.typ):
        return item.typ(value)

    origin = get_origin(item.typ)

    if _is_list_of_internal_items(item.typ):
        # Tagged case: item.typ is only a list when the model is tagged (implicitly as a union or explicitly).
        # See convert_to_item for that assignment.
        match (item.tag, value):
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
                try:
                    key, value = next(iter(value.items()))  # pyright: ignore [reportUnknownArgumentType, reportUnknownVariableType]
                except StopIteration:
                    msg = "Unknown key found ``"
                    raise UnknownUnionKey(msg) from None

                for internal_item in item.typ:
                    assert internal_item.type_name

                    if internal_item.type_name == key:
                        value = validate_value(internal_item, model, value, root_item, root_value)
                        model._tag_map[internal_item.key] = key
                        break
                else:
                    msg = f"Unknown key found `{key}`"
                    raise UnknownUnionKey(msg)

            case ("adjacent", dict()):
                tag_key = item.tag_info["tag"]
                content_key = item.tag_info["content"]

                try:
                    key, content = value[tag_key], value[content_key]  # pyright: ignore [reportUnknownVariableType]
                except KeyError:
                    raise InvalidType.from_expected(model, item, root_item, root_value) from None

                for internal_item in item.typ:
                    assert internal_item.type_name

                    if internal_item.type_name == key:
                        value = validate_value(internal_item, model, content, root_item, root_value)
                        model._tag_map[internal_item.key] = key
                        break
                else:
                    msg = f"Unknown key found `{key}`"
                    raise UnknownUnionKey(msg)

            case ("internal", dict()):
                tag_key = item.tag_info["tag"]

                try:
                    key = value[tag_key]  # pyright: ignore [reportUnknownVariableType]
                except KeyError:
                    msg = f"Missing required key '{type(model).__name__}.{tag_key}'."
                    raise MissingRequiredKey(msg) from None

                for internal_item in item.typ:
                    assert internal_item.type_name

                    if internal_item.type_name == key:
                        value = validate_value(internal_item, model, value, root_item, root_value)
                        model._tag_map[internal_item.key] = key
                        break
                else:
                    msg = f"Unknown key found `{key}`"
                    raise UnknownUnionKey(msg)

            case ("external" | "adjacent" | "internal", _):
                raise InvalidType.from_expected(model, item, root_item, root_value)

    elif not isinstance(value, origin):
        # Base case: This is where we go if the type of the value doesn't match its annotation.
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
        msg = f"'{type(model).__name__}.{item.key}' failed validation."
        raise FailedValidation(msg)

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

    if _is_union(origin):
        # Provide None as a default if Optional is in the annotation and no default value was set.
        if any(x is types.NoneType for x in args) and "_default" not in item._modified:
            item._default = lambda: None

        internal_types: list[_InternalItem] = []

        for typ in args:
            inner_item = convert_to_item(klass, key, typ)

            if item._tag != "untagged":
                if is_model(typ) and not inner_item._type_name:
                    inner_item._type_name = typ._spec_model_type_name

                if not inner_item._type_name:
                    msg = f"'{klass.__name__}.{key}' union type is missing a type name for '{typ}'."
                    raise MissingTypeName(msg)

            internal_types.append(inner_item._to_internal())

        item._typ = internal_types

    if existing:
        for modified in existing._modified:
            setattr(item, modified, getattr(existing, modified))

    item._internal_items = [convert_to_item(klass, key, typ)._to_internal() for typ in get_args(annotation)]

    return item


def value_to_dict(value: Any, tag_map: dict[str, Any], item: _InternalItem) -> Any:
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
            )

        case dict():
            output = {
                inner_key: (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value)
                for inner_key, inner_value in value.items()  # pyright: ignore [reportUnknownVariableType]
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


# region Renaming schemes


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
            case _:  # pragma: no cover
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


class HasRename(Protocol):
    def rename(self, key: str) -> str: ...


# endregion


OnExtrasCallback: TypeAlias = Callable[[set[str], set[str], dict[str, Any]], Any]


class Model:
    if TYPE_CHECKING:
        _spec_model_items: ClassVar[dict[str, _InternalItem]]
        _spec_model_type_name: ClassVar[str]
        _spec_model_extras_policy: ClassVar[Literal["allow", "deny"]] | OnExtrasCallback

    def __init_subclass__(
        cls,
        *,
        type_name: str | None = None,
        rename: HasRename = Default,
        extras_policy: Literal["allow", "deny"] | OnExtrasCallback = "allow",
    ) -> None:
        if extras_policy not in {"allow", "deny"} and not callable(extras_policy):
            msg = "'with_extras' must either be 'allow', 'deny', or a custom callback function."
            raise ValueError(msg)

        cls._spec_model_type_name = type_name or cls.__name__
        cls._spec_model_extras_policy = extras_policy

        items: dict[str, _InternalItem] = {}

        for base in reversed(cls.__mro__):
            if issubclass(base, Model) and (base_items := getattr(base, "_spec_model_items", None)):
                items.update(base_items)

        for key, annotation in cls.__annotations__.items():
            item = convert_to_item(cls, key, annotation)

            if "rename" not in item._modified and item._rename is None:
                item._rename = rename.rename(key)

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
        self._tag_map: dict[str, str] = {}

        if data is None and not kwargs:
            msg = "No data or kwargs passed to Model."
            raise MissingArgument(msg)

        if data and kwargs:
            msg = "Only data or kwargs is accepted, not both."
            raise ValueError(msg)

        data = data or kwargs

        if self._spec_model_extras_policy != "allow":
            allowed_keys = set(self._spec_model_items)
            extra_keys = set(data) - allowed_keys
            if extra_keys:
                if self._spec_model_extras_policy != "deny":
                    self._spec_model_extras_policy(extra_keys, allowed_keys, data)
                else:
                    raise ExtraKeysDisallowed(extra_keys, allowed_keys)

        for key, item in self._spec_model_items.items():
            if key not in data:
                if item.default:
                    setattr(self, item.key, item.default())
                else:
                    msg = f"Missing required key '{type(self).__name__}.{key}'."
                    raise MissingRequiredKey(msg)

        for key, value in data.items():
            if not (item := self._spec_model_items.get(key)):
                continue

            new_value = validate_value(item, self, value)

            setattr(self, item.key, new_value)

        if post_init := getattr(self, "__post_init__", None):
            post_init()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        items = [f"{item.key}={getattr(self, item.key)!r}" for item in self._spec_model_items.values()]
        return f"<{type(self).__name__} {' '.join(items)}>"

    def to_dict(self) -> dict[str, Any]:
        """Return the validated model as a dict, respecting the tag scheme."""

        return {
            item.actual_key: value_to_dict(getattr(self, item.key), self._tag_map, item)
            for item in self._spec_model_items.values()
        }


class TransparentModel(Generic[_T], Model):
    value: _T

    def __init_subclass__(cls, *, item: Item | None = None) -> None:
        typ = get_args(get_original_bases(cls)[0])[0]
        cls._spec_model_items = {"value": convert_to_item(cls, "value", typ, item)._to_internal()}

    def __init__(self, data: object) -> None:
        super().__init__({"value": data})

    def to_dict(self) -> dict[str, Any]:
        return value_to_dict(self.value, self._tag_map, self._spec_model_items["value"])

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.value!r}>"


def transparent(typ: type[_T] | Any, item: Item | None = None) -> type[TransparentModel[_T]]:
    if _is_union(typ):  # noqa: SIM108 # Readability.
        name = "Or".join([_get_type_name(t) for t in get_args(typ)])
    else:
        name = _get_type_name(typ)

    return types.new_class(name, (TransparentModel[typ],), {"item": item})


# endregion
