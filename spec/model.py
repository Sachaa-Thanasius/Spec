from __future__ import annotations

import types
from typing import Annotated, Any, ClassVar, Generic, TypeAlias, TypeGuard, TypeVar, get_args, overload

from .errors import (
    FailedValidation,
    InvalidType,
    MissingArgument,
    MissingRequiredKey,
    MissingTypeName,
    SpecError,
    UnknownUnionKey,
)
from .item import InternalItem, Item
from .util import (
    Missing,
    generate_type_from_data,
    get_origin,
    get_original_bases,
    get_type_name,
    is_union,
    pretty_type,
)

__all__ = (
    "is_model",
    "generate_invalid_type",
    "convert_to_item",
    "value_to_dict",
    "RenameBase",
    "Default",
    "Upper",
    "CamelCase",
    "PascalCase",
    "KebabCase",
    "ScreamingKebabCase",
    "RenameScheme",
    "Model",
    "TransparentModel",
    "transparent",
)

T = TypeVar("T")


def is_model(obj: Any) -> TypeGuard[type[Model]]:
    return isinstance(obj, type) and issubclass(obj, Model)


def generate_invalid_type(model: Model, item: InternalItem, root_item: InternalItem, root_value: Any) -> InvalidType:
    return InvalidType(
        f"{model.__class__.__name__}.{item.key} expected type {pretty_type(root_item)}"
        f" but found {generate_type_from_data(root_value)}"
    )


def validate(  # noqa: PLR0912, PLR0915
    item: InternalItem[Any],
    model: Model,
    value: Any,
    root_item: InternalItem | None = None,
    root_value: Any = Missing,
) -> Any:
    root_item = root_item or item
    root_value = root_value or value

    if is_model(item.ty):
        return item.ty(value)

    origin = get_origin(item.ty)

    if isinstance(item.ty, list):
        match item.tag:
            case "untagged":
                for arg in item.ty:
                    assert isinstance(arg, InternalItem)

                    try:
                        value = validate(arg, model, value, root_item, root_value)
                    except SpecError:
                        pass
                    else:
                        break
                else:
                    raise generate_invalid_type(model, item, root_item, root_value)

            case "external":
                if not isinstance(value, dict):
                    raise generate_invalid_type(model, item, root_item, root_value)

                try:
                    key, value = next(iter(value.items()))
                except StopIteration:
                    raise UnknownUnionKey("Unknown key found ``") from None

                for internal_item in item.ty:
                    assert isinstance(internal_item, InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = validate(internal_item, model, value, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

            case "adjacent":
                if not isinstance(value, dict):
                    raise generate_invalid_type(model, item, root_item, root_value)

                tag_key = item.tag_info["tag"]
                content_key = item.tag_info["content"]

                try:
                    key = value[tag_key]
                    content = value[content_key]
                except KeyError:
                    raise generate_invalid_type(model, item, root_item, root_value) from None

                for internal_item in item.ty:
                    assert isinstance(internal_item, InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = validate(internal_item, model, content, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

            case "internal":
                if not isinstance(value, dict):
                    raise generate_invalid_type(model, item, root_item, root_value)

                tag_key = item.tag_info["tag"]

                try:
                    key = value[tag_key]
                except KeyError:
                    raise MissingRequiredKey(f"Missing required key {model.__class__.__name__}.{tag_key}") from None

                for internal_item in item.ty:
                    assert isinstance(internal_item, InternalItem)
                    assert internal_item.type_name

                    if key == internal_item.type_name:
                        value = validate(internal_item, model, value, root_item, root_value)
                        model.__tag_map__[internal_item.key] = key
                        break
                else:
                    raise UnknownUnionKey(f"Unknown key found `{key}`")

    elif not isinstance(value, origin):
        # breakpoint()
        raise generate_invalid_type(model, item, root_item, root_value)

    if origin in (list, set, tuple):
        internal_item = item.internal_items[0]

        list_output: list[InternalItem] = [
            validate(internal_item, model, internal_value, root_item, root_value) for internal_value in value
        ]

        value = origin(list_output)

    elif origin is dict:
        internal_item_key, internal_item_value = item.internal_items

        dict_output: dict[Any, Any] = {}

        for internal_key, internal_value in value.items():
            validated_key = validate(internal_item_key, model, internal_key, root_item, root_value)
            validated_value = validate(internal_item_value, model, internal_value, root_item, root_value)

            dict_output[validated_key] = validated_value

        value = dict_output

    if not item.validate(value):
        raise FailedValidation(f"{model.__class__.__name__}.{item.key} failed validation")

    return item.hook(value)


def convert_to_item(cls: type, key: str, annotation: Any, existing: Item | None = None) -> Item:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        item = convert_to_item(cls, key, args[0], args[1])
    elif existing:
        existing._key = key
        existing._ty = annotation

        item = existing
    else:
        item = Item(_key=key, _ty=annotation)

    if is_union(origin):
        if any(x is types.NoneType for x in args) and "_default" not in item._modified:  # for handling Optional
            item._default = lambda: None

        internal_types: list[InternalItem] = []

        for ty in args:
            inner_item = convert_to_item(cls, key, ty)

            if item._tag != "untagged":
                if is_model(ty) and not inner_item._type_name:
                    inner_item._type_name = ty._type_name

                if not inner_item._type_name:
                    raise MissingTypeName(f"{cls.__name__}.{key} union type is missing a type name for {ty}")

            internal_types.append(inner_item._to_internal())

        item._ty = internal_types

    if existing:
        for modified in existing._modified:
            setattr(item, modified, getattr(existing, modified))

    item._internal_items = [convert_to_item(cls, key, ty)._to_internal() for ty in get_args(annotation)]

    return item


def value_to_dict(value: Any, tag_map: dict[str, Any], item: InternalItem) -> Any:
    output = value

    if isinstance(value, Model):
        output = value.to_dict()

    if isinstance(value, (list, set, tuple)):
        list_inner_output: list[Any] = [
            (inner_value.to_dict() if isinstance(inner_value, Model) else inner_value) for inner_value in value
        ]

        output: Any = type(value)(list_inner_output)

    if isinstance(value, dict):
        dict_inner_output: dict[Any, Any] = {}

        for inner_key, inner_value in value.items():
            if isinstance(inner_value, Model):
                inner_value = inner_value.to_dict()  # noqa: PLW2901

            dict_inner_output[inner_key] = inner_value

        output = dict_inner_output

    if item.tag == "external":
        output = {tag_map[item.key]: output}

    elif item.tag == "internal":
        output[item.tag_info["tag"]] = tag_map[item.key]

    elif item.tag == "adjacent":
        output = {item.tag_info["tag"]: tag_map[item.key], item.tag_info["content"]: output}

    return output


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
        parts: list[str] = []

        for i, word in enumerate(key.split("_")):
            if i == 0:
                parts.append(word)

            elif word:
                parts.append(word[0].upper() + word[1:])

        return "".join(parts)


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


class Model:
    _items: ClassVar[dict[str, InternalItem]]
    _type_name: ClassVar[str]

    def __init_subclass__(cls, *, type_name: str | None = None, rename: type[RenameScheme] = Default) -> None:
        items: dict[str, InternalItem] = {}

        cls._type_name = type_name or cls.__name__

        for key, annotation in cls.__annotations__.items():
            it = convert_to_item(cls, key, annotation)

            if "rename" not in it._modified and it._rename is None:
                it._rename = rename.rename(key)

            item = it._to_internal()

            if (default := getattr(cls, key, Missing)) is not Missing:
                item.default = lambda: default

            items[item.actual_key] = item

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
                    raise MissingRequiredKey(f"Missing required key {self.__class__.__name__}.{key}")

        for key, value in data.items():
            if not (item := self._items.get(key)):
                continue

            new_value = validate(item, self, value)

            setattr(self, item.key, new_value)

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}

        for item in self._items.values():
            value = getattr(self, item.key)
            output[item.actual_key] = value_to_dict(value, self.__tag_map__, item)

        return output

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        items = [f"{item.key}={getattr(self, item.key)!r}" for item in self._items.values()]

        return f"<{self.__class__.__name__} {' '.join(items)}>"


class TransparentModel(Generic[T], Model):
    value: T

    def __init_subclass__(cls, *, item: Item | None = None) -> None:
        ty = get_args(get_original_bases(cls)[0])[0]

        cls._items = {"value": convert_to_item(cls, "value", ty, item)._to_internal()}

    def __init__(self, data: Any):
        super().__init__({"value": data})

    def to_dict(self) -> dict[str, Any]:
        return value_to_dict(self.value, self.__tag_map__, self._items["value"])

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.value!r}>"


def transparent(ty: type[T] | Any, item: Item | None = None) -> type[TransparentModel[T]]:
    if is_union(ty):  # noqa: SIM108
        name = "Or".join([get_type_name(v) for v in get_args(ty)])
    else:
        name = get_type_name(ty)

    class Mod(TransparentModel[ty], item=item):
        pass

    Mod.__name__ = name

    return Mod
