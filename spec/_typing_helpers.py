# The code below is modified from discord.py, published at https://github.com/Rapptz/discord.py/blob/v2.4.0/discord/utils.py#L1098-L1187.
# It was made available under MIT and is copyright 2015-present Rapptz.
# The license in its original form may be found at https://github.com/Rapptz/discord.py/blob/v2.4.0/LICENSE
# and is also included in this repository's `LICENSE` file.

import sys
import types
from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Any, ForwardRef, Literal, Union

if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from typing import TypeAliasType
else:  # pragma: <3.12 cover
    from typing_extensions import TypeAliasType

__all__ = ("resolve_annotation", "evaluate_annotation")


def _normalise_optional_params(parameters: Iterable[Any]) -> tuple[Any, ...]:
    return (*[p for p in parameters if p is not types.NoneType], types.NoneType)


def evaluate_annotation(  # noqa: PLR0911
    tp: Any,
    globalns: dict[str, Any],
    localns: dict[str, Any],
    cache: dict[str, Any],
    *,
    implicit_str: bool = True,
    with_extras: bool = True,
) -> Any:
    if isinstance(tp, ForwardRef):
        tp = tp.__forward_arg__
        # ForwardRefs always evaluate their internals
        implicit_str = True

    if implicit_str and isinstance(tp, str):
        try:
            return cache[tp]
        except KeyError:
            cache[tp] = evaluated = evaluate_annotation(
                eval(tp, globalns, localns),  # noqa: S307
                globalns,
                localns,
                cache,
                with_extras=with_extras,
            )
            return evaluated

    if (  # pragma: >=3.12 cover
        sys.version_info >= (3, 12) and getattr(tp.__repr__, "__objclass__", None) is TypeAliasType
    ):
        temp_locals: dict[str, Any] = localns | {t.__name__: t for t in tp.__type_params__}
        annotation = evaluate_annotation(tp.__value__, globalns, temp_locals, cache.copy(), with_extras=with_extras)
        if hasattr(tp, "__args__"):
            annotation = annotation[tp.__args__]
        return annotation

    if hasattr(tp, "__supertype__"):
        return evaluate_annotation(tp.__supertype__, globalns, localns, cache, with_extras=with_extras)

    if hasattr(tp, "__metadata__"):
        # Annotated[X, Y] can access Y via __metadata__
        origin = tp.__origin__
        metadata = tp.__metadata__
        return Annotated[(evaluate_annotation(origin, globalns, localns, cache, with_extras=with_extras), *metadata)]

    if hasattr(tp, "__args__"):
        implicit_str = True
        is_literal = False
        args: tuple[Any, ...] = tp.__args__
        if not hasattr(tp, "__origin__"):
            if type(tp) is types.UnionType:
                converted = Union[args]  # type: ignore  # noqa: UP007
                return evaluate_annotation(converted, globalns, localns, cache, with_extras=with_extras)

            return tp  # pragma: no cover # Nothing in the standard library meets this case yet.
        if tp.__origin__ is Union:
            try:
                none_index = args.index(types.NoneType)
            except ValueError:
                pass
            else:
                if none_index != (len(args) - 1):
                    args = _normalise_optional_params(tp.__args__)
        if tp.__origin__ is Literal:
            implicit_str = False
            is_literal = True

        evaluated_args = tuple(
            [
                evaluate_annotation(arg, globalns, localns, cache, implicit_str=implicit_str, with_extras=with_extras)
                for arg in args
            ]
        )

        if is_literal and not all(isinstance(x, (str, bytes, int, bool, Enum, types.NoneType)) for x in evaluated_args):
            msg = "Literal arguments must be of type str, bytes, int, bool, Enum, or NoneType."
            raise TypeError(msg)

        try:
            return tp.copy_with(evaluated_args)
        except AttributeError:
            return tp.__origin__[evaluated_args]

    return tp


def resolve_annotation(
    annotation: Any,
    globalns: dict[str, Any],
    localns: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    *,
    with_extras: bool = True,
) -> Any:
    if annotation is None:
        return types.NoneType
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)

    localns = globalns if localns is None else localns
    if cache is None:
        cache = {}
    return evaluate_annotation(annotation, globalns, localns, cache, with_extras=with_extras)
