import sys


__all__ = ("TypeVar", "TypeAliasType", "get_original_bases")


if sys.version_info >= (3, 13):  # pragma: >=3.13 cover
    from typing import TypeVar
else:  # pragma: <3.13 cover
    from typing_extensions import TypeVar

if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from types import TypeAliasType, get_original_bases
else:  # pragma: <3.12 cover
    from typing_extensions import TypeAliasType, get_original_bases
