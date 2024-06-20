# SPDX-FileCopyrightText: 2023-present Zomatree <me@zomatree.live>
#
# SPDX-License-Identifier: MIT

__version__ = "0.0.1"

from . import errors
from .item import *
from .model import *

__all__ = (
    # from .item
    "Item",
    "rename",
    "default",
    "validate",
    "hook",
    "tag",
    "type_name",
    # from .model
    "is_model",
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
    # .errors
    "errors",
)
