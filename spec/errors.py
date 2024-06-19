__all__ = (
    "SpecError",
    "MissingArgument",
    "MissingRequiredKey",
    "InvalidType",
    "FailedValidation",
    "UnknownUnionKey",
    "MissingTypeName",
)


class SpecError(Exception):
    """Base exception class for Spec."""


class MissingArgument(SpecError):
    """Exception that's raised when a Model is instantiated with no arguments."""


class MissingRequiredKey(SpecError):
    """Exception that's raised when a Model is instantiated without a required key."""


class InvalidType(SpecError):
    pass


class FailedValidation(SpecError):
    pass


class UnknownUnionKey(SpecError):
    pass


class MissingTypeName(SpecError):
    pass
