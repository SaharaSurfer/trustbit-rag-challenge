from enum import Enum


class QuestionKind(str, Enum):
    """
    Enumeration of supported question types.

    Attributes
    ----------
    NUMBER : str
        Represents questions requiring a specific numerical value (e.g.,
        currency, percentage, count).
    NAME : str
        Represents questions asking for a specific entity name (e.g., CEO name,
        product name).
    BOOLEAN : str
        Represents questions requiring a True/False answer (e.g., existence of
        a specific policy).
    NAMES : str
        Represents questions asking for a list of entities or items.
    COMPARATIVE : str
        Represents complex questions comparing metrics across multiple
        companies.
    """

    NUMBER = "number"
    NAME = "name"
    BOOLEAN = "boolean"
    NAMES = "names"
    COMPARATIVE = "comparative"


class UnitScale(str, Enum):
    ONES = "ones"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"
