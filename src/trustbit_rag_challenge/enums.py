from enum import Enum


class QuestionKind(str, Enum):
    NUMBER = "number"
    NAME = "name"
    BOOLEAN = "boolean"
    NAMES = "names"
    COMPARATIVE = "comparative"
