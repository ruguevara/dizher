# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class Actionable:
    name: str


@dataclass(frozen=True)
class Parameter(Actionable):
    type: type
    default: Any = None
    description: str = ''


@dataclass(frozen=True)
class NumParameter(Parameter):
    range: tuple = ()


@dataclass(frozen=True)
class FloatParameter(Parameter):
    range: tuple = ()


@dataclass(frozen=True)
class EnumParameter(Parameter):
    choices: List[str] = []
