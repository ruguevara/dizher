# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, List, Tuple


class Parameter:
    def __init__(self, default: float=0.0, range: Tuple[float, float]=(-4.0, 4.0)) -> None:
        self.default = default
        self.range = range

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name
        if not getattr(owner, 'defaults', None):
            owner.defaults = dict()
        if not getattr(owner, 'ranges', None):
            owner.ranges = dict()
        owner.defaults[name] = self.default
        owner.ranges[name] = self.range

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name, self.default)
        return value

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class ParamSet:
    defaults = {}
    ranges = {}

    @classmethod
    def get_default(cls, name: str) -> Any:
        return cls.defaults.get(name)

    @classmethod
    def get_range(cls, name: str) -> Any:
        return cls.ranges.get(name)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return repr(self.__dict__)
