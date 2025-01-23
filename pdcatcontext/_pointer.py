from typing import Any
from typing_extensions import TypeAlias

PointerName: TypeAlias = str


class Pointer:
    """This class abstracts and implementation of pointers in Python. Given the string name of
    a variable, this class can look for the object in globals with 'dereference' property or get attributes
    of the object like -> operation in C/C++ does."""

    GLOBALS: dict[str, Any] = globals()

    @classmethod
    def set_globals(cls, globals_: dict[str, Any]) -> None:
        cls.GLOBALS = globals_

    def __init__(self, pointer_name: PointerName):
        self._pointer_name = pointer_name

    @property
    def dereference(self) -> Any:
        """Call to derreference the pointer"""
        obj = self.GLOBALS.get(self._pointer_name)
        if obj is None:
            raise ValueError(f"Object '{self._pointer_name}' not found")
        return obj

    @dereference.setter
    def dereference(self, value: Any) -> None:
        self.GLOBALS[self._pointer_name] = value

    def arrow(self, attr) -> Any:
        """To get attributes from the object that the pointer points to"""

        obj = self.GLOBALS.get(self._pointer_name)
        if obj is None:
            raise AttributeError(f"Object '{self._pointer_name}' not found")

        return getattr(obj, attr)

    def __getattr__(self, attr) -> Any:
        return self.arrow(attr)
