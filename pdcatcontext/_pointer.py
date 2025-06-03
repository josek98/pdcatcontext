from typing import Any, Optional
from typing_extensions import TypeAlias
from types import FrameType
import ctypes

def _locals_to_fast(frame):
    """Updates the fast locals from the frame's f_locals using CPython's API."""
    ctypes.pythonapi.PyFrame_LocalsToFast(
        ctypes.py_object(frame),
        ctypes.c_int(1)  # Use 1 to update both variables and cell vars
    )

PointerName: TypeAlias = str

class Pointer:
    """Pointer class for referencing variables in a specific scope.
    This class allows you to create a pointer to a variable by its name,
    and dereference it to access or modify the variable's value.
    """
    GLOBALS: dict[str, Any] = globals()
    FRAME: Optional[FrameType] = None  # Track the frame for local variables

    @classmethod
    def set_globals(cls, globals_: dict[str, Any], frame: Optional[FrameType] = None) -> None:
        cls.GLOBALS = globals_
        cls.FRAME = frame

    def __init__(self, pointer_name: PointerName):
        self._pointer_name = pointer_name

    @property
    def dereference(self) -> Any:
        if Pointer.FRAME is not None:
            obj = Pointer.FRAME.f_locals.get(self._pointer_name)
        else:
            obj = Pointer.GLOBALS.get(self._pointer_name)
        if obj is None:
            raise ValueError(f"Object '{self._pointer_name}' not found")
        return obj

    @dereference.setter
    def dereference(self, value: Any) -> None:
        if Pointer.FRAME is not None:
            Pointer.FRAME.f_locals[self._pointer_name] = value
            _locals_to_fast(Pointer.FRAME)
        else:
            Pointer.GLOBALS[self._pointer_name] = value

    def arrow(self, attr) -> Any:
        """To get attributes from the object that the pointer points to"""

        obj = self.dereference
        return getattr(obj, attr)

    def __getattr__(self, attr) -> Any:
        return self.arrow(attr)
