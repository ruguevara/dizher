"""Generic editor for an op's params dataclass: one widget per field, driven by the type and `meta()` metadata.

    params_editor(params, on_change, id="params", help="hint")   # on_change(new_params) after any edit

int -> input_int (clamped to min/max, -/+ buttons with `step`), float -> slider when min and max are set else input_float,
bool -> checkbox, str -> combo when `choices` else input_text, Path -> text + Browse (meta editor "file"/"folder",
output paths pick folders), Optional[X] -> X with an empty value meaning None. Tuples and other types are shown
read-only. Label = meta(label) or the field name with spaces; meta(help) draws a hint under the widget, or with
help="tooltip" shows on hovering it.
"""
from dataclasses import fields, replace
from pathlib import Path
from typing import Callable, Optional

from imgui_bundle import imgui, em_size

from ..graph import _strip_annotated, _type_hints
from . import style, widgets


def field_label(f) -> str:
    return f.metadata.get("label") or f.name.replace("_", " ")


def _clamp(value, meta):
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def params_editor(params, on_change: Callable, id: str = "params", help: str = "hint") -> None:
    if params is None:
        widgets.hint("no parameters")
        return
    hints = _type_hints(type(params))
    imgui.push_id(id)
    try:
        for f in fields(params):
            _field(params, f, _strip_annotated(hints.get(f.name, f.type)), on_change, help)
    finally:
        imgui.pop_id()


def _field(params, f, tp, on_change, help: str = "hint") -> None:
    meta = f.metadata
    value = getattr(params, f.name)
    label = field_label(f)
    set_value = lambda v: on_change(replace(params, **{f.name: v}))
    if tp is bool:
        changed, new = imgui.checkbox(label, bool(value))
        if changed:
            set_value(new)
    elif tp is int:
        widgets.input_int(label, int(value or 0), lambda v: set_value(_clamp(v, meta)), step=meta.get("step", 0))
    elif tp is float:
        imgui.set_next_item_width(em_size(style.FIELD_WIDTH))
        if meta.get("min") is not None and meta.get("max") is not None:
            changed, new = imgui.slider_float(label, float(value or 0.0), float(meta["min"]), float(meta["max"]))
        else:
            changed, new = imgui.input_float(label, float(value or 0.0))
        if changed:
            set_value(_clamp(new, meta))
    elif tp is str:
        if meta.get("choices"):
            widgets.combo(label, str(value), [str(c) for c in meta["choices"]], set_value)
        else:
            widgets.input_text(label, str(value or ""), set_value)
    elif tp is Path:
        kind = meta.get("editor") or "folder"
        imgui.text(label)
        widgets.path_input(f"##{f.name}", str(value) if value else "",
                           lambda v: set_value(Path(v) if v else None), kind, label, str(value or Path.cwd()))
    else:
        imgui.text(f"{label}: {value!r}")
        widgets.hint("(not editable here)")
        return
    if meta.get("help"):
        if help == "tooltip":
            imgui.set_item_tooltip(meta["help"])
        else:
            widgets.hint(meta["help"])
