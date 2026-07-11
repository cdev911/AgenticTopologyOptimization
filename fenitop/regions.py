"""Declarative region DSL for boundary/load/zone markers.

Compiles a JSON-safe region spec (a plain dict, as it would appear inside a
JSON config file) into a callable matching the `f(x) -> bool array` convention
already used throughout fenitop for Dirichlet/traction markers, solid/void
zones, and mechanism spring locators: `x` has shape (gdim-or-more, N) (dolfinx
always carries 3 coordinate rows internally, even for 2D meshes) and the
return value is a boolean array of shape (N,).

This exists so an agent authoring a config from natural language can emit
structured JSON (e.g. {"op": "plane", "axis": "x", "value": 0}) instead of
having to generate syntactically-valid Python lambda source, which is both an
easier target for an LLM and closes off the code-execution surface that
`fenitop.config._materialize()`'s `eval()` of lambda strings otherwise opens.
Raw lambda strings remain fully supported for backward compatibility; this is
an additional, preferred authoring surface, not a replacement.
"""
import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}
_KNOWN_OPS = {"plane", "range", "circle", "all", "none", "and", "or", "not"}


class RegionError(ValueError):
    """Raised when a region spec is malformed or references an unknown op."""


def is_region_spec(value):
    """True if `value` looks like a region-DSL dict (has a known 'op' key).

    Checking membership in the known op set (rather than just "has an 'op'
    key") avoids false-triggering on some unrelated config dict that happens
    to use "op" as a key name for something else.
    """
    return isinstance(value, dict) and value.get("op") in _KNOWN_OPS


def _require(spec, key, op):
    if key not in spec:
        raise RegionError(f"Region op '{op}' requires a '{key}' field; got {spec!r}.")
    return spec[key]


def _axis_index(axis, op):
    if axis not in _AXES:
        raise RegionError(
            f"Region op '{op}' has unknown axis {axis!r}; expected one of {sorted(_AXES)}.")
    return _AXES[axis]


def compile_region(spec):
    """Compile a region spec dict into a callable f(x) -> bool ndarray.

    Supported ops:
      {"op": "plane", "axis": "x"|"y"|"z", "value": number, "tol": number=1e-8}
      {"op": "range", "axis": ..., "min": number, "max": number,
       "min_inclusive": bool=True, "max_inclusive": bool=True}
      {"op": "circle", "center": [x, y] or [x, y, z], "radius": number,
       "inside": bool=True}
      {"op": "all"}    -- matches every point
      {"op": "none"}   -- matches no point
      {"op": "and", "regions": [spec, ...]}
      {"op": "or",  "regions": [spec, ...]}
      {"op": "not", "region": spec}
    """
    if not isinstance(spec, dict):
        raise RegionError(f"Region spec must be a dict; got {spec!r}.")
    op = spec.get("op")
    if op not in _KNOWN_OPS:
        raise RegionError(f"Unknown region op {op!r}; expected one of {sorted(_KNOWN_OPS)}.")

    if op == "plane":
        axis = _axis_index(_require(spec, "axis", op), op)
        value = float(_require(spec, "value", op))
        tol = float(spec.get("tol", 1e-8))
        return lambda x: np.isclose(x[axis], value, atol=tol)

    if op == "range":
        axis = _axis_index(_require(spec, "axis", op), op)
        lo = float(_require(spec, "min", op))
        hi = float(_require(spec, "max", op))
        if hi < lo:
            raise RegionError(f"Region op 'range' needs max >= min; got min={lo}, max={hi}.")
        lo_inclusive = bool(spec.get("min_inclusive", True))
        hi_inclusive = bool(spec.get("max_inclusive", True))

        def region_range(x):
            v = x[axis]
            lo_ok = v >= lo if lo_inclusive else v > lo
            hi_ok = v <= hi if hi_inclusive else v < hi
            return lo_ok & hi_ok
        return region_range

    if op == "circle":
        center = np.asarray(_require(spec, "center", op), dtype=float)
        radius = float(_require(spec, "radius", op))
        if radius < 0:
            raise RegionError(f"Region op 'circle' needs radius >= 0; got {radius}.")
        inside = bool(spec.get("inside", True))
        dim = center.size

        def region_circle(x):
            diff = x[:dim] - center.reshape(-1, 1)
            dist2 = np.sum(diff**2, axis=0)
            return dist2 <= radius**2 if inside else dist2 > radius**2
        return region_circle

    if op == "all":
        return lambda x: np.full(x.shape[1], True)

    if op == "none":
        return lambda x: np.full(x.shape[1], False)

    if op == "and":
        subregions = [compile_region(s) for s in _require(spec, "regions", op)]
        if not subregions:
            raise RegionError("Region op 'and' needs a non-empty 'regions' list.")

        def region_and(x):
            result = np.full(x.shape[1], True)
            for sub in subregions:
                result &= sub(x)
            return result
        return region_and

    if op == "or":
        subregions = [compile_region(s) for s in _require(spec, "regions", op)]
        if not subregions:
            raise RegionError("Region op 'or' needs a non-empty 'regions' list.")

        def region_or(x):
            result = np.full(x.shape[1], False)
            for sub in subregions:
                result |= sub(x)
            return result
        return region_or

    # op == "not"
    subregion = compile_region(_require(spec, "region", op))
    return lambda x: ~subregion(x)
