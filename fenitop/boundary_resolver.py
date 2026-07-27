"""One mesh-aware boundary selection policy for validation and FEM execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fenitop.regions import compile_region


class BoundaryResolutionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ResolvedBoundary:
    facets: np.ndarray
    node_indices: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]
    requested_extent: tuple[float, float] | None
    resolved_extent: tuple[float, float] | None
    measure: float
    centroid: tuple[float, float]
    outward_normal: tuple[float, float] | None
    resolution_error: float | None
    warning: str | None
    requested_point: tuple[float, float] | None = None
    resolved_point: tuple[float, float] | None = None


_EDGE_NORMALS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
}


def _selector_dict(selector: Any) -> dict:
    if hasattr(selector, "model_dump"):
        return selector.model_dump(mode="json")
    if isinstance(selector, dict):
        return selector
    raise BoundaryResolutionError(
        "invalid_boundary_selector",
        "Boundary selector must be a JSON object.",
    )


def _mesh_boundary_geometry(mesh):
    from dolfinx.mesh import locate_entities_boundary

    dimension = mesh.topology.dim
    facet_dimension = dimension - 1
    mesh.topology.create_connectivity(facet_dimension, 0)
    connectivity = mesh.topology.connectivity(facet_dimension, 0)
    boundary_facets = locate_entities_boundary(
        mesh,
        facet_dimension,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    return facet_dimension, connectivity, boundary_facets


def _facet_data(mesh, connectivity, facets: np.ndarray):
    facet_nodes: list[np.ndarray] = []
    midpoints: list[np.ndarray] = []
    lengths: list[float] = []
    for facet in facets:
        nodes = np.asarray(connectivity.links(int(facet)), dtype=np.int32)
        coords = np.asarray(mesh.geometry.x[nodes, :2], dtype=float)
        facet_nodes.append(nodes)
        midpoints.append(coords.mean(axis=0))
        if len(coords) < 2:
            lengths.append(0.0)
        else:
            lengths.append(float(np.linalg.norm(coords[-1] - coords[0])))
    return facet_nodes, np.asarray(midpoints), np.asarray(lengths)


def _bounds(coords: np.ndarray):
    if not len(coords):
        return ((0.0, 0.0), (0.0, 0.0))
    return (
        (float(coords[:, 0].min()), float(coords[:, 1].min())),
        (float(coords[:, 0].max()), float(coords[:, 1].max())),
    )


def _summarize(
    mesh,
    connectivity,
    facets: np.ndarray,
    *,
    requested_extent: tuple[float, float] | None,
    resolved_extent: tuple[float, float] | None,
    outward_normal: tuple[float, float] | None,
    resolution_error: float | None,
    warning: str | None,
) -> ResolvedBoundary:
    facet_nodes, midpoints, lengths = _facet_data(mesh, connectivity, facets)
    nodes = (
        np.unique(np.concatenate(facet_nodes)).astype(np.int32)
        if facet_nodes
        else np.asarray([], dtype=np.int32)
    )
    coords = (
        np.asarray(mesh.geometry.x[nodes, :2], dtype=float)
        if nodes.size
        else np.empty((0, 2))
    )
    measure = float(lengths.sum())
    if measure <= 0 or not math.isfinite(measure):
        raise BoundaryResolutionError(
            "boundary_has_zero_measure",
            "Selected boundary facets have zero or invalid physical measure.",
        )
    centroid_array = np.average(midpoints, axis=0, weights=lengths)
    return ResolvedBoundary(
        facets=np.asarray(facets, dtype=np.int32),
        node_indices=nodes,
        bounds=_bounds(coords),
        requested_extent=requested_extent,
        resolved_extent=resolved_extent,
        measure=measure,
        centroid=(float(centroid_array[0]), float(centroid_array[1])),
        outward_normal=outward_normal,
        resolution_error=resolution_error,
        warning=warning,
    )


def _edge_requested_extent(
    selector: dict,
    mesh_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    edge = selector["edge"]
    interval = selector["interval"]
    (x0, y0), (x1, y1) = mesh_bounds
    low, high = (y0, y1) if edge in {"left", "right"} else (x0, x1)
    start, end = float(interval["start"]), float(interval["end"])
    if interval["kind"] == "fraction":
        return (
            low + start * (high - low),
            low + end * (high - low),
        )
    tolerance = 1e-10 * max(1.0, high - low)
    if start < low - tolerance or end > high + tolerance:
        raise BoundaryResolutionError(
            "selector_outside_edge",
            f"Coordinate interval [{start:g}, {end:g}] lies outside "
            f"the {edge} edge extent [{low:g}, {high:g}].",
        )
    return max(start, low), min(end, high)


def _resolve_rectangle_edge(
    mesh,
    selector: dict,
    connectivity,
    boundary_facets: np.ndarray,
    mesh_bounds,
) -> ResolvedBoundary:
    edge = selector["edge"]
    requested = _edge_requested_extent(selector, mesh_bounds)
    (x0, y0), (x1, y1) = mesh_bounds
    normal_axis = 0 if edge in {"left", "right"} else 1
    along_axis = 1 - normal_axis
    edge_coordinate = {
        "left": x0,
        "right": x1,
        "bottom": y0,
        "top": y1,
    }[edge]
    tolerance = 1e-9 * max(1.0, x1 - x0, y1 - y0)

    edge_facets: list[int] = []
    midpoint_positions: list[float] = []
    facet_extents: list[tuple[float, float]] = []
    for facet in boundary_facets:
        nodes = np.asarray(connectivity.links(int(facet)), dtype=np.int32)
        coords = np.asarray(mesh.geometry.x[nodes, :2], dtype=float)
        if np.all(np.isclose(
            coords[:, normal_axis],
            edge_coordinate,
            rtol=0.0,
            atol=tolerance,
        )):
            edge_facets.append(int(facet))
            midpoint_positions.append(float(coords[:, along_axis].mean()))
            facet_extents.append((
                float(coords[:, along_axis].min()),
                float(coords[:, along_axis].max()),
            ))

    if not edge_facets:
        raise BoundaryResolutionError(
            "rectangle_edge_has_no_facets",
            f"No boundary facets were found on the named {edge} edge.",
        )
    order = np.argsort(np.asarray(midpoint_positions))
    ordered_facets = np.asarray(edge_facets, dtype=np.int32)[order]
    ordered_midpoints = np.asarray(midpoint_positions)[order]
    ordered_extents = np.asarray(facet_extents)[order]
    start, end = requested
    selected_mask = (
        (ordered_midpoints >= start - tolerance)
        & (ordered_midpoints <= end + tolerance)
    )
    warning = None
    if not selected_mask.any() and end > start:
        center = 0.5 * (start + end)
        closest = int(np.argmin(np.abs(ordered_midpoints - center)))
        selected_mask[closest] = True
        warning = (
            f"Requested positive interval [{start:g}, {end:g}] contained no "
            "facet midpoint; selected the single closest facet."
        )
    selected_facets = ordered_facets[selected_mask]
    selected_extents = ordered_extents[selected_mask]
    if not selected_facets.size:
        raise BoundaryResolutionError(
            "selector_matches_no_facets",
            f"Requested interval [{start:g}, {end:g}] matched no facet midpoint.",
        )
    resolved = (
        float(selected_extents[:, 0].min()),
        float(selected_extents[:, 1].max()),
    )
    error = max(abs(resolved[0] - start), abs(resolved[1] - end))
    return _summarize(
        mesh,
        connectivity,
        selected_facets,
        requested_extent=requested,
        resolved_extent=resolved,
        outward_normal=_EDGE_NORMALS[edge],
        resolution_error=float(error),
        warning=warning,
    )


def _infer_expert_normal(bounds, mesh_bounds, tolerance):
    (bx0, by0), (bx1, by1) = bounds
    (x0, y0), (x1, y1) = mesh_bounds
    candidates = []
    if abs(bx0 - x0) <= tolerance and abs(bx1 - x0) <= tolerance:
        candidates.append(_EDGE_NORMALS["left"])
    if abs(bx0 - x1) <= tolerance and abs(bx1 - x1) <= tolerance:
        candidates.append(_EDGE_NORMALS["right"])
    if abs(by0 - y0) <= tolerance and abs(by1 - y0) <= tolerance:
        candidates.append(_EDGE_NORMALS["bottom"])
    if abs(by0 - y1) <= tolerance and abs(by1 - y1) <= tolerance:
        candidates.append(_EDGE_NORMALS["top"])
    return candidates[0] if len(candidates) == 1 else None


def _resolve_boundary_node(
    mesh,
    selector: dict,
    connectivity,
    boundary_facets: np.ndarray,
    mesh_bounds,
) -> ResolvedBoundary:
    requested = np.asarray(selector["point"], dtype=float)
    (x0, y0), (x1, y1) = mesh_bounds
    tolerance = 1e-9 * max(1.0, x1 - x0, y1 - y0)
    on_boundary = (
        x0 - tolerance <= requested[0] <= x1 + tolerance
        and y0 - tolerance <= requested[1] <= y1 + tolerance
        and (
            abs(requested[0] - x0) <= tolerance
            or abs(requested[0] - x1) <= tolerance
            or abs(requested[1] - y0) <= tolerance
            or abs(requested[1] - y1) <= tolerance
        )
    )
    if not on_boundary:
        raise BoundaryResolutionError(
            "point_not_on_boundary",
            f"Requested point [{requested[0]:g}, {requested[1]:g}] is not on "
            "the rectangular domain boundary.",
        )

    facet_nodes = [
        np.asarray(connectivity.links(int(facet)), dtype=np.int32)
        for facet in boundary_facets
    ]
    nodes = np.unique(np.concatenate(facet_nodes)).astype(np.int32)
    coordinates = np.asarray(mesh.geometry.x[nodes, :2], dtype=float)
    distances = np.linalg.norm(coordinates - requested, axis=1)
    closest = int(np.argmin(distances))
    node = int(nodes[closest])
    resolved = coordinates[closest]
    distance = float(distances[closest])
    incident = np.asarray(
        [
            int(facet)
            for facet, current_nodes in zip(boundary_facets, facet_nodes)
            if node in current_nodes
        ],
        dtype=np.int32,
    )
    warning = (
        None
        if distance <= tolerance
        else (
            f"Requested boundary point [{requested[0]:g}, {requested[1]:g}] "
            f"snapped {distance:.6g} to mesh node "
            f"[{resolved[0]:g}, {resolved[1]:g}]."
        )
    )
    point = (float(resolved[0]), float(resolved[1]))
    return ResolvedBoundary(
        facets=incident,
        node_indices=np.asarray([node], dtype=np.int32),
        bounds=(point, point),
        requested_extent=None,
        resolved_extent=None,
        measure=0.0,
        centroid=point,
        outward_normal=None,
        resolution_error=distance,
        warning=warning,
        requested_point=(float(requested[0]), float(requested[1])),
        resolved_point=point,
    )


def resolve_boundary(mesh, selector: Any) -> ResolvedBoundary:
    """Resolve an expert region or semantic rectangle-edge interval once."""
    from dolfinx.mesh import locate_entities_boundary

    spec = _selector_dict(selector)
    facet_dimension, connectivity, boundary_facets = _mesh_boundary_geometry(mesh)
    mesh_coords = np.asarray(mesh.geometry.x[:, :2], dtype=float)
    mesh_bounds = _bounds(mesh_coords)
    if spec.get("kind") == "boundary_node":
        return _resolve_boundary_node(
            mesh,
            spec,
            connectivity,
            boundary_facets,
            mesh_bounds,
        )
    if spec.get("kind") == "rectangle_edge":
        return _resolve_rectangle_edge(
            mesh,
            spec,
            connectivity,
            boundary_facets,
            mesh_bounds,
        )
    if spec.get("kind") != "expert_region":
        raise BoundaryResolutionError(
            "invalid_boundary_selector",
            f"Unsupported boundary selector kind {spec.get('kind')!r}.",
        )
    try:
        facets = locate_entities_boundary(
            mesh,
            facet_dimension,
            compile_region(spec["region"]),
        ).astype(np.int32)
    except Exception as exc:
        raise BoundaryResolutionError(
            "geometry_marker_error",
            f"Region failed while matching boundary facets: {exc}",
        ) from exc
    if not facets.size:
        raise BoundaryResolutionError(
            "region_matches_no_facets",
            f"Region matched 0 of {len(boundary_facets)} boundary facets.",
        )
    nodes = np.unique(np.concatenate([
        np.asarray(connectivity.links(int(facet)), dtype=np.int32)
        for facet in facets
    ]))
    selected_bounds = _bounds(np.asarray(mesh.geometry.x[nodes, :2], dtype=float))
    (x0, y0), (x1, y1) = mesh_bounds
    tolerance = 1e-9 * max(1.0, x1 - x0, y1 - y0)
    return _summarize(
        mesh,
        connectivity,
        facets,
        requested_extent=None,
        resolved_extent=None,
        outward_normal=_infer_expert_normal(
            selected_bounds, mesh_bounds, tolerance
        ),
        resolution_error=None,
        warning=None,
    )
