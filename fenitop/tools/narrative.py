"""Deterministic, template-based English summaries for Tool 3.

Explicitly NOT a second LLM call -- plain conditional composition over
already-computed structured fields, so identical inputs always produce
identical text and the result can be unit-tested with keyword assertions
instead of anything fuzzy.
"""
from typing import Any, Dict, List, Optional

_PROBLEM_TYPE_LABELS = {
    "minimize_compliance": "minimize-compliance",
    "compliant_mechanism": "compliant-mechanism",
}


def _fmt(value: Optional[float], spec: str = ".4g") -> str:
    return format(value, spec) if value is not None else "unknown"


def _capitalize_first(text: str) -> str:
    """Capitalize only the first character -- unlike str.capitalize(), never
    lowercases the rest of the string."""
    return text[:1].upper() + text[1:] if text else text


def build_narrative(convergence: Dict[str, Any], quality_flags: Dict[str, Any],
                     metrics: Dict[str, Any], problem_type: str) -> str:
    sentences: List[str] = []

    label = _PROBLEM_TYPE_LABELS.get(problem_type, problem_type)
    iterations = convergence.get("iterations")
    sentences.append(f"Ran a {label} topology optimization for {iterations} iterations.")

    converged = convergence.get("converged")
    final_change = convergence.get("final_change")
    opt_tol = convergence.get("opt_tol")
    if converged:
        sentences.append(
            f"The optimizer converged: the design change ({_fmt(final_change, '.2e')}) reached "
            f"the tolerance ({_fmt(opt_tol, '.2e')}).")
    else:
        sentences.append(
            "The optimizer did NOT converge: it stopped after reaching the iteration cap with "
            f"the design change still at {_fmt(final_change, '.2e')} (tolerance is {_fmt(opt_tol, '.2e')}).")
        fraction_pinned = convergence.get("fraction_iterations_at_move_limit")
        move_limit = convergence.get("move_limit")
        if fraction_pinned is not None and fraction_pinned > 0.5:
            sentences.append(
                f"The design update was pinned at the move limit ({_fmt(move_limit, 'g')}) for "
                f"{fraction_pinned:.1%} of iterations, suggesting the move limit, beta schedule, "
                "or iteration budget may need adjustment.")

    grayness = quality_flags.get("grayness")
    low_grayness = quality_flags.get("low_grayness_warning")
    if grayness is not None:
        if low_grayness:
            sentences.append(
                "The final design has substantial gray (intermediate-density) material "
                f"(grayness={_fmt(grayness, '.3g')}), meaning the topology is not yet cleanly "
                "binarized -- consider a higher beta_max or a longer beta continuation schedule.")
        elif not converged:
            sentences.append(
                f"The final design is reasonably well-binarized (grayness={_fmt(grayness, '.3g')}), "
                "so this looks like an optimizer-tolerance issue rather than a design-quality problem.")
        else:
            sentences.append(f"The final design is well-binarized (grayness={_fmt(grayness, '.3g')}).")

    if quality_flags.get("has_disconnected_material"):
        num_components = quality_flags.get("num_components")
        largest_fraction = quality_flags.get("largest_component_fraction")
        sentences.append(
            f"Warning: the density field has {num_components} disconnected material regions "
            f"(the largest holds {_fmt(largest_fraction, '.1%')} of the material) -- this usually "
            "indicates floating material that isn't structurally load-bearing.")

    if quality_flags.get("load_path_connected") is False:
        sentences.append(
            "Warning: the support and load regions do not appear to be connected through the "
            "same piece of material -- the structure may not actually carry the applied load.")

    if quality_flags.get("checkerboard_detected"):
        sentences.append(
            "A checkerboard-like density pattern was detected, which can indicate a numerical "
            "instability in the density field rather than a genuine design feature.")

    compliance = metrics.get("final_compliance")
    volume = metrics.get("final_volume")
    vol_target = metrics.get("vol_frac_target")
    tail = []
    if compliance is not None:
        tail.append(f"final compliance={_fmt(compliance, '.4g')}")
    if volume is not None:
        vol_str = f"volume fraction={_fmt(volume, '.4g')}"
        if vol_target is not None:
            vol_str += f" (target {_fmt(vol_target, 'g')})"
        tail.append(vol_str)
    if tail:
        sentences.append(_capitalize_first(", ".join(tail)) + ".")

    return " ".join(sentences)
