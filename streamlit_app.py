"""Thin Streamlit chat UI over the typed deterministic workflow."""

from __future__ import annotations

import base64
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from agentic.explainer import ExplanationError, FactPreservingExplainer
from agentic.approval import (
    classify_run_approval,
    format_run_approval_request,
)
from agentic.boundary_presentation import (
    BoundaryCard,
    boundary_preview_svg,
    draft_boundary_cards,
    validated_boundary_cards,
)
from agentic.formulation import (
    ConversationFormulator,
    FormulationSession,
    FormulationStep,
)
from agentic.formulation_openai import (
    FormulationAPIError,
    OpenAIResponsesFormulationAgent,
)
from agentic.orchestrator import (
    AnalysisFailedWorkflow,
    AwaitingClarification,
    AwaitingRunApproval,
    CompletedWorkflow,
    DeterministicOrchestrator,
    ExplainedWorkflow,
    RunFailedWorkflow,
    UnsupportedWorkflow,
    ValidatedWorkflow,
    ValidationFailedWorkflow,
)
from agentic.presentation import verified_display_plots
from fenitop.tools.lifecycle import read_lifecycle


st.set_page_config(
    page_title="Agentic Topology Optimization",
    page_icon="🧩",
    layout="wide",
)


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="agentic-ui")


def _new_orchestrator() -> DeterministicOrchestrator:
    return DeterministicOrchestrator(
        explainer=FactPreservingExplainer.from_environment(),
    )


def _new_formulator() -> ConversationFormulator:
    return ConversationFormulator(
        OpenAIResponsesFormulationAgent.from_environment(),
        max_repair_attempts=1,
    )


def _initialize_session() -> None:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = _new_orchestrator()
    if "formulator" not in st.session_state:
        st.session_state.formulator = _new_formulator()
    if "formulation_session" not in st.session_state:
        st.session_state.formulation_session = FormulationSession()
    if "formulation_step" not in st.session_state:
        st.session_state.formulation_step = None
    if "outcome" not in st.session_state:
        st.session_state.outcome = None
    if "job_future" not in st.session_state:
        st.session_state.job_future = None
    if "job_state" not in st.session_state:
        st.session_state.job_state = None
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Describe a supported rectangular 2D topology-optimization "
                    "problem in plain language. I will ask for missing physics, "
                    "show the validated parameters, and wait for your approval "
                    "before starting the solver."
                ),
            }
        ]


def _append(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _run_and_explain(orchestrator, validated):
    terminal = orchestrator.execute(validated)
    if isinstance(terminal, CompletedWorkflow):
        try:
            return orchestrator.explain(terminal), None
        except ExplanationError as exc:
            return terminal, str(exc)
    return terminal, None


def _submit_job(validated: ValidatedWorkflow) -> None:
    st.session_state.job_state = validated
    st.session_state.job_future = _executor().submit(
        _run_and_explain,
        st.session_state.orchestrator,
        validated,
    )


def _handle_outcome(outcome) -> None:
    st.session_state.outcome = outcome
    if isinstance(outcome, AwaitingClarification):
        questions = "\n".join(f"- {question}" for question in outcome.questions)
        _append("assistant", f"I need a little more information:\n\n{questions}")
    elif isinstance(outcome, UnsupportedWorkflow):
        _append("assistant", outcome.explanation)
    elif isinstance(outcome, ValidationFailedWorkflow):
        issues = "\n".join(
            f"- `{issue.path}`: {issue.message}"
            for issue in outcome.validation.errors
        )
        _append(
            "assistant",
            "The formulated request did not pass deterministic validation:\n\n"
            + issues,
        )
    elif isinstance(outcome, AwaitingRunApproval):
        _append(
            "assistant",
            format_run_approval_request(outcome.compilation, outcome.validation),
        )


def _formulation_chat_message(step: FormulationStep) -> str:
    sections = [step.turn.assistant_message]
    if step.turn.questions:
        questions = "\n".join(
            f"- {question}" for question in step.turn.questions
        )
        sections.append("Questions to continue:\n\n" + questions)
    if step.merge.issues:
        issues = "\n".join(
            f"- `{issue.path}`: {issue.message}"
            for issue in step.merge.issues
        )
        sections.append(
            "I could not safely apply part of that interpretation after the "
            "bounded repair attempt. No run was prepared:\n\n" + issues
        )
    return "\n\n".join(sections)


def _advance_formulation(message: str) -> None:
    st.session_state.outcome = None
    step = st.session_state.formulator.advance(
        st.session_state.formulation_session,
        message,
    )
    st.session_state.formulation_session = step.session
    st.session_state.formulation_step = step
    _append("assistant", _formulation_chat_message(step))
    if step.finalized_draft is not None:
        outcome = st.session_state.orchestrator.prepare_formulation(step)
        _handle_outcome(outcome)


def _handle_user_message(message: str) -> None:
    _append("user", message)
    prior = st.session_state.outcome
    orchestrator = st.session_state.orchestrator
    try:
        if isinstance(prior, AwaitingRunApproval):
            decision = classify_run_approval(message)
            if decision == "approve":
                validated = orchestrator.approve(prior)
                st.session_state.outcome = validated
                _append(
                    "assistant",
                    "Approved. I am starting the solver run now.",
                )
                _submit_job(validated)
                return
            if decision == "reject":
                _append(
                    "assistant",
                    "The run remains stopped. Describe any parameter changes "
                    "when you are ready, or reply **yes** to approve the "
                    "current proposal.",
                )
                return

        with st.spinner("Understanding and checking the problem formulation…"):
            if isinstance(prior, AwaitingClarification):
                # Compatibility for a session serialized by the former v1 UI.
                outcome = orchestrator.resume(prior, message)
                _handle_outcome(outcome)
            else:
                # This also handles requested changes to a proposal. Clearing the
                # prior outcome first prevents a failed revision attempt from
                # leaving the older proposal approvable.
                _advance_formulation(message)
    except FormulationAPIError as exc:
        _append(
            "assistant",
            "I could not continue the problem conversation because the model "
            "service did not return a usable formulation "
            f"(`{exc.kind}` / `{exc.provider_error_type}`). No proposal was "
            "prepared and no solver was started. You can retry that message.",
        )
    except ValueError as exc:
        _append(
            "assistant",
            "I could not safely accept that formulation step: "
            f"{exc} No solver was started.",
        )
    except Exception:
        _append(
            "assistant",
            "The workflow could not start. Inspect the local application logs.",
        )


def _finish_future(future: Future) -> None:
    try:
        outcome, explanation_error = future.result()
    except Exception:
        _append(
            "assistant",
            "The background workflow failed unexpectedly. Inspect local logs.",
        )
        st.session_state.job_future = None
        return

    st.session_state.outcome = outcome
    st.session_state.job_future = None
    if isinstance(outcome, ExplainedWorkflow):
        _append("assistant", outcome.explanation.markdown)
    elif isinstance(outcome, CompletedWorkflow):
        narrative = outcome.analysis.narrative or "Deterministic analysis completed."
        suffix = (
            f"\n\n_Optional explanation unavailable: {explanation_error}_"
            if explanation_error
            else ""
        )
        _append("assistant", narrative + suffix)
    elif isinstance(outcome, RunFailedWorkflow):
        details = "\n".join(
            f"- {issue.message}" for issue in outcome.run.errors
        )
        _append("assistant", "The solver run failed.\n\n" + details)
    elif isinstance(outcome, AnalysisFailedWorkflow):
        details = "\n".join(
            f"- {issue.message}" for issue in outcome.analysis.errors
        )
        _append("assistant", "Result analysis failed.\n\n" + details)


def _render_trace(outcome) -> None:
    if outcome is None:
        return
    with st.expander("Inspectable workflow trace", expanded=False):
        for event in outcome.events:
            st.markdown(f"**{event.sequence}. `{event.stage}`** — {event.message}")

        if hasattr(outcome, "compilation"):
            st.markdown("#### Compiled agent-safe configuration")
            st.json(outcome.compilation.config.model_dump(mode="json"))
        if hasattr(outcome, "validation"):
            st.markdown("#### Validation evidence")
            evidence = {
                "status": outcome.validation.status,
                "warnings": [
                    item.model_dump(mode="json")
                    for item in outcome.validation.warnings
                ],
                "estimated_cost": (
                    outcome.validation.estimated_cost.model_dump(mode="json")
                    if outcome.validation.estimated_cost
                    else None
                ),
                "geometry_report": (
                    outcome.validation.geometry_report.model_dump(mode="json")
                    if outcome.validation.geometry_report
                    else None
                ),
            }
            st.json(evidence)
        if hasattr(outcome, "analysis"):
            st.markdown("#### Deterministic analysis evidence")
            evidence = {
                "convergence": (
                    outcome.analysis.convergence.model_dump(mode="json")
                    if outcome.analysis.convergence
                    else None
                ),
                "metrics": (
                    outcome.analysis.metrics.model_dump(mode="json")
                    if outcome.analysis.metrics
                    else None
                ),
                "quality_flags": (
                    outcome.analysis.quality_flags.model_dump(mode="json")
                    if outcome.analysis.quality_flags
                    else None
                ),
            }
            st.json(evidence)


def _render_results(outcome) -> None:
    if not hasattr(outcome, "analysis") or outcome.analysis.source is None:
        return
    plots = verified_display_plots(
        outcome.analysis.source.run_directory,
        outcome.analysis.plots,
    )
    if not plots:
        return

    st.subheader("Optimization results")
    columns = st.columns(2)
    for index, plot in enumerate(plots):
        with columns[index % len(columns)]:
            st.image(str(plot.path), caption=plot.label, width="stretch")
            st.download_button(
                f"Download {plot.label}",
                data=plot.path.read_bytes(),
                file_name=plot.path.name,
                mime="image/png",
                key=f"download_plot_{plot.role}",
            )


def _render_boundary_card(card: BoundaryCard) -> None:
    with st.container(border=True):
        st.markdown(
            f"#### {card.bc_id} · {card.title}  \n"
            f"**{card.physics}**  \n"
            f"{card.location}"
        )
        st.caption(f"Status: {card.status}")
        for detail in card.details:
            st.write(detail)
        if card.warning:
            st.warning(card.warning)
        st.caption(card.correction_hint)


def _validated_boundary_view(outcome) -> bool:
    validation = getattr(outcome, "validation", None)
    compilation = getattr(outcome, "compilation", None)
    if (
        validation is None
        or compilation is None
        or validation.status != "ok"
        or validation.geometry_report is None
    ):
        return False

    st.markdown("**Validated boundary conditions**")
    st.caption(
        "Orange dashed spans are the requested continuous locations; solid "
        "spans are the facets the mesh will actually use."
    )
    preview = boundary_preview_svg(compilation.config, validation).encode("utf-8")
    st.image(
        "data:image/svg+xml;base64," + base64.b64encode(preview).decode("ascii"),
        width="stretch",
    )
    for card in validated_boundary_cards(compilation.config, validation):
        _render_boundary_card(card)
    return True


def _render_formulation() -> None:
    session = st.session_state.formulation_session
    step = st.session_state.formulation_step
    boundary_state = session.draft.boundary_state
    outcome = st.session_state.outcome
    has_validated_boundary_view = (
        getattr(getattr(outcome, "validation", None), "status", None) == "ok"
        and getattr(outcome, "compilation", None) is not None
    )
    if (
        step is None
        and not session.draft.facts
        and not boundary_state.conditions
        and not has_validated_boundary_view
    ):
        return

    st.subheader("Current problem formulation")
    st.caption(
        "This application-owned draft—not provider conversation memory—is the "
        "current source of truth."
    )
    st.markdown(
        f"**Status:** `{session.status.replace('_', ' ')}`"
    )

    if session.draft.facts:
        st.markdown("**Accepted facts**")
        for fact in session.draft.facts:
            if fact.path in {"supports", "support_edges", "tractions"}:
                continue
            value = json.dumps(
                fact.value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            st.write(f"`{fact.path}` = `{value}` — {fact.basis}")

    ordinary_assumptions = [
        fact for fact in session.draft.facts if fact.basis == "assumption"
    ]
    boundary_assumptions = boundary_state.pending_confirmations()
    if ordinary_assumptions or boundary_assumptions:
        assumption_labels = [
            f"{fact.path}={json.dumps(fact.value, ensure_ascii=False)}"
            for fact in ordinary_assumptions
        ]
        assumption_labels.extend(
            f"{item.bc_id}.{item.field}="
            f"{json.dumps(item.value, ensure_ascii=False)}"
            for item in boundary_assumptions
        )
        st.warning(
            "These model-proposed assumptions still require your confirmation: "
            + ", ".join(assumption_labels)
        )

    shown_validated = _validated_boundary_view(outcome)
    if not shown_validated and boundary_state.conditions:
        st.markdown("**Boundary conditions being formulated**")
        st.caption(
            "Use the stable label in your next message—for example, "
            "“Change L1 to the upper half of the right edge.”"
        )
        for card in draft_boundary_cards(boundary_state):
            _render_boundary_card(card)

    if session.unsupported_features:
        st.warning(
            "Capability limits currently under discussion: "
            + ", ".join(session.unsupported_features)
        )

    if step is not None:
        if step.readiness.missing_fields:
            st.info(
                "Still needed before review: "
                + ", ".join(step.readiness.missing_fields)
            )
        for error in step.readiness.semantic_errors:
            st.error("Unresolved formulation conflict: " + error)
        for issue in step.merge.issues:
            st.error(
                f"Rejected formulation update `{issue.path}`: {issue.message}"
            )

    with st.expander(
        "Formulation provenance and revision history",
        expanded=False,
    ):
        st.caption(
            "Public source quotes and concise modeling rationales are shown; "
            "private model reasoning is not retained or displayed."
        )
        st.json(
            {
                "facts": [
                    fact.model_dump(mode="json")
                    for fact in session.draft.facts
                ],
                "revisions": [
                    revision.model_dump(mode="json")
                    for revision in session.draft.revisions
                ],
                "boundary_conditions": [
                    condition.model_dump(mode="json")
                    for condition in boundary_state.conditions
                ],
                "boundary_revisions": [
                    revision.model_dump(mode="json")
                    for revision in boundary_state.revisions
                ],
            }
        )


def _reset() -> None:
    future = st.session_state.get("job_future")
    if future is not None and not future.done():
        return
    for key in (
        "orchestrator",
        "formulator",
        "formulation_session",
        "formulation_step",
        "outcome",
        "job_future",
        "job_state",
        "messages",
    ):
        st.session_state.pop(key, None)
    st.rerun()


_initialize_session()

st.title("Agentic Topology Optimization")
st.caption(
    "Conversational LLM formulation and evidence organization; deterministic "
    "draft acceptance, compilation, validation, approval, solving, and analysis."
)

with st.sidebar:
    st.markdown("### Session")
    st.caption("The API key stays in the parent; solver workers receive no key.")
    st.button(
        "Reset conversation",
        on_click=_reset,
        disabled=(
            st.session_state.job_future is not None
            and not st.session_state.job_future.done()
        ),
    )

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])


run_every = (
    1.0
    if st.session_state.job_future is not None
    and not st.session_state.job_future.done()
    else None
)


@st.fragment(run_every=run_every)
def _job_status() -> None:
    future = st.session_state.job_future
    if future is None:
        return
    if future.done():
        _finish_future(future)
        st.rerun()

    validated = st.session_state.job_state
    identity = st.session_state.orchestrator.execution_identity(validated)
    lifecycle = None
    run_dir = Path("results") / identity.run_id
    if run_dir.is_dir():
        try:
            lifecycle = read_lifecycle(run_dir)
        except Exception:
            lifecycle = None

    with st.status("Solver job is running…", expanded=True):
        st.write(f"Run ID: `{identity.run_id}`")
        if lifecycle is not None:
            st.write(f"Lifecycle: `{lifecycle['state']}`")
            if lifecycle.get("last_iteration") is not None:
                st.write(f"Last completed iteration: {lifecycle['last_iteration']}")
        if st.button("Request cancellation", key="cancel_job"):
            accepted = st.session_state.orchestrator.request_cancel(validated)
            if accepted:
                st.warning("Cancellation requested; waiting for the worker to stop.")
            else:
                st.info("The job is not cancellable yet or is already terminal.")


_job_status()
_render_formulation()
_render_results(st.session_state.outcome)
_render_trace(st.session_state.outcome)

job_running = (
    st.session_state.job_future is not None
    and not st.session_state.job_future.done()
)
placeholder = (
    "Reply yes to start, no to stop, or describe changes"
    if isinstance(st.session_state.outcome, AwaitingRunApproval)
    else (
        "Confirm or correct the visible assumptions"
        if st.session_state.formulation_step is not None
        and st.session_state.formulation_step.readiness.unconfirmed_fields
        else (
            "Describe a supported reformulation or add missing details"
            if st.session_state.formulation_session.status == "unsupported"
            else (
                "Answer the formulation questions or add/correct details"
                if st.session_state.formulation_step is not None
                else "Describe a rectangular 2D design problem"
            )
        )
    )
)
user_message = st.chat_input(placeholder, disabled=job_running)
if user_message:
    _handle_user_message(user_message)
    st.rerun()
