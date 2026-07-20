"""Thin Streamlit chat UI over the typed deterministic workflow."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from agentic.explainer import ExplanationError, FactPreservingExplainer
from agentic.interpreter import IntentInterpreter, InterpretationError
from agentic.approval import (
    classify_run_approval,
    format_run_approval_request,
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
        IntentInterpreter.from_environment(),
        explainer=FactPreservingExplainer.from_environment(),
    )


def _initialize_session() -> None:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = _new_orchestrator()
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
            "The interpreted request did not pass deterministic validation:\n\n"
            + issues,
        )
    elif isinstance(outcome, AwaitingRunApproval):
        _append(
            "assistant",
            format_run_approval_request(outcome.compilation, outcome.validation),
        )


def _handle_user_message(message: str) -> None:
    _append("user", message)
    prior = st.session_state.outcome
    orchestrator = st.session_state.orchestrator
    try:
        with st.spinner("Interpreting and validating the request…"):
            if isinstance(prior, AwaitingClarification):
                outcome = orchestrator.resume(prior, message)
            elif isinstance(prior, AwaitingRunApproval):
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
                outcome = orchestrator.revise(prior, message)
            else:
                outcome = orchestrator.start(message)
        _handle_outcome(outcome)
    except (InterpretationError, ValueError) as exc:
        _append("assistant", f"I could not interpret that request: {exc}")
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


def _reset() -> None:
    future = st.session_state.get("job_future")
    if future is not None and not future.done():
        return
    for key in ("orchestrator", "outcome", "job_future", "job_state", "messages"):
        st.session_state.pop(key, None)
    st.rerun()


_initialize_session()

st.title("Agentic Topology Optimization")
st.caption(
    "LLM interpretation and evidence organization; deterministic compilation, "
    "validation, solving, and analysis."
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
_render_results(st.session_state.outcome)
_render_trace(st.session_state.outcome)

job_running = (
    st.session_state.job_future is not None
    and not st.session_state.job_future.done()
)
placeholder = (
    "Answer the clarification question"
    if isinstance(st.session_state.outcome, AwaitingClarification)
    else (
        "Reply yes to start, no to stop, or describe changes"
        if isinstance(st.session_state.outcome, AwaitingRunApproval)
        else "Describe a rectangular 2D design problem"
    )
)
user_message = st.chat_input(placeholder, disabled=job_running)
if user_message:
    _handle_user_message(user_message)
    st.rerun()
