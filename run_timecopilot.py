from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _fail(message: str) -> None:
    print(f"[run_timecopilot] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    evaluator = project_root / "evaluation" / "evaluate_timecopilot.py"
    reference = project_root / "evaluation" / "evaluate_llm.py"

    print(
        f"[run_timecopilot] Python {sys.version.split()[0]}: {sys.executable}",
        flush=True,
    )
    print(f"[run_timecopilot] TemporalBench root: {project_root}", flush=True)

    if sys.version_info < (3, 10):
        _fail("TimeCopilot requires Python 3.10 or newer.")
    if not evaluator.is_file():
        _fail(f"Missing evaluator: {evaluator}")
    if not reference.is_file():
        _fail(f"Missing TemporalBench reference scorer: {reference}")

    # Ensure the repository root is importable even if this script is invoked
    # through an absolute path from another working directory.
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        package = importlib.import_module("timecopilot")
    except Exception as exc:
        _fail(
            "Could not import TimeCopilot. Activate the Python 3.10+ environment "
            "and run `pip install timecopilot` (or `pip install -e ../timecopilot` "
            f"for a local clone). Details: {exc}"
        )

    package_path = getattr(package, "__file__", None)
    package_version = getattr(package, "__version__", "unknown")
    print(
        f"[run_timecopilot] TimeCopilot package: {package_path} "
        f"(version={package_version})",
        flush=True,
    )
    if not hasattr(package, "TimeCopilot"):
        _fail(
            "The imported package does not export TimeCopilot. This usually "
            "means an old package is installed or a local timecopilot.py is "
            "shadowing the cloned repository."
        )

    # The evaluator relies on TimeCopilot.query()/clear_conversation_history(),
    # which require the agent to have completed a forecast()/analyze() run
    # before it becomes queryable. Surface a clear, early error instead of a
    # confusing per-task runtime_error if the installed version lacks them.
    agent_cls = getattr(package, "TimeCopilot")
    missing_methods = [
        name
        for name in ("forecast", "query", "clear_conversation_history")
        if not hasattr(agent_cls, name)
    ]
    if missing_methods:
        _fail(
            "The installed TimeCopilot version is missing required method(s): "
            f"{', '.join(missing_methods)}. Upgrade with "
            "`pip install -U timecopilot`."
        )

    try:
        module = importlib.import_module("evaluation.evaluate_timecopilot")
    except Exception as exc:
        _fail(f"Could not import evaluation/evaluate_timecopilot.py: {exc}")

    print("[run_timecopilot] Starting evaluation...", flush=True)
    module.main()


if __name__ == "__main__":
    main()
