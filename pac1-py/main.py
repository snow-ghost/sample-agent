import os
import json
import sys
import textwrap
import time
from contextlib import redirect_stdout
from pathlib import Path

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    RunState,
    StartPlaygroundRequest,
    StartRunRequest,
    StartTrialRequest,
    SubmitRunRequest,
    StatusRequest,
)
from connectrpc.errors import ConnectError

from agent import run_agent
from pac1_agent.telemetry import AgentRunTelemetry

BITGN_URL = os.getenv("BENCHMARK_HOST") or "https://api.bitgn.com"
BENCHMARK_ID = os.getenv("BENCHMARK_ID") or "bitgn/pac1-dev"
MODEL_ID = os.getenv("MODEL_ID") or "gpt-4.1-2025-04-14"
BITGN_API_KEY = os.getenv("BITGN_API_KEY")
BITGN_RUN_NAME = os.getenv("BITGN_RUN_NAME") or "pac1-py-run"

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"


class _TeeStdout:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _render_summary_table(task_rows: list[dict[str, str | int | float]]) -> str:
    headers = [
        ("task_id", "Task"),
        ("score", "Score"),
        ("total_tokens", "Tokens"),
        ("llm_calls", "LLM"),
        ("wall_time_ms", "Wall ms"),
        ("llm_time_ms", "LLM ms"),
    ]
    widths: dict[str, int] = {}
    for key, title in headers:
        widths[key] = len(title)
        for row in task_rows:
            widths[key] = max(widths[key], len(str(row[key])))

    def fmt_row(row: dict[str, str | int | float] | None = None) -> str:
        cells: list[str] = []
        for key, title in headers:
            value = title if row is None else str(row[key])
            cells.append(value.rjust(widths[key]) if key != "task_id" else value.ljust(widths[key]))
        return " | ".join(cells)

    separator = "-+-".join(
        ("-" * widths[key]) for key, _ in headers
    )
    lines = [fmt_row(), separator]
    for row in task_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _build_totals(task_rows: list[dict[str, str | int | float]], final_score: float) -> dict[str, int | float]:
    task_count = len(task_rows) or 1
    scored_rows = [row for row in task_rows if str(row["score"]).strip() not in {"N/A", "NA", "", "None", "ERR"}]
    llm_task_count = sum(1 for row in task_rows if int(row["llm_calls"]) > 0)
    llm_divisor = llm_task_count or 1
    scored_failed_rows = [row for row in scored_rows if row["score"] != "1.00"]
    return {
        "tasks_run": len(task_rows),
        "tasks_passed": sum(1 for row in task_rows if row["score"] == "1.00"),
        "tasks_failed": len(scored_failed_rows),
        "scores_available": 1 if scored_rows else 0,
        "scored_tasks_count": len(scored_rows),
        "final_score_percent": round(final_score, 2),
        "wall_time_ms": sum(int(row["wall_time_ms"]) for row in task_rows),
        "llm_calls": sum(int(row["llm_calls"]) for row in task_rows),
        "llm_time_ms": sum(int(row["llm_time_ms"]) for row in task_rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in task_rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in task_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in task_rows),
        "llm_tasks_run": llm_task_count,
        "avg_wall_time_ms": round(sum(int(row["wall_time_ms"]) for row in task_rows) / task_count, 2),
        "avg_llm_calls_per_task": round(sum(int(row["llm_calls"]) for row in task_rows) / task_count, 2),
        "avg_llm_time_ms": round(sum(int(row["llm_time_ms"]) for row in task_rows) / task_count, 2),
        "avg_tokens_per_task": round(sum(int(row["total_tokens"]) for row in task_rows) / task_count, 2),
        "avg_llm_time_ms_when_used": round(sum(int(row["llm_time_ms"]) for row in task_rows) / llm_divisor, 2),
        "avg_tokens_when_used": round(sum(int(row["total_tokens"]) for row in task_rows) / llm_divisor, 2),
    }


def _save_metrics(task_rows: list[dict[str, str | int | float]], final_score: float) -> tuple[Path, Path]:
    output_dir = Path("benchmark-runs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "latest_metrics.json"
    csv_path = output_dir / "latest_metrics.csv"
    totals = _build_totals(task_rows, final_score)

    payload = {
        "totals": totals,
        "tasks": task_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = [
        "task_id",
        "score",
        "wall_time_ms",
        "llm_calls",
        "llm_time_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    lines = [",".join(header)]
    for row in task_rows:
        lines.append(",".join(str(row[key]) for key in header))
    lines.append(
        ",".join(
            [
                "TOTAL",
                f"{totals['final_score_percent']:.2f}",
                str(totals["wall_time_ms"]),
                str(totals["llm_calls"]),
                str(totals["llm_time_ms"]),
                str(totals["prompt_tokens"]),
                str(totals["completion_tokens"]),
                str(totals["total_tokens"]),
            ]
        )
    )
    lines.append(
        ",".join(
            [
                "AVERAGE",
                "",
                f"{totals['avg_wall_time_ms']:.2f}",
                f"{totals['avg_llm_calls_per_task']:.2f}",
                f"{totals['avg_llm_time_ms']:.2f}",
                "",
                "",
                f"{totals['avg_tokens_per_task']:.2f}",
            ]
        )
    )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path


def _full_run_log_path(task_filter: list[str]) -> Path | None:
    if task_filter:
        return None
    output_dir = Path("benchmark-runs")
    output_dir.mkdir(exist_ok=True)
    return output_dir / "latest_full_run.txt"


def _clear_previous_run_artifacts() -> None:
    output_dir = Path("benchmark-runs")
    if not output_dir.exists():
        return
    for path in output_dir.glob("latest_*.txt"):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("latest_*.json"):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("latest_*.csv"):
        if path.is_file():
            path.unlink()


def _is_auth_error(exc: ConnectError) -> bool:
    code = getattr(exc, "code", None)
    if code is None:
        return False
    name = getattr(code, "name", None)
    if isinstance(name, str):
        return name in {"UNAUTHENTICATED", "PERMISSION_DENIED"}
    code_value = str(code)
    return code_value in {"Code.UNAUTHENTICATED", "Code.PERMISSION_DENIED"}


def _run_benchmark(task_filter: list[str]) -> None:
    scores = []
    task_rows: list[dict[str, str | int | float]] = []
    run_response = None
    try:
        if not task_filter and os.getenv("CLEAN_PREVIOUS_RUN_ARTIFACTS") == "1":
            _clear_previous_run_artifacts()

        client = HarnessServiceClientSync(BITGN_URL)
        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCHMARK_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        if task_filter:
            tasks = [task for task in res.tasks if task.task_id in set(task_filter)]
            print(f"Running {len(tasks)} filtered tasks via playground")
        else:
            try:
                run_response = client.start_run(
                    StartRunRequest(
                        benchmark_id=BENCHMARK_ID,
                        name=BITGN_RUN_NAME,
                        api_key=BITGN_API_KEY,
                    )
                )
            except ConnectError as exc:
                if _is_auth_error(exc):
                    print(
                        f"{CLI_YELLOW}Run mode unavailable ({exc.code}: {exc.message}) — "
                        f"running via playground fallback.{CLI_CLR}"
                    )
                else:
                    raise
                run_response = None

            if run_response is None:
                tasks = list(res.tasks)
                print("Running full benchmark via playground fallback")
            else:
                trial_ids = list(run_response.trial_ids)
                tasks = list(res.tasks)
                if len(trial_ids) < len(tasks):
                    print(
                        f"start_run returned only {len(trial_ids)} trial ids, expected {len(tasks)}"
                    )
                print(f"Running leaderboard run {run_response.run_id} for {len(tasks)} tasks")

        for idx, task in enumerate(tasks):
            if run_response is None or idx >= len(run_response.trial_ids):
                print(f"{'=' * 30} Starting task: {task.task_id} {'=' * 30}")
                trial = client.start_playground(
                    StartPlaygroundRequest(
                        benchmark_id=BENCHMARK_ID,
                        task_id=task.task_id,
                    )
                )
            else:
                trial_id = run_response.trial_ids[idx]
                print(f"{'=' * 30} Starting task: {task.task_id} / {trial_id} {'=' * 16}")
                trial = client.start_trial(StartTrialRequest(trial_id=trial_id))

            print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")

            telemetry = AgentRunTelemetry()
            try:
                telemetry = run_agent(MODEL_ID, trial.harness_url, trial.instruction)
            except Exception as exc:
                print(exc)

            result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
            score_value = getattr(result, "score", None)
            score_text = "N/A"
            if score_value is not None:
                score_float = float(score_value)
                scores.append((task.task_id, score_float))
                score_text = f"{score_float:0.2f}"
            task_rows.append(
                {
                    "task_id": task.task_id,
                    "score": score_text,
                    "wall_time_ms": telemetry.wall_time_ms,
                    "llm_calls": telemetry.llm_calls,
                    "llm_time_ms": telemetry.llm_time_ms,
                    "prompt_tokens": telemetry.prompt_tokens,
                    "completion_tokens": telemetry.completion_tokens,
                    "total_tokens": telemetry.total_tokens,
                }
            )

            if score_value is None:
                print(f"\n{CLI_YELLOW}Score: N/A (scoring hidden/unavailable){CLI_CLR}")
            else:
                style = CLI_GREEN if score_float == 1 else CLI_RED
                explain = textwrap.indent("\n".join(result.score_detail), "  ")
                print(f"\n{style}Score: {score_float:0.2f}\n{explain}\n{CLI_CLR}")
            print(
                "Telemetry: "
                f"wall={telemetry.wall_time_ms} ms, "
                f"llm_calls={telemetry.llm_calls}, "
                f"llm_time={telemetry.llm_time_ms} ms, "
                f"tokens={telemetry.total_tokens} "
                f"(prompt={telemetry.prompt_tokens}, completion={telemetry.completion_tokens})"
            )

    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    if task_rows:
        print("\nSummary:")
        print(_render_summary_table(task_rows))

    if scores:
        total = sum(score for _, score in scores) / len(scores) * 100.0
    else:
        total = 0.0
    totals = _build_totals(task_rows, total)
    if totals["scores_available"] == 1:
        print(f"FINAL: {total:0.2f}%")
    else:
        print("FINAL: N/A (scoring hidden/unavailable)")
    if not task_filter:
        if run_response is not None:
            submit_response = client.submit_run(SubmitRunRequest(run_id=run_response.run_id))
            if isinstance(submit_response.state, int):
                state_text = RunState.Name(submit_response.state)
            else:
                state_text = submit_response.state.name
            print(f"SubmitRun state={state_text}")
        else:
            print("SubmitRun skipped (playground fallback mode)")
    print(
        "TOTALS: "
        f"tasks={totals['tasks_run']}, "
        f"passed={totals['tasks_passed']}, "
        f"failed={totals['tasks_failed']}, "
        f"scored={totals['scored_tasks_count']}, "
        f"wall={totals['wall_time_ms']} ms, "
        f"llm_calls={totals['llm_calls']}, "
        f"llm_time={totals['llm_time_ms']} ms, "
        f"tokens={totals['total_tokens']} "
        f"(prompt={totals['prompt_tokens']}, completion={totals['completion_tokens']})"
    )
    print(
        "AVERAGES: "
        f"wall={totals['avg_wall_time_ms']:.2f} ms/task, "
        f"llm_calls={totals['avg_llm_calls_per_task']:.2f}/task, "
        f"llm_time={totals['avg_llm_time_ms']:.2f} ms/task, "
        f"tokens={totals['avg_tokens_per_task']:.2f}/task, "
        f"llm_tasks={totals['llm_tasks_run']}, "
        f"llm_time_when_used={totals['avg_llm_time_ms_when_used']:.2f} ms, "
        f"tokens_when_used={totals['avg_tokens_when_used']:.2f}"
    )
    if not task_filter or os.getenv("SAVE_PARTIAL_METRICS") == "1":
        json_path, csv_path = _save_metrics(task_rows, total)
        print(f"METRICS_JSON: {json_path}")
        print(f"METRICS_CSV: {csv_path}")
    else:
        print("METRICS_JSON: skipped for partial run")
        print("METRICS_CSV: skipped for partial run")


def main() -> None:
    task_filter = os.sys.argv[1:]
    log_path = _full_run_log_path(task_filter)
    if log_path is None:
        _run_benchmark(task_filter)
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_TeeStdout(sys.stdout, log_file)):
            _run_benchmark(task_filter)


if __name__ == "__main__":
    main()
