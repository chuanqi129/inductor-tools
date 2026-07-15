from __future__ import annotations

import argparse
import csv
from collections import Counter
from email.message import EmailMessage
from html import escape
import json
import os
import re
import smtplib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_BRANCH_REGEX = r"^(main|release[/-].*)$"
DEFAULT_EMAIL_SUBJECT = "Buildkite Intel CI Report"
DEFAULT_EMAIL_TO = "wenjun.liu@intel.com"
PASS_STATES = {"passed"}
FAIL_STATES = {"failing", "failed", "timed_out", "canceled", "canceling", "blocked"}
FAILED_JOB_STATES = {"failing", "failed", "timed_out"}
DONE_STATES = PASS_STATES | FAIL_STATES | {"finished", "skipped", "broken"}
TERMINAL_BUILD_STATES = PASS_STATES | FAIL_STATES | {"finished", "skipped", "broken"}
BUILDKITE_TS_RE = re.compile(r"\x1b_bk;t=(\d+)\x07")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
FAILED_CASE_RE = re.compile(r"(?:^|\s)FAILED\s+([^\s|]+::[^\s|]+)")
SUMMARY_RE = re.compile(r"=+\s+(\d+)\s+failed,.*?in\s+([0-9:.]+)\s+=+")
TEST_START_RE = re.compile(
    r"(pytest\s|python\s+-m\s+pytest|collected\s+\d+\s+items|\[api_utils\.py:273\]\s+non-default\s+args:|Initializing\s+a\s+V1\s+LLM\s+engine|Starting\s+to\s+load\s+model)",
    re.IGNORECASE,
)
DOCKER_PULL_RE = re.compile(
    r"(docker\s+pull|pulling\s+fs\s+layer|already\s+exists|download\s+complete|extracting|pull\s+access\s+denied)",
    re.IGNORECASE,
)
DOCKER_FAIL_RE = re.compile(
    r"(docker\s+pull|pull\s+access\s+denied|manifest\s+unknown|error\s+pulling\s+image|failed\s+to\s+resolve\s+reference|tls\s+handshake\s+timeout)",
    re.IGNORECASE,
)
TIMEOUT_RE = re.compile(
    r"(timed?\s+out|deadline\s+exceeded|command\s+timed\s+out|job\s+timed\s+out|the\s+command\s+exited\s+with\s+status\s+124|user\s+command\s+error:\s+signal:\s+terminated|interrupted\s+by\s+a\s+signal:\s+signal:\s+terminated)",
    re.IGNORECASE,
)
EXCEPTION_LINE_RE = re.compile(
    r"\b(?:assertionerror|runtimeerror|valueerror|typeerror|keyerror|indexerror|importerror|modulenotfounderror|notimplementederror|attributeerror|oserror)\b",
    re.IGNORECASE,
)
EXCEPTION_MESSAGE_RE = re.compile(
    r"\b(?:AssertionError|RuntimeError|ValueError|TypeError|KeyError|IndexError|ImportError|ModuleNotFoundError|NotImplementedError|AttributeError|OSError):",
)
ERROR_LINE_RE = re.compile(
    r"(traceback|error[:\s]|runtimeerror|assertionerror|exception:|segmentation\s+fault|segfault|device\s+lost)",
    re.IGNORECASE,
)
CASE_TOKEN_RE = re.compile(r"([A-Za-z0-9_./-]+\.py::[^\s,|]+)")
FREE_MEMORY_STARTUP_RE = re.compile(
    r"ValueError:\s+Free memory on device .* on startup is less than desired GPU memory utilization",
    re.IGNORECASE,
)
AGENT_IO_ERROR_RE = re.compile(
    r"error\s+starting\s+pty:.*buildkite-agent.*input/output\s+error",
    re.IGNORECASE,
)
TLS_CERT_BUNDLE_RE = re.compile(
    r"could\s+not\s+find\s+a\s+suitable\s+tls\s+ca\s+certificate\s+bundle|certifi/cacert\.pem",
    re.IGNORECASE,
)
EXIT_127_RE = re.compile(r"exit\s+status\s+127|command\s+not\s+found", re.IGNORECASE)
TRITON_CUBIN_MISSING_RE = re.compile(
    r"cubin\s+file\s+saved\s+by\s+tritonbundler\s+not\s+found|\.zebin",
    re.IGNORECASE,
)
CASE_COUNT_TREND_HISTORY_FILE = "nightly_case_count_trend_history.json"


@dataclass
class ParsedLine:
    timestamp_ms: int | None
    text: str


class BuildkiteClient:
    def __init__(self, org: str, pipeline: str, token: str, timeout: int = 60) -> None:
        self.org = org
        self.pipeline = pipeline
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        for attempt in range(4):
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code == 429 and attempt < 3:
                retry_after = int(response.headers.get("Retry-After", "30"))
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            if response.content:
                return response.json()
            return None
        raise RuntimeError(f"Buildkite API request failed after retries: {url}")

    def list_builds(self, created_from: datetime, created_to: datetime, per_page: int = 100) -> list[dict[str, Any]]:
        builds: list[dict[str, Any]] = []
        page = 1
        base_url = f"https://api.buildkite.com/v2/organizations/{self.org}/pipelines/{self.pipeline}/builds"
        created_from_str = created_from.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        created_to_str = created_to.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        while True:
            params = {
                "page": page,
                "per_page": per_page,
                "created_from": created_from_str,
                "created_to": created_to_str,
            }
            page_items = self._request("GET", base_url, params=params)
            if not page_items:
                break
            builds.extend(page_items)
            if len(page_items) < per_page:
                break
            page += 1

        return builds

    def get_build(self, build_number: int) -> dict[str, Any]:
        url = f"https://api.buildkite.com/v2/organizations/{self.org}/pipelines/{self.pipeline}/builds/{build_number}"
        return self._request("GET", url)

    def download_job_log(self, build_number: int, job_uuid: str) -> str:
        url = (
            f"https://buildkite.com/organizations/{self.org}/pipelines/"
            f"{self.pipeline}/builds/{build_number}/jobs/{job_uuid}/download.txt"
        )
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Buildkite intel-ci jobs for branch merges or PRs and export failure summaries."
    )
    parser.add_argument("--org", default="vllm")
    parser.add_argument("--pipeline", default="intel-ci")
    parser.add_argument("--token-env", default="BUILDKITE_TOKEN")
    parser.add_argument("--branch-regex", default=DEFAULT_BRANCH_REGEX)
    parser.add_argument(
        "--build-scope",
        choices=["merge", "pr", "all"],
        default="merge",
        help="merge filters to branch builds without pull_request; pr filters to PR builds.",
    )
    parser.add_argument("--days", type=int, default=7, help="Look back this many days when from/to are not given.")
    parser.add_argument("--from-date", help="Start date/time in YYYY-MM-DD or ISO-8601 format.")
    parser.add_argument("--to-date", help="End date/time in YYYY-MM-DD or ISO-8601 format.")
    parser.add_argument("--max-builds", type=int, default=0, help="Optional hard limit after filtering builds.")
    parser.add_argument(
        "--include-running-builds",
        action="store_true",
        help="Include non-terminal builds. By default only completed builds are analyzed.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each Buildkite API or log download request.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: output/buildkite_intel_ci_<timestamp>",
    )
    parser.add_argument(
        "--download-passed-logs",
        action="store_true",
        help="Download passed job logs too. By default only failed jobs are downloaded.",
    )
    parser.add_argument(
        "--email-to",
        default=DEFAULT_EMAIL_TO,
        help="Comma-separated recipient list. If set, send summary.html as an HTML email after generation.",
    )
    parser.add_argument(
        "--email-from",
        default="",
        help="Sender email address. Defaults to SMTP_FROM env var when omitted.",
    )
    parser.add_argument(
        "--email-subject",
        default=DEFAULT_EMAIL_SUBJECT,
        help="Email subject prefix for report delivery.",
    )
    parser.add_argument(
        "--smtp-host",
        default="",
        help="SMTP host used when --email-to is set. Defaults to SMTP_HOST env var when omitted.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=0,
        help="SMTP port. Defaults to 25, or 465 when --smtp-ssl is set.",
    )
    parser.add_argument(
        "--smtp-user-env",
        default="SMTP_USERNAME",
        help="Environment variable holding the SMTP username, if authentication is needed.",
    )
    parser.add_argument(
        "--smtp-password-env",
        default="SMTP_PASSWORD",
        help="Environment variable holding the SMTP password, if authentication is needed.",
    )
    parser.add_argument(
        "--smtp-starttls",
        action="store_true",
        help="Upgrade the SMTP connection with STARTTLS before optional login.",
    )
    parser.add_argument(
        "--smtp-ssl",
        action="store_true",
        help="Connect using SMTP over SSL.",
    )
    parser.add_argument(
        "--email-dry-run",
        action="store_true",
        help="Prepare email metadata and validate inputs, but do not connect to SMTP or send the message.",
    )
    parser.add_argument(
        "--nightly-name",
        default="Full intel CI-daily",
        help="Nightly build name keyword used to identify nightly runs.",
    )
    parser.add_argument(
        "--nightly-source",
        default="scheduled,schedule",
        help="Expected source for nightly runs. Supports comma/pipe separated values (for example: scheduled,schedule). Empty disables source filtering.",
    )
    parser.add_argument(
        "--nightly-lookback-days",
        type=int,
        default=14,
        help="Look back this many days to find latest two nightly runs for delta comparison.",
    )
    parser.add_argument(
        "--xpu-only",
        action="store_true",
        help="Filter nightly comparison to XPU failures only. When enabled, only test suites containing 'XPU' will be included in delta comparison.",
    )
    return parser.parse_args()


def parse_datetime_arg(value: str, is_end: bool) -> datetime:
    if "T" in value or value.endswith("Z"):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    parsed_date = datetime.fromisoformat(value)
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    if is_end:
        return parsed_date + timedelta(days=1) - timedelta(microseconds=1)
    return parsed_date


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if args.from_date:
        start = parse_datetime_arg(args.from_date, is_end=False)
    else:
        start = now - timedelta(days=args.days)

    if args.to_date:
        end = parse_datetime_arg(args.to_date, is_end=True)
    else:
        end = now

    if end < start:
        raise ValueError("to-date must be later than from-date")

    return start, end


def normalize_pull_request(value: Any) -> str | None:
    if value in (None, False, "false", "False", "0", 0, ""):
        return None
    return str(value)


def matches_scope(build: dict[str, Any], scope: str, branch_re: re.Pattern[str]) -> bool:
    branch = build.get("branch") or ""
    if not branch_re.search(branch):
        return False

    pull_request = normalize_pull_request(build.get("pull_request"))
    if scope == "merge":
        return pull_request is None
    if scope == "pr":
        return pull_request is not None
    return True


def parse_recipients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def send_html_report_email(summary: dict[str, Any], html_path: Path, args: argparse.Namespace) -> str:
    recipients = parse_recipients(args.email_to)
    if not recipients:
        return "disabled"

    sender = args.email_from or os.getenv("SMTP_FROM", "")
    subject = (
        f"{args.email_subject} | {summary['window_start']} -> {summary['window_end']} | "
        f"failed builds {summary['failed_builds']}"
    )

    if args.email_dry_run:
        if not sender:
            sender = "<unset SMTP_FROM>"
        print("Email dry run enabled; skipping SMTP delivery.")
        print(f"Email From: {sender}")
        print(f"Email To: {', '.join(recipients)}")
        print(f"Email Subject: {subject}")
        print(f"Email HTML: {html_path}")
        return "dry-run"

    smtp_host = args.smtp_host or os.getenv("SMTP_HOST", "")
    if not smtp_host:
        print("Skipping email delivery: SMTP host is not configured.")
        print(f"Email To: {', '.join(recipients)}")
        print(f"Email HTML: {html_path}")
        return "skipped"

    if not sender:
        raise ValueError("Email delivery requested but sender address is not configured. Set --email-from or SMTP_FROM.")

    smtp_port = args.smtp_port or (465 if args.smtp_ssl else 25)
    smtp_username = os.getenv(args.smtp_user_env, "")
    smtp_password = os.getenv(args.smtp_password_env, "")
    html_body = html_path.read_text(encoding="utf-8")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(
        "Buildkite Intel CI report generated successfully. Use an HTML-capable mail client to view the full report body."
    )
    message.add_alternative(html_body, subtype="html")

    smtp_cls = smtplib.SMTP_SSL if args.smtp_ssl else smtplib.SMTP
    with smtp_cls(smtp_host, smtp_port, timeout=args.request_timeout) as server:
        if not args.smtp_ssl:
            server.ehlo()
            if args.smtp_starttls:
                server.starttls()
                server.ehlo()
        if smtp_username:
            if not smtp_password:
                raise ValueError(
                    f"SMTP username was provided via {args.smtp_user_env}, but {args.smtp_password_env} is empty."
                )
            server.login(smtp_username, smtp_password)
        server.send_message(message)
    return "sent"


def is_terminal_build(build: dict[str, Any]) -> bool:
    return build.get("state") in TERMINAL_BUILD_STATES


def parse_buildkite_log(raw_log: str) -> list[ParsedLine]:
    parsed_lines: list[ParsedLine] = []
    for raw_line in raw_log.splitlines():
        timestamp_match = BUILDKITE_TS_RE.search(raw_line)
        timestamp_ms = int(timestamp_match.group(1)) if timestamp_match else None
        text = BUILDKITE_TS_RE.sub("", raw_line)
        text = ANSI_RE.sub("", text)
        text = text.replace("\r", "").replace("\x07", "").strip()
        if text:
            parsed_lines.append(ParsedLine(timestamp_ms=timestamp_ms, text=text))
    return parsed_lines


def find_first_timestamp(lines: list[ParsedLine]) -> int | None:
    for line in lines:
        if line.timestamp_ms is not None:
            return line.timestamp_ms
    return None


def find_last_timestamp(lines: list[ParsedLine]) -> int | None:
    for line in reversed(lines):
        if line.timestamp_ms is not None:
            return line.timestamp_ms
    return None


def first_match_timestamp(lines: list[ParsedLine], pattern: re.Pattern[str]) -> int | None:
    for line in lines:
        if line.timestamp_ms is not None and pattern.search(line.text):
            return line.timestamp_ms
    return None


def last_summary_timestamp(lines: list[ParsedLine]) -> int | None:
    for line in reversed(lines):
        if line.timestamp_ms is not None and (SUMMARY_RE.search(line.text) or "short test summary info" in line.text.lower()):
            return line.timestamp_ms
    return None


def duration_seconds(start_ms: int | None, end_ms: int | None) -> float | None:
    if start_ms is None or end_ms is None or end_ms < start_ms:
        return None
    return round((end_ms - start_ms) / 1000.0, 3)


def collect_failed_cases(lines: list[ParsedLine]) -> list[str]:
    seen: set[str] = set()
    cases: list[str] = []
    for line in lines:
        for match in FAILED_CASE_RE.finditer(line.text):
            case_name = match.group(1)
            if case_name not in seen:
                seen.add(case_name)
                cases.append(case_name)
    return cases


def pick_error_message(lines: list[ParsedLine]) -> str | None:
    for line in reversed(lines):
        if EXCEPTION_MESSAGE_RE.search(line.text) and "see root cause above" not in line.text.lower():
            return line.text[:1000]
    for line in reversed(lines):
        if ERROR_LINE_RE.search(line.text):
            return line.text[:1000]
    for line in reversed(lines[-20:]):
        if line.text:
            return line.text[:1000]
    return None


def classify_failure(lines: list[ParsedLine], failed_cases: list[str]) -> str:
    joined = "\n".join(line.text for line in lines[-400:])
    if AGENT_IO_ERROR_RE.search(joined):
        return "infra_agent_io_error"
    if TLS_CERT_BUNDLE_RE.search(joined):
        return "env_tls_cert_error"
    if EXIT_127_RE.search(joined):
        return "env_command_not_found"
    if TRITON_CUBIN_MISSING_RE.search(joined):
        return "triton_cubin_missing"
    if DOCKER_FAIL_RE.search(joined):
        return "docker_pull_fail"
    if failed_cases:
        return "test_case_fail"
    if EXCEPTION_LINE_RE.search(joined):
        return "runtime_error"
    if TIMEOUT_RE.search(joined):
        return "timeout"
    if re.search(r"(segmentation\s+fault|segfault|device\s+lost|oom|out\s+of\s+resources)", joined, re.IGNORECASE):
        return "runtime_crash"
    return "other"


def infer_timeout_case_name(lines: list[ParsedLine]) -> str:
    for line in reversed(lines):
        matches = CASE_TOKEN_RE.findall(line.text)
        if matches:
            return matches[-1]
    return ""


def infer_specific_case_name(
    lines: list[ParsedLine],
    failed_cases: list[str],
    error_message: str | None,
    fail_reason: str,
    timeout_case_name: str,
) -> str:
    if error_message and FREE_MEMORY_STARTUP_RE.search(error_message):
        return "startup_free_memory_insufficient"
    if any(FREE_MEMORY_STARTUP_RE.search(line.text) for line in lines):
        return "startup_free_memory_insufficient"
    if fail_reason == "timeout" and timeout_case_name:
        return timeout_case_name
    if failed_cases:
        return failed_cases[0]
    return ""


def analyze_log(raw_log: str, started_at: str | None, finished_at: str | None) -> dict[str, Any]:
    lines = parse_buildkite_log(raw_log)
    failed_cases = collect_failed_cases(lines)
    error_message = pick_error_message(lines)
    fail_reason = classify_failure(lines, failed_cases)
    timeout_case_name = infer_timeout_case_name(lines)
    if fail_reason == "timeout" and not failed_cases and timeout_case_name:
        failed_cases = [timeout_case_name]

    log_start_ms = find_first_timestamp(lines)
    log_end_ms = find_last_timestamp(lines)
    docker_start_ms = first_match_timestamp(lines, DOCKER_PULL_RE)
    test_start_ms = first_match_timestamp(lines, TEST_START_RE)
    test_end_ms = last_summary_timestamp(lines) or log_end_ms

    if docker_start_ms is None:
        docker_start_ms = log_start_ms

    started_dt = parse_api_datetime(started_at)
    finished_dt = parse_api_datetime(finished_at)
    job_start_ms = int(started_dt.timestamp() * 1000) if started_dt else log_start_ms
    job_end_ms = int(finished_dt.timestamp() * 1000) if finished_dt else log_end_ms

    return {
        "fail_reason": fail_reason,
        "fail_case_names": failed_cases,
        "specific_case_name": infer_specific_case_name(lines, failed_cases, error_message, fail_reason, timeout_case_name),
        "docker_prepare_duration_seconds": duration_seconds(docker_start_ms, test_start_ms) or duration_seconds(job_start_ms, test_start_ms),
        "test_duration_seconds": duration_seconds(test_start_ms, test_end_ms) or duration_seconds(test_start_ms, job_end_ms),
        "error_message": error_message,
    }


def parse_api_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def safe_suite_name(job: dict[str, Any]) -> str:
    return str(job.get("label") or job.get("name") or job.get("step_key") or job.get("id") or "unknown")


def format_test_node(job: dict[str, Any]) -> str:
    agent = job.get("agent") or {}
    hostname = str(agent.get("hostname") or "").strip()
    agent_name = str(agent.get("name") or "").strip()
    if hostname and agent_name and hostname != agent_name:
        return f"{hostname} ({agent_name})"
    return hostname or agent_name


def build_type(build: dict[str, Any]) -> str:
    return "pr" if normalize_pull_request(build.get("pull_request")) else "merge"


def is_target_nightly_build(build: dict[str, Any], nightly_name: str, nightly_source: str) -> bool:
    text = " ".join(
        [
            str(build.get("message") or ""),
            str(build.get("name") or ""),
            str(build.get("title") or ""),
        ]
    ).lower()
    if nightly_name and nightly_name.lower() not in text:
        return False

    if nightly_source:
        source = str(build.get("source") or "").lower()
        allowed_sources = {
            item.strip().lower()
            for item in re.split(r"[,|]", nightly_source)
            if item.strip()
        }
        if allowed_sources and source not in allowed_sources:
            return False
    return True


def collect_failed_rows_for_build(client: BuildkiteClient, build: dict[str, Any], output_dir: Path, prefix: str) -> list[dict[str, Any]]:
    build_number = int(build["number"])
    build_details = client.get_build(build_number)
    jobs = build_details.get("jobs") or []
    build_rows: list[dict[str, Any]] = []

    for job in jobs:
        job_state = job.get("state")
        if job_state not in FAILED_JOB_STATES:
            continue
        if job.get("type") and job.get("type") != "script":
            continue

        log_analysis: dict[str, Any] | None = None
        saved_log_path: str | None = None
        job_uuid = job.get("uuid") or job.get("id")

        if job_uuid:
            try:
                raw_log = client.download_job_log(build_number, str(job_uuid))
                log_file = output_dir / "logs" / f"{prefix}_build_{build_number}_job_{job_uuid}.log"
                log_file.write_text(raw_log, encoding="utf-8")
                saved_log_path = str(log_file)
                log_analysis = analyze_log(raw_log, job.get("started_at"), job.get("finished_at"))
            except requests.RequestException as exc:
                log_analysis = {
                    "fail_reason": "log_download_failed",
                    "fail_case_names": [],
                    "specific_case_name": "",
                    "docker_prepare_duration_seconds": None,
                    "test_duration_seconds": None,
                    "error_message": f"Failed to download log: {exc}",
                }

        build_rows.append(flatten_row(build_details, job, log_analysis, saved_log_path))

    return collapse_build_level_failures(build_rows)


def case_signatures(rows: list[dict[str, Any]], xpu_only: bool = False) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    # XPU only: failures where test actually ran (not infra/env issues at startup)
    xpu_ran_reasons = {"test_case_fail", "runtime_error", "runtime_crash", "timeout"}
    
    for row in rows:
        fail_reason = str(row.get("fail_reason") or "unknown").lower()
        if xpu_only and fail_reason not in xpu_ran_reasons:
            continue
        
        suite = str(row.get("test_suite_name") or "unknown")
        specific = str(row.get("specific_case_name") or "").strip()
        fail_cases = [item.strip() for item in str(row.get("fail_case_name") or "").split(";") if item.strip()]

        keys: list[str] = []
        if specific:
            keys.append(f"{suite}::{specific}")
        keys.extend(f"{suite}::{item}" for item in fail_cases)
        if not keys:
            fallback = str(row.get("fail_reason") or "unknown")
            keys.append(f"{suite}::[{fallback}]")

        for key in keys:
            if key not in result:
                result[key] = row
    return result


def normalize_case_count_cell(value: str) -> str:
    cleaned = ANSI_RE.sub("", str(value or "")).strip()
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\*+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_case_count_row(line: str) -> tuple[str, int, int, int] | None:
    normalized_line = normalize_case_count_cell(ANSI_RE.sub("", line).strip())
    if not normalized_line:
        return None

    if "|" in normalized_line:
        cells = [normalize_case_count_cell(cell) for cell in normalized_line.strip("|").split("|")]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 4 and all(re.fullmatch(r"\d+", item) for item in cells[1:4]):
            return cells[0], int(cells[1]), int(cells[2]), int(cells[3])

    match = re.match(r"^(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", normalized_line)
    if not match:
        # Some logs collapse the first boundary between group text and the first number
        # (e.g. "Basic Correctness200 0 0"). Accept optional whitespace there.
        match = re.match(r"^(.+?)(\d+)\s+(\d+)\s+(\d+)\s*$", normalized_line)
        if not match:
            return None

    group = normalize_case_count_cell(match.group(1))
    if not group:
        return None
    return group, int(match.group(2)), int(match.group(3)), int(match.group(4))


def parse_case_count_table_from_log(raw_log: str) -> dict[str, dict[str, int]] | None:
    rows: list[tuple[str, int, int, int]] = []
    in_table = False

    for raw_line in raw_log.splitlines():
        plain = ANSI_RE.sub("", raw_line).strip()
        if not plain:
            if in_table and rows:
                break
            continue

        header_key = re.sub(r"[^a-z]", "", plain.lower())
        if "grouppassedskippedfailed" in header_key:
            in_table = True
            continue

        if not in_table:
            continue

        if re.fullmatch(r"[|\-:\s]+", plain):
            continue

        parsed = parse_case_count_row(plain)
        if parsed:
            rows.append(parsed)
            continue

        if rows:
            break

    if not rows:
        return None

    stats: dict[str, dict[str, int]] = {}
    for group, passed, skipped, failed in rows:
        key = "TOTAL" if group.upper() == "TOTAL" else group
        stats[key] = {
            "passed": int(passed),
            "skipped": int(skipped),
            "failed": int(failed),
            "total": int(passed + skipped + failed),
        }

    if "TOTAL" not in stats:
        total_passed = sum(item["passed"] for item in stats.values())
        total_skipped = sum(item["skipped"] for item in stats.values())
        total_failed = sum(item["failed"] for item in stats.values())
        stats["TOTAL"] = {
            "passed": total_passed,
            "skipped": total_skipped,
            "failed": total_failed,
            "total": total_passed + total_skipped + total_failed,
        }

    ordered = [item for item in stats.items() if item[0] != "TOTAL"]
    ordered.sort(key=lambda pair: (-int(pair[1].get("total") or 0), pair[0].lower()))
    ordered.append(("TOTAL", stats["TOTAL"]))
    return {label: values for label, values in ordered}


def parse_pytest_case_counts(raw_log: str) -> tuple[int, int, int] | None:
    # Prefer the last summary-like line that carries passed/skipped/failed tokens.
    summary_lines = [ANSI_RE.sub("", line).strip() for line in raw_log.splitlines()]
    for line in reversed(summary_lines):
        if not line:
            continue
        if "passed" not in line and "failed" not in line and "skipped" not in line:
            continue

        counts = {"passed": 0, "skipped": 0, "failed": 0}
        for value, key in re.findall(r"(\d+)\s+(passed|skipped|failed)", line, flags=re.IGNORECASE):
            counts[key.lower()] += int(value)

        if counts["passed"] or counts["skipped"] or counts["failed"]:
            return counts["passed"], counts["skipped"], counts["failed"]

    return None


def collect_nightly_case_count_stats(client: BuildkiteClient, build: dict[str, Any]) -> dict[str, dict[str, int]]:
    build_number = int(build["number"])
    build_details = client.get_build(build_number)
    jobs = build_details.get("jobs") or []
    stats: dict[str, dict[str, int]] = {}

    for job in jobs:
        if job.get("type") and job.get("type") != "script":
            continue
        job_uuid = job.get("uuid") or job.get("id")
        if not job_uuid:
            continue
        try:
            raw_log = client.download_job_log(build_number, str(job_uuid))
        except requests.RequestException:
            continue

        parsed_counts = parse_pytest_case_counts(raw_log)
        if not parsed_counts:
            continue

        passed, skipped, failed = parsed_counts
        job_label = safe_suite_name(job).strip() or "unknown"
        stats[job_label] = {
            "passed": int(passed),
            "skipped": int(skipped),
            "failed": int(failed),
            "total": int(passed + skipped + failed),
        }

    if not stats:
        return {}

    ordered = [item for item in stats.items() if item[0] != "TOTAL"]
    ordered.sort(key=lambda pair: (-int(pair[1].get("total") or 0), pair[0].lower()))

    total_passed = sum(item["passed"] for _, item in ordered)
    total_skipped = sum(item["skipped"] for _, item in ordered)
    total_failed = sum(item["failed"] for _, item in ordered)
    ordered.append((
        "TOTAL",
        {
            "passed": total_passed,
            "skipped": total_skipped,
            "failed": total_failed,
            "total": total_passed + total_skipped + total_failed,
        },
    ))

    return {label: values for label, values in ordered}


def load_case_count_trend_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []

    history: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date_value = str(item.get("date") or "").strip()
        total_value = int(item.get("total") or 0)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
            continue
        history.append({"date": date_value, "total": total_value})
    history.sort(key=lambda row: row["date"])
    return history


def update_case_count_trend_history(history: list[dict[str, Any]], date_value: str, total: int) -> list[dict[str, Any]]:
    updated = [row for row in history if str(row.get("date") or "") != date_value]
    updated.append({"date": date_value, "total": int(total)})
    updated.sort(key=lambda row: row["date"])
    # Keep a reasonable retention window for chart readability.
    return updated[-120:]


def build_case_count_trend_svg(history: list[dict[str, Any]]) -> str:
    width = 720
    height = 260
    left = 56
    right = 24
    top = 24
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not history:
        return '<svg viewBox="0 0 720 160" xmlns="http://www.w3.org/2000/svg"><text x="24" y="84" font-size="13" fill="#64748b">No trend data yet</text></svg>'

    values = [int(row.get("total") or 0) for row in history]
    labels = [str(row.get("date") or "") for row in history]
    max_value = max(1, max(values))
    count = len(values)

    def y_pos(value: int) -> float:
        return top + plot_height - (plot_height * value / max_value)

    def x_pos(index: int) -> float:
        if count <= 1:
            return left + plot_width / 2
        return left + (plot_width * index / (count - 1))

    points = [(x_pos(i), values[i]) for i in range(count)]

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Case run count trend">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#c5b8a6" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#c5b8a6" stroke-width="1.2"/>',
    ]

    for label_value in sorted({0, max_value}):
        y = y_pos(label_value)
        svg_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#eadfce" stroke-width="1"/>')
        svg_parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#5f6b7a">{label_value}</text>')

    if any(values):
        path_points = " ".join(f"{x:.2f},{y_pos(value):.2f}" for x, value in points)
        svg_parts.append(f'<polyline points="{path_points}" fill="none" stroke="#0f766e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')

    step = 1 if count <= 10 else max(2, count // 10)
    for idx, (x, value) in enumerate(points):
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y_pos(value):.2f}" r="3.5" fill="#0f766e"/>')
        if idx % step == 0 or idx == count - 1:
            svg_parts.append(f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="10" fill="#334155">{escape(labels[idx])}</text>')
            svg_parts.append(f'<text x="{x:.2f}" y="{y_pos(value) - 8:.2f}" text-anchor="middle" font-size="10" fill="#334155">{value}</text>')

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def compute_nightly_comparison(
    client: BuildkiteClient,
    end: datetime,
    output_dir: Path,
    nightly_name: str,
    nightly_source: str,
    lookback_days: int,
    xpu_only: bool = False,
) -> dict[str, Any] | None:
    start = end - timedelta(days=max(1, lookback_days))
    builds = client.list_builds(start, end)
    candidates = [
        build
        for build in builds
        if is_terminal_build(build) and is_target_nightly_build(build, nightly_name, nightly_source)
    ]
    candidates.sort(key=lambda item: item.get("number", 0), reverse=True)

    if len(candidates) < 2:
        return None

    latest = candidates[0]
    previous = candidates[1]
    latest_rows = collect_failed_rows_for_build(client, latest, output_dir, "nightly_latest")
    previous_rows = collect_failed_rows_for_build(client, previous, output_dir, "nightly_previous")
    latest_case_count_stats = collect_nightly_case_count_stats(client, latest)
    previous_case_count_stats = collect_nightly_case_count_stats(client, previous)

    latest_cases = case_signatures(latest_rows, xpu_only=xpu_only)
    previous_cases = case_signatures(previous_rows, xpu_only=xpu_only)

    new_fail_keys = sorted(set(latest_cases.keys()) - set(previous_cases.keys()))
    new_pass_keys = sorted(set(previous_cases.keys()) - set(latest_cases.keys()))
    unchanged_fail_keys = sorted(set(previous_cases.keys()) & set(latest_cases.keys()))

    first_seen_for_new_fail: dict[str, dict[str, Any]] = {}
    unresolved = set(new_fail_keys)
    scan_builds = [
        build
        for build in builds
        if (
            normalize_pull_request(build.get("pull_request")) is None
            and str(build.get("branch") or "") == str(latest.get("branch") or "")
            and int(previous.get("number") or 0) < int(build.get("number") or 0) <= int(latest.get("number") or 0)
            and is_terminal_build(build)
        )
    ]
    scan_builds.sort(key=lambda item: int(item.get("number") or 0))

    for build in scan_builds:
        if not unresolved:
            break
        if build.get("state") not in FAIL_STATES:
            continue
        build_rows = collect_failed_rows_for_build(client, build, output_dir, f"nightly_scan_{build.get('number')}")
        build_cases = set(case_signatures(build_rows).keys())
        hit = sorted(unresolved & build_cases)
        for key in hit:
            first_seen_for_new_fail[key] = {
                "build_id": build.get("number"),
                "build_url": build.get("web_url") or "",
                "commit_id": build.get("commit") or "",
            }
        unresolved -= set(hit)

    return {
        "nightly_name": nightly_name,
        "nightly_source": nightly_source,
        "xpu_only": xpu_only,
        "latest": {
            "build_id": latest.get("number"),
            "build_url": latest.get("web_url") or "",
            "commit_id": latest.get("commit") or "",
            "created_at": latest.get("created_at") or "",
            "state": latest.get("state") or "",
        },
        "previous": {
            "build_id": previous.get("number"),
            "build_url": previous.get("web_url") or "",
            "commit_id": previous.get("commit") or "",
            "created_at": previous.get("created_at") or "",
            "state": previous.get("state") or "",
        },
        "latest_case_count_stats": latest_case_count_stats,
        "previous_case_count_stats": previous_case_count_stats,
        "new_fails": [
            {
                "signature": key,
                "suite": str(latest_cases[key].get("test_suite_name") or ""),
                "fail_reason": str(latest_cases[key].get("fail_reason") or ""),
                "guilty_commit": (first_seen_for_new_fail.get(key) or {}).get("commit_id") or latest.get("commit") or "",
                "guilty_build_id": (first_seen_for_new_fail.get(key) or {}).get("build_id") or latest.get("number"),
                "guilty_build_url": (first_seen_for_new_fail.get(key) or {}).get("build_url") or latest.get("web_url") or "",
            }
            for key in new_fail_keys
        ],
        "new_passes": [
            {
                "signature": key,
                "suite": str(previous_cases[key].get("test_suite_name") or ""),
                "previous_fail_reason": str(previous_cases[key].get("fail_reason") or ""),
                "candidate_fix_commit": latest.get("commit") or "",
                "candidate_fix_build_id": latest.get("number"),
                "candidate_fix_build_url": latest.get("web_url") or "",
            }
            for key in new_pass_keys
        ],
        "unchanged_fails": [
            {
                "signature": key,
                "suite": str(latest_cases[key].get("test_suite_name") or ""),
                "latest_fail_reason": str(latest_cases[key].get("fail_reason") or ""),
                "previous_fail_reason": str(previous_cases[key].get("fail_reason") or ""),
                "latest_build_id": latest.get("number"),
                "latest_build_url": latest.get("web_url") or "",
                "latest_commit": latest.get("commit") or "",
                "previous_build_id": previous.get("number"),
                "previous_build_url": previous.get("web_url") or "",
                "previous_commit": previous.get("commit") or "",
            }
            for key in unchanged_fail_keys
        ],
    }


def ensure_output_dir(path_arg: str | None) -> Path:
    if path_arg:
        output_dir = Path(path_arg)
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/root/wenjun/code/xpu-issue-tracker/output") / f"buildkite_intel_ci_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    return output_dir


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "build_id",
        "build_type",
        "build_state",
        "job_id",
        "job_state",
        "test_node",
        "vllm_branch",
        "commit_id",
        "trigger_time",
        "pull_request",
        "test_suite_name",
        "fail_reason",
        "specific_case_name",
        "fail_case_name",
        "docker_prepare_duration_seconds",
        "test_duration_seconds",
        "error_message",
        "build_url",
        "job_url",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "window_start",
        "window_end",
        "branch_regex",
        "build_scope",
        "total_builds",
        "passed_builds",
        "failed_builds",
        "failed_job_rows",
        "build_errors",
        "log_download_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: summary.get(key, "") for key in fieldnames})


def write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Buildkite Intel CI Summary",
        "",
        f"- Window: {summary['window_start']} to {summary['window_end']}",
        f"- Branch filter: {summary['branch_regex']}",
        f"- Build scope: {summary['build_scope']}",
        f"- Builds: {summary['total_builds']} total, {summary['passed_builds']} passed, {summary['failed_builds']} failed",
        f"- Failed job rows: {summary['failed_job_rows']}",
        f"- Build detail fetch errors: {summary['build_errors']}",
        f"- Log download errors: {summary['log_download_errors']}",
        "",
        "## Failed Jobs",
        "",
        "| Build | Node | Branch | Suite | Reason | Specific Case | Cases | Docker Prepare(s) | Test(s) | Error |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]

    failed_rows = [row for row in rows if row["job_state"] in FAIL_STATES]
    for row in failed_rows:
        lines.append(
            "| {build_id} | {test_node} | {vllm_branch} | {test_suite_name} | {fail_reason} | {specific_case_name} | {fail_case_name} | {docker_prepare_duration_seconds} | {test_duration_seconds} | {error_message} |".format(
                **{
                    **row,
                    "test_node": row.get("test_node") or "",
                    "specific_case_name": row.get("specific_case_name") or "",
                    "fail_case_name": row["fail_case_name"] or "",
                    "docker_prepare_duration_seconds": row["docker_prepare_duration_seconds"] or "",
                    "test_duration_seconds": row["test_duration_seconds"] or "",
                    "error_message": (row["error_message"] or "").replace("|", " ")[:160],
                }
            )
        )

    if not failed_rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_html_table(headers: list[str], data_rows: list[list[str]], raw_html_columns: set[int] | None = None) -> str:
    raw_html_columns = raw_html_columns or set()
    head_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{value if index in raw_html_columns else escape(value)}</td>"
            for index, value in enumerate(row)
        )
        + "</tr>"
        for row in data_rows
    )
    if not body_html:
        body_html = f"<tr><td colspan=\"{len(headers)}\">No data</td></tr>"
    return (
        "<table>"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def write_html_report(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
    nightly_comparison: dict[str, Any] | None = None,
    case_count_trend_history: list[dict[str, Any]] | None = None,
) -> None:
    state_counts = Counter(str(row.get("job_state") or "unknown") for row in rows)
    reason_counts = Counter(str(row.get("fail_reason") or "unknown") for row in rows)
    suite_counts = Counter(str(row.get("test_suite_name") or "unknown") for row in rows)

    metric_cards = [
        ("Window Start", summary["window_start"]),
        ("Window End", summary["window_end"]),
        ("Build Scope", str(summary["build_scope"])),
        ("Total Builds", str(summary["total_builds"])),
        ("Passed Builds", str(summary["passed_builds"])),
        ("Failed Builds", str(summary["failed_builds"])),
        ("Failed Job Rows", str(summary["failed_job_rows"])),
        ("Log Download Errors", str(summary["log_download_errors"])),
    ]

    cards_html = "".join(
        (
            '<div class="metric-card">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in metric_cards
    )

    state_rows = [[state, str(count)] for state, count in state_counts.most_common()]
    reason_rows = [[reason, str(count)] for reason, count in reason_counts.most_common()]
    suite_rows = [[suite, str(count)] for suite, count in suite_counts.most_common(15)]
    failed_job_rows = [
        [
            (
                f'<a href="{escape(str(row.get("build_url") or ""), quote=True)}" target="_blank" rel="noopener noreferrer">'
                f'{escape(str(row.get("build_id") or ""))}</a>'
                if row.get("build_url")
                else escape(str(row.get("build_id") or ""))
            ),
            (
                f'<a href="{escape(str(row.get("job_url") or ""), quote=True)}" target="_blank" rel="noopener noreferrer">Open job</a>'
                if row.get("job_url")
                else ""
            ),
            str(row.get("test_node") or ""),
            str(row.get("vllm_branch") or ""),
            str(row.get("test_suite_name") or ""),
            str(row.get("fail_reason") or ""),
            str(row.get("specific_case_name") or ""),
            str(row.get("fail_case_name") or "")[:240],
            str(row.get("job_state") or ""),
            str(row.get("docker_prepare_duration_seconds") or ""),
            str(row.get("test_duration_seconds") or ""),
            str(row.get("error_message") or "")[:240],
        ]
        for row in rows
    ]

    nightly_section_html = (
        "<section class=\"section\">"
        "<div class=\"section-header\"><h2>Nightly Delta</h2>"
        "<span class=\"hint\">Latest vs previous scheduled nightly (Full intel CI-daily)</span></div>"
        "<p>No nightly comparison data for this run. Check matching rules: nightly name/source/lookback.</p>"
        "</section>"
    )
    if nightly_comparison:
        latest = nightly_comparison.get("latest") or {}
        previous = nightly_comparison.get("previous") or {}
        overview_rows = [
            ["Nightly Name", str(nightly_comparison.get("nightly_name") or "")],
            ["Nightly Source", str(nightly_comparison.get("nightly_source") or "")],
            ["Latest Build", str(latest.get("build_id") or "")],
            ["Latest Date", str(latest.get("created_at") or "")[:10]],
            ["Latest Commit", str(latest.get("commit_id") or "")],
            ["Previous Build", str(previous.get("build_id") or "")],
            ["Previous Date", str(previous.get("created_at") or "")[:10]],
            ["Previous Commit", str(previous.get("commit_id") or "")],
            ["New Fail", str(len(nightly_comparison.get("new_fails") or []))],
            ["New Pass", str(len(nightly_comparison.get("new_passes") or []))],
            ["Unchanged Fail", str(len(nightly_comparison.get("unchanged_fails") or []))],
        ]

        new_fail_rows = [
            [
                str(item.get("signature") or ""),
                str(item.get("suite") or ""),
                str(item.get("fail_reason") or ""),
                str(item.get("guilty_commit") or ""),
                (
                    f'<a href="{escape(str(item.get("guilty_build_url") or ""), quote=True)}" target="_blank" rel="noopener noreferrer">{escape(str(item.get("guilty_build_id") or ""))}</a>'
                    if item.get("guilty_build_url")
                    else str(item.get("guilty_build_id") or "")
                ),
            ]
            for item in (nightly_comparison.get("new_fails") or [])
        ]

        new_pass_rows = [
            [
                str(item.get("signature") or ""),
                str(item.get("suite") or ""),
                str(item.get("previous_fail_reason") or ""),
                str(item.get("candidate_fix_commit") or ""),
                (
                    f'<a href="{escape(str(item.get("candidate_fix_build_url") or ""), quote=True)}" target="_blank" rel="noopener noreferrer">{escape(str(item.get("candidate_fix_build_id") or ""))}</a>'
                    if item.get("candidate_fix_build_url")
                    else str(item.get("candidate_fix_build_id") or "")
                ),
            ]
            for item in (nightly_comparison.get("new_passes") or [])
        ]

        unchanged_fail_rows = [
            [
                str(item.get("signature") or ""),
                str(item.get("suite") or ""),
                str(item.get("latest_fail_reason") or ""),
                str(item.get("previous_fail_reason") or ""),
                str(item.get("latest_commit") or ""),
                (
                    f'<a href="{escape(str(item.get("latest_build_url") or ""), quote=True)}" target="_blank" rel="noopener noreferrer">{escape(str(item.get("latest_build_id") or ""))}</a>'
                    if item.get("latest_build_url")
                    else str(item.get("latest_build_id") or "")
                ),
            ]
            for item in (nightly_comparison.get("unchanged_fails") or [])
        ]

        latest_case_count_stats = nightly_comparison.get("latest_case_count_stats") or {}
        latest_case_count_rows = [
            [
                group,
                str(int((counts or {}).get("passed") or 0)),
                str(int((counts or {}).get("skipped") or 0)),
                str(int((counts or {}).get("failed") or 0)),
                str(int((counts or {}).get("total") or 0)),
            ]
            for group, counts in latest_case_count_stats.items()
        ]
        case_count_trend_svg = build_case_count_trend_svg(case_count_trend_history or [])

        nightly_section_html = f"""
        <section class=\"section\">
            <div class=\"section-header\">
                <h2>Nightly Delta</h2>
                <span class=\"hint\">Latest vs previous scheduled nightly (Full intel CI-daily)</span>
            </div>
            <div class="nightly-stack">
                <div class="nightly-card">
                    <h3>Overview</h3>
                    {render_html_table(["Field", "Value"], overview_rows)}
                </div>
                <div class="nightly-card">
                    <h3>New Fail</h3>
                    {render_html_table(["Case", "Suite", "Reason", "Guilty Commit", "Guilty Build"], new_fail_rows, raw_html_columns={4})}
                </div>
                <div class="nightly-card">
                    <h3>New Pass</h3>
                    {render_html_table(["Case", "Suite", "Previous Reason", "Candidate Fix Commit", "Candidate Build"], new_pass_rows, raw_html_columns={4})}
                </div>
                <div class="nightly-card">
                    <h3>Unchanged Fail</h3>
                    {render_html_table(["Case", "Suite", "Latest Reason", "Previous Reason", "Latest Commit", "Latest Build"], unchanged_fail_rows, raw_html_columns={5})}
                </div>
                <div class="nightly-card">
                    <h3>Case Run Count (Current Week)</h3>
                    {render_html_table(["Job", "Passed", "Skipped", "Failed", "Total"], latest_case_count_rows)}
                </div>
                <div class="nightly-card">
                    <h3>Case Run Count Trend</h3>
                    <div class="trend-wrap">{case_count_trend_svg}</div>
                </div>
            </div>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Buildkite Intel CI Report</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f1ea;
            --panel: #fffdf8;
            --ink: #1b2430;
            --muted: #5f6b7a;
            --line: #d8d2c7;
            --accent: #0f766e;
            --accent-soft: #d7f3ef;
            --warn: #92400e;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: "Segoe UI", "Helvetica Neue", sans-serif; background: linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%); color: var(--ink); }}
        .page {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px 48px; }}
        h1, h2 {{ margin: 0 0 12px; line-height: 1.2; }}
        h3 {{ margin: 0 0 12px; }}
        p {{ margin: 0; color: var(--muted); }}
        a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
        a:hover {{ text-decoration: underline; }}
        .hero {{ padding: 28px; border: 1px solid var(--line); border-radius: 20px; background: radial-gradient(circle at top left, #ffffff 0%, #f6efe4 55%, #f1e8da 100%); box-shadow: 0 10px 30px rgba(27, 36, 48, 0.08); }}
        .hero h1 {{ font-size: 32px; }}
        .hero p {{ margin-top: 10px; max-width: 880px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 24px; }}
        .metric-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 18px; }}
        .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: 700; }}
        .section {{ margin-top: 28px; padding: 24px; background: var(--panel); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 10px 30px rgba(27, 36, 48, 0.05); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
        .nightly-stack {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
        .nightly-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px; background: #fffaf2; }}
        .nightly-card h3 {{ margin-bottom: 10px; }}
        .trend-wrap {{ width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: #fffdf9; }}
        .trend-wrap svg {{ width: 100%; min-width: 680px; height: auto; display: block; }}
        .section-header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 16px; }}
        .hint {{ font-size: 13px; color: var(--muted); }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; table-layout: fixed; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }}
        th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
        tbody tr:hover {{ background: #faf6ef; }}
        .pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 600; }}
        .footer-note {{ margin-top: 14px; color: var(--warn); font-size: 13px; }}
        .table-wrap {{ width: 100%; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class=\"page\">
        <section class=\"hero\">
            <span class=\"pill\">Buildkite Intel CI HTML Report</span>
            <h1>Intel CI Job Distribution and Failure Summary</h1>
            <p>Time window: {escape(summary['window_start'])} to {escape(summary['window_end'])}. Branch filter: {escape(summary['branch_regex'])}. The top section summarizes build coverage and failed job distributions in the selected window; the bottom section lists the failed job summary rows exported by the analyzer.</p>
            <div class=\"metrics\">{cards_html}</div>
        </section>

        <section class=\"section\">
            <div class=\"section-header\">
                <h2>Job Distribution</h2>
                <span class=\"hint\">Distribution is computed from the exported failed job rows for this time window.</span>
            </div>
            <div class=\"grid\">
                <div>
                    <h3>By Job State</h3>
                    {render_html_table(["Job State", "Count"], state_rows)}
                </div>
                <div>
                    <h3>By Fail Reason</h3>
                    {render_html_table(["Fail Reason", "Count"], reason_rows)}
                </div>
                <div>
                    <h3>Top Suites</h3>
                    {render_html_table(["Test Suite", "Count"], suite_rows)}
                </div>
            </div>
        </section>

        {nightly_section_html}

        <section class=\"section\">
            <div class=\"section-header\">
                <h2>Failed Job Summary</h2>
                <span class=\"hint\">Same content basis as failed_jobs.csv, rendered for quick browsing.</span>
            </div>
            <div class="table-wrap">{render_html_table(["Build", "Job Link", "Node", "Branch", "Suite", "Reason", "Specific Case", "Cases", "Job State", "Docker Prepare(s)", "Test(s)", "Error"], failed_job_rows, raw_html_columns={0, 1})}</div>
            <div class=\"footer-note\">Long error messages are truncated to keep the report readable. Use the CSV or downloaded logs for full details.</div>
        </section>
    </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def summarize(rows: list[dict[str, Any]], selected_builds: list[dict[str, Any]], args: argparse.Namespace, start: datetime, end: datetime) -> dict[str, Any]:
    passed_builds = sum(1 for build in selected_builds if build.get("state") in PASS_STATES)
    failed_builds = sum(1 for build in selected_builds if build.get("state") in FAIL_STATES)
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "branch_regex": args.branch_regex,
        "build_scope": args.build_scope,
        "total_builds": len(selected_builds),
        "passed_builds": passed_builds,
        "failed_builds": failed_builds,
        "failed_job_rows": len(rows),
        "build_errors": sum(1 for row in rows if row.get("fail_reason") == "build_detail_fetch_failed"),
        "log_download_errors": sum(1 for row in rows if row.get("fail_reason") == "log_download_failed"),
    }


def flatten_row(build: dict[str, Any], job: dict[str, Any], log_analysis: dict[str, Any] | None, log_path: str | None) -> dict[str, Any]:
    fail_cases = [] if not log_analysis else log_analysis.get("fail_case_names") or []
    fail_reason = "" if not log_analysis else log_analysis.get("fail_reason") or ""
    if job.get("state") == "timed_out":
        fail_reason = "timeout"
    return {
        "build_id": build.get("number"),
        "build_type": build_type(build),
        "build_state": build.get("state"),
        "job_id": job.get("id") or job.get("uuid"),
        "job_state": job.get("state"),
        "test_node": format_test_node(job),
        "vllm_branch": build.get("branch"),
        "commit_id": build.get("commit"),
        "trigger_time": build.get("created_at"),
        "pull_request": normalize_pull_request(build.get("pull_request")) or "",
        "test_suite_name": safe_suite_name(job),
        "fail_reason": fail_reason,
        "specific_case_name": "" if not log_analysis else log_analysis.get("specific_case_name") or "",
        "fail_case_name": "; ".join(fail_cases),
        "docker_prepare_duration_seconds": "" if not log_analysis else log_analysis.get("docker_prepare_duration_seconds") or "",
        "test_duration_seconds": "" if not log_analysis else log_analysis.get("test_duration_seconds") or "",
        "error_message": "" if not log_analysis else log_analysis.get("error_message") or "",
        "build_url": build.get("web_url") or "",
        "job_url": job.get("web_url") or "",
        "log_path": log_path or "",
    }


def collapse_build_level_failures(build_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_build_row = next(
        (
            row
            for row in build_rows
            if "build xpu image" in str(row.get("test_suite_name", "")).lower()
        ),
        None,
    )
    if not image_build_row:
        return build_rows

    downstream_rows = [row for row in build_rows if row is not image_build_row]
    if not downstream_rows:
        image_build_row["fail_reason"] = "build_fail"
        return [image_build_row]

    if all(row.get("fail_reason") == "docker_pull_fail" for row in downstream_rows):
        image_build_row["fail_reason"] = "build_fail"
        if not image_build_row.get("error_message"):
            image_build_row["error_message"] = "Build XPU image failed; downstream jobs were skipped from failure details."
        return [image_build_row]

    return build_rows


def validate_buildkite_token(token: str, env_name: str) -> str | None:
    candidate = token.strip()
    if not candidate:
        return f"Missing required token in environment variable {env_name}"
    if candidate != token:
        return f"{env_name} contains leading or trailing whitespace; export the raw Buildkite token only."
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        return (
            f"{env_name} contains non-ASCII characters. "
            "It looks like a placeholder token was exported instead of a real Buildkite token."
        )
    if candidate in {"your_bkua_token", "<your_bkua_token>"}:
        return f"{env_name} is still set to a placeholder. Replace it with a real token that starts with 'bkua_'."
    return None


def main() -> int:
    args = parse_args()
    token = os.getenv(args.token_env)
    token_error = validate_buildkite_token(token or "", args.token_env)
    if token_error:
        print(token_error, file=sys.stderr)
        return 2

    try:
        start, end = resolve_window(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    branch_re = re.compile(args.branch_regex)
    output_dir = ensure_output_dir(args.output_dir)
    client = BuildkiteClient(args.org, args.pipeline, token, timeout=args.request_timeout)

    print(f"Fetching builds for {args.org}/{args.pipeline} from {start.isoformat()} to {end.isoformat()}...")
    builds = client.list_builds(start, end)
    selected_builds = [build for build in builds if matches_scope(build, args.build_scope, branch_re)]
    if not args.include_running_builds:
        selected_builds = [build for build in selected_builds if is_terminal_build(build)]
    selected_builds.sort(key=lambda item: item.get("number", 0), reverse=True)
    if args.max_builds > 0:
        selected_builds = selected_builds[: args.max_builds]

    print(
        f"Selected {len(selected_builds)} builds after filters"
        f"{' (including running builds)' if args.include_running_builds else ' (completed builds only)'}"
    )

    rows: list[dict[str, Any]] = []
    total_builds = len(selected_builds)
    for index, build in enumerate(selected_builds, start=1):
        build_number = build["number"]
        print(
            f"[{index}/{total_builds}] Processing build {build_number}"
            f" state={build.get('state')} branch={build.get('branch')}"
        )
        try:
            build_details = client.get_build(build_number)
        except requests.RequestException as exc:
            rows.append(
                {
                    "build_id": build_number,
                    "build_type": build_type(build),
                    "build_state": build.get("state"),
                    "job_id": "",
                    "job_state": "",
                    "test_node": "",
                    "vllm_branch": build.get("branch"),
                    "commit_id": build.get("commit"),
                    "trigger_time": build.get("created_at"),
                    "pull_request": normalize_pull_request(build.get("pull_request")) or "",
                    "test_suite_name": "",
                    "fail_reason": "build_detail_fetch_failed",
                    "specific_case_name": "",
                    "fail_case_name": "",
                    "docker_prepare_duration_seconds": "",
                    "test_duration_seconds": "",
                    "error_message": f"Failed to fetch build detail: {exc}",
                    "build_url": build.get("web_url") or "",
                    "job_url": "",
                    "log_path": "",
                }
            )
            print(f"  build detail fetch failed: {exc}")
            continue

        jobs = build_details.get("jobs") or []
        build_failed = build_details.get("state") in FAIL_STATES
        if not build_failed:
            print("  build passed; keeping summary stats only")
            continue

        job_count = sum(
            1
            for job in jobs
            if (
                job.get("state") in FAILED_JOB_STATES
                and (not job.get("type") or job.get("type") == "script")
            )
        )
        print(f"  found {job_count} failed script jobs")

        build_rows: list[dict[str, Any]] = []
        for job in jobs:
            job_state = job.get("state")
            if job_state not in FAILED_JOB_STATES:
                continue
            if job.get("type") and job.get("type") != "script":
                continue

            should_download = True
            log_analysis: dict[str, Any] | None = None
            saved_log_path: str | None = None

            job_uuid = job.get("uuid") or job.get("id")
            if should_download and job_uuid:
                try:
                    raw_log = client.download_job_log(build_number, str(job_uuid))
                    log_file = output_dir / "logs" / f"build_{build_number}_job_{job_uuid}.log"
                    log_file.write_text(raw_log, encoding="utf-8")
                    saved_log_path = str(log_file)
                    log_analysis = analyze_log(raw_log, job.get("started_at"), job.get("finished_at"))
                except requests.RequestException as exc:
                    log_analysis = {
                        "fail_reason": "log_download_failed",
                        "fail_case_names": [],
                        "specific_case_name": "",
                        "docker_prepare_duration_seconds": None,
                        "test_duration_seconds": None,
                        "error_message": f"Failed to download log: {exc}",
                    }

            build_rows.append(flatten_row(build_details, job, log_analysis, saved_log_path))

        rows.extend(collapse_build_level_failures(build_rows))

    nightly_comparison: dict[str, Any] | None = None
    try:
        nightly_comparison = compute_nightly_comparison(
            client=client,
            end=end,
            output_dir=output_dir,
            nightly_name=args.nightly_name,
            nightly_source=args.nightly_source,
            lookback_days=args.nightly_lookback_days,
            xpu_only=args.xpu_only,
        )
    except requests.RequestException as exc:
        print(f"Nightly comparison skipped due to API error: {exc}")

    summary = summarize(rows, selected_builds, args, start, end)
    if nightly_comparison:
        summary["nightly_comparison"] = {
            "latest_build_id": (nightly_comparison.get("latest") or {}).get("build_id"),
            "previous_build_id": (nightly_comparison.get("previous") or {}).get("build_id"),
            "new_fail_count": len(nightly_comparison.get("new_fails") or []),
            "new_pass_count": len(nightly_comparison.get("new_passes") or []),
            "unchanged_fail_count": len(nightly_comparison.get("unchanged_fails") or []),
        }
    write_summary_csv(summary, output_dir / "summary.csv")
    write_csv(rows, output_dir / "failed_jobs.csv")
    write_markdown(summary, rows, output_dir / "summary.md")
    history_path = output_dir.parent / CASE_COUNT_TREND_HISTORY_FILE
    case_count_trend_history = load_case_count_trend_history(history_path)
    if nightly_comparison:
        latest_build = nightly_comparison.get("latest") or {}
        latest_case_stats = nightly_comparison.get("latest_case_count_stats") or {}
        latest_total = int((latest_case_stats.get("TOTAL") or {}).get("total") or 0)
        latest_date = str(latest_build.get("created_at") or "")[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", latest_date):
            case_count_trend_history = update_case_count_trend_history(case_count_trend_history, latest_date, latest_total)
            history_path.write_text(json.dumps(case_count_trend_history, indent=2), encoding="utf-8")

    write_html_report(
        summary,
        rows,
        output_dir / "summary.html",
        nightly_comparison=nightly_comparison,
        case_count_trend_history=case_count_trend_history,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "failed_jobs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if nightly_comparison:
        (output_dir / "nightly_comparison.json").write_text(json.dumps(nightly_comparison, indent=2), encoding="utf-8")

    email_status = "disabled"
    try:
        email_status = send_html_report_email(summary, output_dir / "summary.html", args)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        print(f"Email delivery failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    print(f"CSV summary: {output_dir / 'summary.csv'}")
    print(f"CSV failed jobs: {output_dir / 'failed_jobs.csv'}")
    print(f"HTML report: {output_dir / 'summary.html'}")
    if parse_recipients(args.email_to):
        if email_status == "dry-run":
            print(f"HTML report email dry run prepared for {', '.join(parse_recipients(args.email_to))}")
        elif email_status == "sent":
            print(f"HTML report emailed to {', '.join(parse_recipients(args.email_to))}")
        elif email_status == "skipped":
            print(f"HTML report email skipped for {', '.join(parse_recipients(args.email_to))}")
    print(f"Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
