"""Twelve direct Gemini REST requests; no pool, LangChain, SDK, retries or failover."""
from __future__ import annotations

import argparse
import asyncio
import contextvars
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from dotenv import dotenv_values
import httpx

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash")
KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY2")
PROMPT = "Reply with exactly one word: ok"
CURRENT = contextvars.ContextVar("direct_api_call", default=None)


def load_keys(path):
    if not path.is_file():
        raise ValueError("Selected environment file does not exist")
    values = dotenv_values(path)
    keys = {name: values.get(name) for name in KEY_NAMES}
    if not all(keys.values()):
        raise ValueError("Both GEMINI_API_KEY and GEMINI_API_KEY2 must exist in the selected file")
    if len(set(keys.values())) != 2:
        raise ValueError("The two configured keys must be distinct")
    return keys


def safe_json(data, keys):
    text = json.dumps(data, indent=2)
    for key in keys.values():
        text = text.replace(key, "[REDACTED]")
    return text


async def transport_trace(name, info):
    # info can contain authentication headers: save event names and times only.
    current = CURRENT.get()
    if current is not None:
        row, start = current
        row["events"].append(dict(event=name, at_s=time.perf_counter() - start))


async def request_hook(request):
    request.extensions["trace"] = transport_trace


async def response_hook(response):
    current = CURRENT.get()
    if current is not None:
        row, start = current
        row["headers_s"] = time.perf_counter() - start
        row["timing_headers"] = {name: response.headers[name] for name in
                                  ("server-timing", "x-envoy-upstream-service-time")
                                  if name in response.headers}


async def run_cases(client, keys, on_complete=None):
    rows, active, peak = [], 0, {}

    async def one(phase, model, key_name):
        nonlocal active
        row = dict(phase=phase, model=model, key_variable=key_name, events=[], ok=False)
        start = time.perf_counter()
        token = CURRENT.set((row, start))
        active += 1
        peak[phase] = max(peak.get(phase, 0), active)
        try:
            # Bound total caller time, in addition to httpx's per-operation timeout.
            async with asyncio.timeout(60):
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": keys[key_name]},
                    json={"contents": [{"role": "user", "parts": [{"text": PROMPT}]}]},
                )
                row.update(status=response.status_code, ok=response.is_success,
                           body_bytes=len(response.content))
                try:
                    body = response.json()
                    row["usage"] = body.get("usageMetadata")
                    if not response.is_success:
                        error = body.get("error", {})
                        row["provider_error"] = dict(code=error.get("code"), status=error.get("status"))
                    else:
                        parts = [part.get("text", "") for candidate in body.get("candidates", [])
                                 for part in candidate.get("content", {}).get("parts", [])]
                        row["response_text_chars"] = sum(map(len, parts))
                except (ValueError, AttributeError, TypeError):
                    row["body_parse_error"] = True
        except Exception as exc:
            # Exception messages can include URLs/credentials; retain type only.
            row["error_type"] = type(exc).__name__
        finally:
            row["total_s"] = time.perf_counter() - start
            active -= 1
            CURRENT.reset(token)
            rows.append(row)
            if on_complete:
                on_complete(row)

    pairs = [(model, name) for model in MODELS for name in KEY_NAMES]
    for model, name in pairs:
        await one("sequential", model, name)
    await asyncio.gather(*(one("concurrent", model, name) for model, name in pairs))
    return dict(cases=rows, peak_active_requests=peak)


async def execute(keys, folder):
    def completed(row):
        line = safe_json(row, keys)
        # Each completion is preserved even if the process is interrupted later.
        with (folder / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json.loads(line)) + "\n")
        print(f"{row['phase']} {row['model']} {row['key_variable']}: "
              f"status={row.get('status', row.get('error_type', '-'))} total={row['total_s']:.3f}s "
              f"headers={row.get('headers_s', float('nan')):.3f}s", flush=True)

    started = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=60, transport=httpx.AsyncHTTPTransport(
            retries=0, limits=httpx.Limits(max_connections=12, max_keepalive_connections=12)),
        follow_redirects=False,
        event_hooks={"request": [request_hook], "response": [response_hook]},
    ) as client:
        result = await run_cases(client, keys, completed)
    result.update(wall_s=time.perf_counter() - started, prompt=PROMPT, retries=0,
                  timeout_s=60, http2=False, pool_used=False, langchain_used=False,
                  sdk_used=False, models=MODELS, key_variables=KEY_NAMES,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  note="Raw REST with minimal request body. Six sequential requests followed by six concurrent "
                       "requests on one reused client. Not a paired pool benchmark or a tail-latency estimate.")
    (folder / "results.json").write_text(safe_json(result, keys), encoding="utf-8")
    lines = ["# Direct Gemini REST measurements", "", result["note"], "",
             "| Phase | Model | Key variable | HTTP status/error | Total seconds | Headers seconds | Server timing |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for row in result["cases"]:
        lines.append(f"| {row['phase']} | {row['model']} | {row['key_variable']} | "
                     f"{row.get('status', row.get('error_type', '-'))} | {row['total_s']:.3f} | "
                     f"{row.get('headers_s', float('nan')):.3f} | "
                     f"{row.get('timing_headers', {}).get('server-timing', '-')} |")
    report = "\n".join(lines) + "\n"
    for key in keys.values():
        report = report.replace(key, "[REDACTED]")
    (folder / "summary.md").write_text(report, encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args(argv)
    if not args.yes:
        parser.error("--yes is required to send twelve direct REST requests")
    keys = load_keys(args.env_file)
    folder = ROOT / "evals/results/direct_runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    print("Direct REST only: 6 sequential + 6 concurrent requests; no retries. Artifacts:", folder, flush=True)
    result = asyncio.run(execute(keys, folder))
    successes = sum(row["ok"] for row in result["cases"])
    print(f"Completed {successes}/12 successful; artifacts: {folder}", flush=True)
    return 0 if successes == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
