"""One direct Gemini request to retain a full sanitized provider response. No pool or retry."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from dotenv import dotenv_values
import httpx

ROOT = Path(__file__).resolve().parents[1]


async def diagnose(key_name, model, values):
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60, transport=httpx.AsyncHTTPTransport(retries=0),
                                 follow_redirects=False) as client:
        async with asyncio.timeout(60):
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": values[key_name]},
                json={"contents": [{"role": "user", "parts": [{"text": "Reply with exactly one word: ok"}]}]},
            )
    try:
        body = response.json()
    except ValueError:
        body = {"unparsed_response": response.text}
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(), key_variable=key_name,
                credential_source=".env, no shell fallback", model=model, status=response.status_code,
                elapsed_s=time.perf_counter() - started, retries=0, pool_used=False,
                timing_headers={name: response.headers[name] for name in
                                ("retry-after", "server-timing") if name in response.headers}, response=body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--key-name", choices=("GEMINI_API_KEY", "GEMINI_API_KEY2"), default="GEMINI_API_KEY2")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    args = parser.parse_args()
    if not args.yes:
        parser.error("--yes is required to make one live request")
    values = dotenv_values(ROOT / ".env")
    if not values.get(args.key_name):
        parser.error("Selected key variable is missing from .env")
    result = asyncio.run(diagnose(args.key_name, args.model, values))
    text = json.dumps(result, indent=2)
    for name, value in values.items():
        if value and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET")):
            text = text.replace(value, "[REDACTED]")
    folder = ROOT / "evals/results/quota_diagnostics"
    folder.mkdir(exist_ok=True)
    path = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6] + ".json")
    path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
