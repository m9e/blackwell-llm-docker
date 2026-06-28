#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def request_json(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--prompt",
        default=(
            "Write a compact Python implementation of an LRU cache with get "
            "and put methods, including a small usage example.\n\n```python\n"
        ),
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = json.loads(urllib.request.urlopen(base + "/models", timeout=10).read())
        model = models["data"][0]["id"]

    for run in range(1, args.runs + 1):
        payload = {
            "model": model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        if args.min_tokens:
            payload["min_tokens"] = args.min_tokens
        if args.ignore_eos:
            payload["ignore_eos"] = True

        start = time.perf_counter()
        response = request_json(base + "/completions", payload, args.timeout)
        elapsed = time.perf_counter() - start
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens") or 0
        prompt_tokens = usage.get("prompt_tokens") or 0
        total_tokens = usage.get("total_tokens") or 0
        tps = completion_tokens / elapsed if elapsed and completion_tokens else 0.0
        text = response["choices"][0].get("text", "")

        print(
            f"run={run} elapsed_s={elapsed:.3f} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
            f"completion_tps={tps:.3f}"
        )
        print("first_500_chars=", text[:500].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
