import argparse
import json
from collections import defaultdict

import requests
from tqdm import tqdm


def stream_load_model(server_url: str, model: str):
    response = requests.post(f"{server_url.rstrip('/')}/load_model", json={"model": model}, stream=True)
    response.raise_for_status()

    stage_bar = tqdm(desc=f"Loading {model}", unit="stage")
    tqdm_bars = {}
    bar_positions = defaultdict(lambda: len(bar_positions))

    try:
        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data: "):
                continue

            payload = json.loads(line.split(b"data: ", 1)[1])
            stage = payload.get("stage")
            message = payload.get("message") or stage

            if stage and stage.startswith("tqdm:"):
                desc = payload.get("desc") or "Loading weights"
                total = payload.get("total")
                bar = tqdm_bars.get(desc)

                if stage == "tqdm:start" and bar is None:
                    bar = tqdm(desc=desc, total=total, position=bar_positions[desc])
                    tqdm_bars[desc] = bar
                elif bar:
                    if total is not None:
                        bar.total = total
                    current = payload.get("n")
                    if current is not None:
                        bar.n = current
                        bar.refresh()
                    elif stage == "tqdm:update":
                        bar.update(1)
                    if stage == "tqdm:close":
                        bar.close()
                continue

            if stage:
                stage_bar.update(1)
                stage_bar.set_postfix_str(message)

            if stage == "ready":
                break
            if stage == "error":
                raise RuntimeError(payload.get("message", "Unknown error"))
    finally:
        stage_bar.close()
        for bar in tqdm_bars.values():
            bar.close()


def main():
    parser = argparse.ArgumentParser(description="Stream model loading progress from `transformers serve`.")
    parser.add_argument("--server", default="http://localhost:8000", help="URL where `transformers serve` is running.")
    parser.add_argument("--model", default="mistralai/Devstral-Small-2505", help="Model ID to load.")
    args = parser.parse_args()

    stream_load_model(args.server, args.model)


if __name__ == "__main__":
    main()
