import os
import re
import json
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# ----------------------------
# Table parsing
# ----------------------------

PATTERN = re.compile(
    r"\|(?P<task>[^|]+)\s*\|"
    r"\s*(?P<version>\d*)\s*\|"
    r"[^|]*\|\s*\d+\s*\|"
    r"(?P<context>[\w\d]*)\s*\|"
    r"↑\s*\|\s*(?P<value>[-+]?\d+(?:\.\d+)?)"
)

def extract_scores(path, skip_first=False):
    results = defaultdict(list)
    current_task = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            m = PATTERN.search(line)
            if not m:
                continue

            task = m.group("task").strip()
            if task:
                current_task = task
            elif current_task is None:
                continue

            try:
                ctx = int(m.group("context"))
            except ValueError:
                ctx = 0

            val = float(m.group("value"))
            if val < 0:
                continue

            results[current_task].append((ctx, val))

    for t in results:
        results[t].sort(key=lambda x: x[0])
        if skip_first:
            results[t] = results[t][1:]

        if len(results[t]) == 0:
            raise RuntimeError(f"❌ Empty task '{t}' in {path}")

    if not results:
        raise RuntimeError(f"❌ No valid results in {path}")

    return results


def average_tasks(task_scores):
    total, count = 0.0, 0
    for scores in task_scores.values():
        vals = [v for _, v in scores]
        total += sum(vals)
        count += len(vals)

    if count == 0:
        raise RuntimeError("❌ No values to average")

    return total / count


# ----------------------------
# Layer collection
# ----------------------------

def collect_layers(model_dir, filename, skip_first):
    layer_re = re.compile(r"layer(\d+)")
    values = {}
    
    for d in os.listdir(model_dir):
        m = layer_re.match(d)
        if not m:
            continue

        layer = int(m.group(1))
        path = os.path.join(model_dir, d, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        scores = extract_scores(path, skip_first)
        values[layer] = average_tasks(scores)

    if not values:
        raise RuntimeError("❌ No layers found")

    layers = sorted(values)
    return {
        "contexts": layers,
        "values": [values[l] for l in layers],
    }


# ----------------------------
# Combine metrics
# ----------------------------

def combine(cs, rt, cs_base, rt_base, mode):
    cs = np.array(cs)
    rt = np.array(rt)

    mode = mode.lower()

    if mode == "jet-nemotron":
        # retrieval drop relative to original
        return rt_base - rt

    elif mode == "halo":
        eps = 1e-10
        cs_base = max(cs)
        rt_base = max(rt)
        out = (rt_base - rt) / (cs_base - cs + eps)
        return np.minimum(50, out)

    else:
        raise ValueError(f"Invalid mode: {mode}")


# ----------------------------
# Main
# ----------------------------

def main(args):
    root = args.root
    model_dir = os.path.join(root, args.model)
    orig_dir = os.path.join(root, "original")

    cs_layers = collect_layers(
        model_dir, "commonsense_eval.log", skip_first=True
    )
    rt_layers = collect_layers(
        model_dir, "retrieval_eval.log", skip_first=False
    )

    cs_orig = extract_scores(
        os.path.join(orig_dir, "commonsense_eval.log"), skip_first=True
    )
    rt_orig = extract_scores(
        os.path.join(orig_dir, "retrieval_eval.log"), skip_first=False
    )

    cs_base = average_tasks(cs_orig)
    rt_base = average_tasks(rt_orig)

    # Save JSONs
    cs_json = os.path.join(root, f"{args.model}_commonsense_avg.json")
    rt_json = os.path.join(root, f"{args.model}_retrieval_avg.json")

    with open(cs_json, "w") as f:
        json.dump(cs_layers, f, indent=4)

    with open(rt_json, "w") as f:
        json.dump(rt_layers, f, indent=4)

    combined = combine(
        cs_layers["values"],
        rt_layers["values"],
        cs_base,
        rt_base,
        args.mode,
    )

    # Plot
    os.makedirs("figures", exist_ok=True)
    fig = f"figures/{args.model}_importance_{args.mode}.png"

    plt.figure()
    plt.plot(cs_layers["contexts"], combined, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Importance")
    plt.title(f"{args.model} ({args.mode})")
    plt.grid(True)
    plt.savefig(fig)
    plt.close()

    # Ranking
    order = np.argsort(combined)[::-1]
    print("\nRank | Layer | Value")
    print("-----|-------|----------------")
    for i, idx in enumerate(order):
        print(f"{i+1:4d} | {cs_layers['contexts'][idx]:5d} | {combined[idx]:.10f}")

    print(f"\n📦 Commonsense baseline: {cs_base:.6f}")
    print(f"📦 Retrieval baseline:   {rt_base:.6f}")
    print(f"📊 Saved plot → {fig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["jet-nemotron", "HALO"],
        help="Combination method: jet-nemotron or HALO",
    )
    args = ap.parse_args()
    main(args)
