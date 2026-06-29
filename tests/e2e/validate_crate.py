"""Validate a real generated RO-Crate for the e2e workflow.

Usage: python validate_crate.py <ro-crate-metadata.json> <expected-strategy> <expected-framework-slug>
Exits non-zero (failing the CI job) if the crate is missing or incomplete.
"""

import json
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <metadata.json> <strategy> <framework-slug>")
    path, expected_strategy, expected_fw = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        graph = json.loads(open(path).read())["@graph"]
    except FileNotFoundError:
        sys.exit(f"FAIL: crate not found at {path} (did the run produce one?)")
    g = {e["@id"]: e for e in graph}

    checks = {
        "root mentions the run (#1)": g["./"].get("mentions") == [{"@id": "#fl-run"}],
        "run completed": g.get("#fl-run", {}).get("actionStatus", {}).get("@id", "").endswith("CompletedActionStatus"),
        f"strategy is {expected_strategy} (#2)": g.get("#fl-strategy", {}).get("name") == expected_strategy,
        f"framework #framework-{expected_fw} captured (#4)": f"#framework-{expected_fw}" in g,
        "at least one metric (#3)": any(k.startswith("#metric") for k in g),
        "endTime set (record_result ran)": bool(g.get("#fl-run", {}).get("endTime")),
        "license + author + agent (#5)": all([
            g["./"].get("license"), g["./"].get("author"), g.get("#fl-run", {}).get("agent"),
        ]),
    }

    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        sys.exit(f"FAIL: {len(failed)} check(s) failed: {failed}")
    print(f"OK: valid end-to-end crate ({expected_strategy} / {expected_fw}).")


if __name__ == "__main__":
    main()
