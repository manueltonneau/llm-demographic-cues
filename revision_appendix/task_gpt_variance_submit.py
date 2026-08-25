#!/usr/bin/env python3
"""
Submit / poll / retrieve the T6 GPT-5.2 variance batch (OpenAI Batch API).

This SPENDS money. Run only after `task_gpt_variance_build.py --emit` has written
gpt_variance_requests.jsonl. Reads the API key from a file (never printed).

    python task_gpt_variance_submit.py submit  --key /path/to/key   # upload + create batch
    python task_gpt_variance_submit.py status  --key /path/to/key   # check progress
    python task_gpt_variance_submit.py fetch   --key /path/to/key   # download output when done

State is kept in gpt_variance_batch.json (batch id, file ids).
"""
import argparse
import json
import os
import sys

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(HERE, "gpt_variance_requests.jsonl")
STATE = os.path.join(HERE, "gpt_variance_batch.json")
OUT = os.path.join(HERE, "gpt_variance_output.jsonl")


def client(key_path):
    with open(key_path) as fh:
        return OpenAI(api_key=fh.read().strip())


def submit(c):
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        print(f"[abort] {STATE} already exists (batch {st.get('batch_id')}). "
              f"Delete it to resubmit.")
        return
    n = sum(1 for _ in open(REQ))
    print(f"uploading {REQ} ({n:,} requests)...")
    f = c.files.create(file=open(REQ, "rb"), purpose="batch")
    b = c.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                         completion_window="24h",
                         metadata={"job": "gpt52_run_to_run_variance", "n_requests": str(n)})
    json.dump({"batch_id": b.id, "input_file_id": f.id, "n_requests": n}, open(STATE, "w"), indent=2)
    print(f"submitted. batch_id={b.id} status={b.status}")
    print(f"state saved -> {STATE}")


def status(c):
    st = json.load(open(STATE))
    b = c.batches.retrieve(st["batch_id"])
    rc = b.request_counts
    print(f"batch {b.id}: {b.status} | completed {rc.completed}/{rc.total} failed {rc.failed}")
    if b.status == "completed":
        print(f"  output_file_id={b.output_file_id}  -> run: fetch")
    if b.status == "failed":
        print(f"  errors: {b.errors}")
    return b


def fetch(c):
    st = json.load(open(STATE))
    b = c.batches.retrieve(st["batch_id"])
    if b.status != "completed":
        print(f"[wait] batch is {b.status}, not completed yet.")
        return
    content = c.files.content(b.output_file_id).read()
    with open(OUT, "wb") as fh:
        fh.write(content)
    print(f"wrote {OUT} ({len(content):,} bytes). Now parse:")
    print("  python task_gpt_variance_build.py --parse gpt_variance_output.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "status", "fetch"])
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    c = client(a.key)
    {"submit": submit, "status": status, "fetch": fetch}[a.action](c)
