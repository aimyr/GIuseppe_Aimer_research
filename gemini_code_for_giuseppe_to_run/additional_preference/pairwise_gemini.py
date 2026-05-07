"""Pairwise human-vs-machine preference judging using Gemini 2.5 Flash.

For every (human_col, machine_col) pair the judge is asked three times:
  - NORMAL1: Abstract 1 = human, Abstract 2 = machine
  - SWAP   : Abstract 1 = machine, Abstract 2 = human
  - NORMAL2: Abstract 1 = human, Abstract 2 = machine (consistency check)

Usage:
    python pairwise_gemini.py --csv old_preference/abstracts_original.csv
    python pairwise_gemini.py --csv old_preference/arguments_original.csv --first_n_rows 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google import genai
from google.genai import types

os.environ.setdefault('TOGETHER_API_KEY', '')
os.environ.setdefault('OPENAI_API_KEY', '')
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")


JUDGE_NAME = "gemini"
MODEL = "gemini-2.5-flash"
RESULTS_DIR = Path("pairwise_results")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def detect_human_and_machine_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    human_cols = [c for c in ["winning_text", "losing_text"] if c in df.columns]
    excluded = {"id", "theme", "proposition", "row_id", "source_row", "custom_id", "item_id"}
    machine_cols: List[str] = []
    for c in df.columns:
        if c in excluded or c in human_cols:
            continue
        if any(k in c.lower() for k in ["paraphrase", "improve", "generate"]):
            machine_cols.append(c)
    return human_cols, machine_cols


def detect_task_kind(csv_path: Path) -> str:
    """`arguments` if the CSV filename suggests debate arguments, else `abstracts`."""
    return "arguments" if "argument" in csv_path.stem.lower() else "abstracts"


def build_prompt(theme: str, text_1: str, text_2: str, kind: str = "abstracts") -> str:
    if kind == "arguments":
        return (
            "You are an expert debate judge. I will provide a debate proposition and two arguments. "
            "Your task is to decide which argument is better written, more persuasive, and better structured.\n\n"
            f"Proposition: {theme}\n\n"
            f"Argument 1:\n{text_1}\n\n"
            f"Argument 2:\n{text_2}\n\n"
            "Which argument is better written? "
            'Answer with ONLY "1" or "2".'
        )
    return (
        "You are an expert researcher. I will provide an abstract topic and two abstracts. "
        "Your task is to decide which abstract is better written, more logical, and better structured.\n\n"
        f"Topic: {theme}\n\n"
        f"Abstract 1:\n{text_1}\n\n"
        f"Abstract 2:\n{text_2}\n\n"
        "Which abstract is better written? "
        'Answer with ONLY "1" or "2".'
    )


def resolve_choice(answer_text: str, source_when_1: str, source_when_2: str) -> str:
    ans = str(answer_text).strip().replace('"', "").replace("'", "")
    if ans == "1":
        return source_when_1
    if ans == "2":
        return source_when_2
    return "Error"


def prepare_inline_requests(
    df: pd.DataFrame,
    kind: str = "abstracts",
) -> Tuple[List[Any], List[str], List[str], List[str]]:
    """Returns (inline_requests, ordered_custom_ids, human_cols, machine_cols)."""
    human_cols, machine_cols = detect_human_and_machine_columns(df)
    print(f"Task kind: {kind}")
    print(f"Human columns: {human_cols}")
    print(f"Machine columns ({len(machine_cols)}): {machine_cols}")

    inline_requests: List[Any] = []
    ordered_cids: List[str] = []

    for idx, row in df.iterrows():
        theme = str(row.get("theme", row.get("proposition", ""))).strip()
        for human_col in human_cols:
            for machine_col in machine_cols:
                val_h = row.get(human_col)
                val_m = row.get(machine_col)
                if pd.isna(val_h) or pd.isna(val_m):
                    continue
                text_h = str(val_h).strip()
                text_m = str(val_m).strip()
                if not text_h or not text_m:
                    continue

                prompt_normal = build_prompt(theme, text_h, text_m, kind)
                prompt_swap = build_prompt(theme, text_m, text_h, kind)

                for tag, prompt in [
                    ("NORMAL1", prompt_normal),
                    ("SWAP", prompt_swap),
                    ("NORMAL2", prompt_normal),
                ]:
                    cid = f"{idx}--{human_col}--{machine_col}--{tag}"
                    ordered_cids.append(cid)
                    inline_requests.append(types.InlinedRequest(
                        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=ANSWER_SCHEMA,
                            max_output_tokens=20,
                        ),
                    ))

    print(f"Prepared {len(inline_requests)} inline requests")
    return inline_requests, ordered_cids, human_cols, machine_cols


def _load_inlined_from_jsonl(output_jsonl: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_batch(
    client,
    inline_requests: List[Any],
    output_jsonl: Path,
    job_name_path: Path,
    poll_seconds: int = 30,
    max_wait_seconds: int = 24 * 3600,
) -> List[Any]:
    """Submit (or resume) a Gemini inline batch.
    If output_jsonl is already populated, reload it. If job_name_path has
    a saved job name, resume polling instead of re-submitting."""
    if output_jsonl.exists() and output_jsonl.stat().st_size > 0:
        print(f"Raw results already present at {output_jsonl}; reloading.")
        return _load_inlined_from_jsonl(output_jsonl)

    job_name = job_name_path.read_text().strip() if job_name_path.exists() else ""

    if job_name:
        print(f"Resuming saved Gemini job: {job_name}")
    else:
        job = client.batches.create(
            model=MODEL,
            src=inline_requests,
            config={"display_name": f"{JUDGE_NAME}-pairwise"},
        )
        job_name = job.name
        # Persist immediately for resume safety.
        job_name_path.write_text(job_name, encoding="utf-8")
        print(f"Batch job saved to {job_name_path}: {job_name}")

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    start = time.time()
    final = None
    while True:
        j = client.batches.get(name=job_name)
        state = getattr(j.state, "name", str(j.state))
        print(f"Batch state: {state}")
        if state in completed_states:
            final = j
            break
        if time.time() - start > max_wait_seconds:
            raise RuntimeError("Timed out waiting for Gemini batch job")
        time.sleep(poll_seconds)

    if getattr(final.state, "name", str(final.state)) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Job ended with state {getattr(final.state, 'name', final.state)}")

    dest = getattr(final, "dest", None)
    inlined = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined:
        raise RuntimeError("Succeeded but no inlined_responses on job")

    # Persist a normalized JSONL log.
    raw_lines = []
    for item in inlined:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        raw_lines.append(json.dumps(d, ensure_ascii=False, default=str))
    output_jsonl.write_text("\n".join(raw_lines), encoding="utf-8")
    print(f"Raw responses written to {output_jsonl}")
    return list(inlined)


def _parse_answer(item: Any) -> str:
    d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
    if d.get("error"):
        return "API_ERROR"
    response = d.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return "API_ERROR"
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") if isinstance(p, dict) else "" for p in parts).strip()
    if not text:
        return "PARSE_ERROR"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except json.JSONDecodeError:
        pass
    return text


def process_results(
    original_df: pd.DataFrame,
    inlined: List[Any],
    ordered_cids: List[str],
    output_csv: Path,
    human_cols: List[str],
    machine_cols: List[str],
) -> None:
    print("Parsing results...")
    results_map: Dict[str, str] = {}
    for cid, item in zip(ordered_cids, inlined):
        results_map[cid] = _parse_answer(item)

    final_rows: List[Dict[str, Any]] = []
    for idx, row in original_df.iterrows():
        item_id = row.get("id", idx)
        theme = row.get("theme", row.get("proposition", ""))
        for human_col in human_cols:
            for machine_col in machine_cols:
                cid_n1 = f"{idx}--{human_col}--{machine_col}--NORMAL1"
                cid_sw = f"{idx}--{human_col}--{machine_col}--SWAP"
                cid_n2 = f"{idx}--{human_col}--{machine_col}--NORMAL2"

                ans_n1 = results_map.get(cid_n1, "MISSING")
                ans_sw = results_map.get(cid_sw, "MISSING")
                ans_n2 = results_map.get(cid_n2, "MISSING")
                if ans_n1 == "MISSING" and ans_sw == "MISSING" and ans_n2 == "MISSING":
                    continue

                chosen_n1 = resolve_choice(ans_n1, human_col, machine_col)
                chosen_sw = resolve_choice(ans_sw, machine_col, human_col)
                chosen_n2 = resolve_choice(ans_n2, human_col, machine_col)

                bad = {"API_ERROR", "PARSE_ERROR", "MISSING"}
                any_err = (
                    ans_n1 in bad or ans_sw in bad or ans_n2 in bad
                    or chosen_n1 == "Error" or chosen_sw == "Error" or chosen_n2 == "Error"
                )
                if any_err:
                    self_consistent = "Error"
                    order_influenced = "Error"
                else:
                    self_consistent = "Yes" if chosen_n1 == chosen_n2 else "No"
                    order_influenced = "No" if chosen_n1 == chosen_sw else "Yes"

                final_rows.append({
                    "row_id": item_id,
                    "theme": theme,
                    "A_source": human_col,
                    "B_source": machine_col,
                    "normal1_final_answer": ans_n1,
                    "swapped_final_answer": ans_sw,
                    "normal2_final_answer": ans_n2,
                    "chosen_source_normal1": chosen_n1,
                    "chosen_source_swapped": chosen_sw,
                    "chosen_source_normal2": chosen_n2,
                    "self_consistent": self_consistent,
                    "order_influenced_decision": order_influenced,
                })

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved processed results to {output_csv}")
    print(f"Total parsed rows: {len(out_df)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--first_n_rows", type=int, default=None)
    args = p.parse_args()

    csv_path = Path(args.csv)
    stem = csv_path.stem
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = RESULTS_DIR / f"{stem}__{JUDGE_NAME}.jsonl"
    output_csv = RESULTS_DIR / f"{stem}__{JUDGE_NAME}.csv"
    job_name_path = RESULTS_DIR / f"{stem}__{JUDGE_NAME}.job_name"

    df = pd.read_csv(csv_path)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows)

    kind = detect_task_kind(csv_path)
    # Always rebuild ordered_cids so we can map results back to (idx, cols)
    # — they're a deterministic function of the CSV, so this is free.
    inline_requests, ordered_cids, human_cols, machine_cols = prepare_inline_requests(df, kind=kind)
    if not ordered_cids:
        print("No requests; aborting.")
        return

    api_key = os.environ["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
    inlined = run_batch(client, inline_requests, output_jsonl, job_name_path)
    process_results(df, inlined, ordered_cids, output_csv, human_cols, machine_cols)


if __name__ == "__main__":
    main()
