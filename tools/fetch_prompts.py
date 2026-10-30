"""Every prompt of the paper's corpora fetched back from its public dataset, by the row id the labels record.

    python -m tools.fetch_prompts --labels data/labels/qwen_train.jsonl [LABELS.jsonl ...] --out PROMPTS.jsonl

The loaders below are the ones that built the paper's corpus, unchanged: each reads its dataset from the Hugging Face
Hub, forms the prompt as the corpus did (SQuAD and GSM8K in fixed templates, MixInstruct as instruction and input,
the first user turn of an English LMSYS-Chat-1M conversation, BeaverTails' harmful prompts, four reward-bench subsets,
CodeUltraFeedback's instructions), and the corpus kept each prompt with its surrounding whitespace removed. Every
label's prompt is found by its prompt_sha256 among its dataset's prompts (the
labels' `source` names the row, but a few rows of the Gemma corpus carry another row's id, so the sha256 decides).
PROMPTS holds one {"prompt_sha256", "prompt"} line per distinct prompt, for `tools/labels.py join`. A prompt that is
not found (a dataset changed on the Hub) is reported and fails the run. LMSYS-Chat-1M is gated: the Hub token's account must have accepted its
licence. It needs pandas, pyarrow, datasets and huggingface_hub, in an environment of its own (the scheduler's pins an
older huggingface_hub); it fetched all 18,498 prompts of data/labels/ with pandas 2.3.3, pyarrow 25.0.1, datasets 5.1.0
and huggingface_hub 2.4.0.
"""
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
from typing import List, Optional

import pandas as pd

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

# the dataset locations the corpus was built from
datapath = SimpleNamespace(
    REWARD_BENCH_BASE="hf://datasets/allenai/reward-bench/",
    REWARD_BENCH_SPLITS={"raw": "data/raw-00000-of-00001.parquet", "filtered": "data/filtered-00000-of-00001.parquet"},
    REWARD_BENCH_DEFAULT_PREFIXES=["xstest-", "refusals-", "donotanswer", "hep-"],
    CODE_ULTRA_FEEDBACK_PATH="hf://datasets/coseal/CodeUltraFeedback/data/train-00000-of-00001.parquet",
    MIX_INSTRUCT_TRAIN_PATH="hf://datasets/llm-blender/mix-instruct/train_data_prepared.jsonl",
    BEAVER_TAILS_30K_TRAIN_PATH="hf://datasets/PKU-Alignment/BeaverTails/round0/30k/train.jsonl.gz",
    GSM8K_DATASET_NAME="openai/gsm8k",       # the corpus loaded it as "gsm8k", its earlier Hub name
    SQUAD_DATASET_NAME="rajpurkar/squad",    # likewise "squad"
    LMSYS_DATASET_NAME="lmsys/lmsys-chat-1m",
)

def load_reward_bench(
    split: str = "filtered",
    include_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load reward-bench from HF and select subsets by prefix.

    Args:
        split: One of {"raw", "filtered"}
        include_prefixes: Subset name prefixes to keep. If None, uses
            ["xstest-", "refusals-", "donotanswer", "hep-"]

    Returns:
        DataFrame with columns ["id", "prompt"].
    """
    if include_prefixes is None:
        include_prefixes = datapath.REWARD_BENCH_DEFAULT_PREFIXES

    if split not in datapath.REWARD_BENCH_SPLITS:
        raise ValueError(f"Invalid reward-bench split: {split}")

    path = datapath.REWARD_BENCH_BASE + datapath.REWARD_BENCH_SPLITS[split]
    df = pd.read_parquet(path)

    mask = False
    for p in include_prefixes:
        mask = mask | df["subset"].str.startswith(p)
    selected = df.loc[mask].copy()
    selected["id"] = selected["subset"] + "/" + selected["id"].astype(str)
    return selected[["id", "prompt"]]


def load_code_ultra_feedback(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load CodeUltraFeedback and sample instructions as prompts.

    Returns DataFrame with ["id", "prompt"].
    """
    path = datapath.CODE_ULTRA_FEEDBACK_PATH
    df = pd.read_parquet(path)
    sdf = df[["instruction"]].reset_index().copy()
    sdf["id"] = "code_ultra_feedback/" + sdf["index"].astype(str)
    sdf = sdf[["id", "instruction"]].rename(columns={"instruction": "prompt"})
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_mix_instruct(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load mix-instruct and form prompt as instruction + input.

    Returns DataFrame with ["id", "prompt"].
    """
    path: str = datapath.MIX_INSTRUCT_TRAIN_PATH
    df = pd.read_json(path, lines=True)
    sdf = df[["id", "instruction", "input"]].copy()
    # Concatenate with a space like the notebook
    sdf["prompt"] = sdf["instruction"].fillna("") + " " + sdf["input"].fillna("")
    sdf = sdf[["id", "prompt"]]
    if sample_n is not None and sample_n > 0:
        sdf = sdf.sample(n=sample_n, random_state=seed)
    return sdf


def load_beaver_tails(sample_n: Optional[int] = None, seed: int = 1) -> pd.DataFrame:
    """Load BeaverTails harmful prompts only and sample.

    Returns DataFrame with ["id", "prompt"].
    """
    path: str = datapath.BEAVER_TAILS_30K_TRAIN_PATH
    df = pd.read_json(path, lines=True)

    def is_prompt_harmful(category: dict) -> bool:
        # True if any category flag is True
        for _, v in category.items():
            if v:
                return True
        return False

    harmful = df[df["category"].apply(is_prompt_harmful)].copy()
    harmful = harmful[["prompt", "category"]].drop_duplicates(subset=["prompt"]).reset_index()
    harmful["id"] = "beaver_tails/" + harmful["index"].astype(str)
    # Keep category dict so downstream can confirm harm type
    harmful = harmful[["id", "prompt", "category"]]
    if sample_n is not None and sample_n > 0:
        harmful = harmful.sample(n=sample_n, random_state=seed)
    return harmful


def load_lmsys(
    sample_n: Optional[int] = None,
    seed: int = 1,
    lang: str = "English",
) -> pd.DataFrame:
    """Load LMSYS-Chat-1M conversations. Extract first user→assistant turn pair.

    Args:
        lang: Language filter. "all" keeps everything, otherwise filters by
              the 'language' field (default: "English").

    Returns DataFrame with ["id", "prompt", "response"].
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "datasets is required for LMSYS loading. Install with: pip install datasets"
        )

    # Load all parquet shards via huggingface_hub (avoids gated dataset issues)
    from huggingface_hub import hf_hub_download
    dfs = []
    shard_files = [
        f"data/train-{i:05d}-of-00006" for i in range(6)
    ]
    # List actual filenames from the repo
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(datapath.LMSYS_DATASET_NAME, repo_type="dataset")
    parquet_files = [f for f in all_files if f.startswith("data/") and f.endswith(".parquet")]

    for pf in parquet_files:
        local = hf_hub_download(
            repo_id=datapath.LMSYS_DATASET_NAME,
            filename=pf,
            repo_type="dataset",
        )
        dfs.append(pd.read_parquet(local))

    df = pd.concat(dfs, ignore_index=True)

    # Language filter
    if lang.lower() != "all" and "language" in df.columns:
        df = df[df["language"].str.lower() == lang.lower()].reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        convs = row.get("conversation", [])
        if convs is None or len(convs) < 2:
            continue

        prompt_text = None
        response_text = None
        for msg in convs:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role == "user" and prompt_text is None:
                prompt_text = content
            elif role == "assistant" and prompt_text is not None and response_text is None:
                response_text = content
                break

        if not prompt_text or not response_text:
            continue

        records.append({
            "id": f"lmsys/{row.get('conversation_id', idx)}",
            "prompt": prompt_text,
            "response": response_text,
        })

    result = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(result) > 0:
        result = result.sample(n=min(sample_n, len(result)), random_state=seed)
    return result


def load_gsm8k(
    sample_n: Optional[int] = None,
    seed: int = 1,
    split: str = "train",
    config: str = "main",
) -> pd.DataFrame:
    """Load GSM8K math word problems; use 'question' as prompt.

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError(
            "datasets is required for gsm8k loading. Install with: pip install datasets"
        )

    # gsm8k requires a config: 'main' or 'socratic'. Default to 'main'.
    try:
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, config, split=split)
    except Exception:
        # Fallback to 'socratic' if 'main' is unavailable in this mirror
        ds = load_dataset(datapath.GSM8K_DATASET_NAME, "socratic", split=split)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        prompt = (
            f"Solve the following problem. Show your reasoning briefly and end with the final numeric answer.\n"
            f"Problem: {q}\n"
            f"Answer with only the final number on the last line."
        )
        records.append({"id": f"gsm8k/{idx}", "prompt": prompt})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df


def load_squad(sample_n: Optional[int] = None, seed: int = 1, split: str = "train") -> pd.DataFrame:
    """Load SQuAD v1.1; build QA prompts with context.

    Returns DataFrame with ["id", "prompt"].
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets is required for SQuAD. pip install datasets")

    ds = load_dataset(datapath.SQUAD_DATASET_NAME, split=split)
    records = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", "")).strip()
        if not q:
            continue
        prompt = (
            "Answer the question based on the context.\n"
            f"Context: {ctx}\n"
            f"Question: {q}"
        )
        records.append({"id": f"squad/{idx}", "prompt": prompt})
    df = pd.DataFrame(records)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    return df

LOADERS = {"reward_bench": load_reward_bench, "code_ultra_feedback": load_code_ultra_feedback,
           "mix_instruct": load_mix_instruct, "beaver_tails": load_beaver_tails, "lmsys": load_lmsys,
           "gsm8k": load_gsm8k, "squad": load_squad}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, nargs="+", help="label files (tools/labels.py extract)")
    ap.add_argument("--out", required=True, help="a new JSONL file")
    a = ap.parse_args(argv)
    want = {}          # prompt sha256 -> dataset
    for path in a.labels:
        with open(path) as f:
            for line in f:
                lab = json.loads(line)
                want.setdefault(lab["prompt_sha256"], lab["dataset"])
    found, bad = {}, 0
    for ds in sorted(set(want.values())):
        if ds not in LOADERS:
            print(f"no loader for dataset {ds}", file=sys.stderr)
            return 1
        needed = {sha for sha, d in want.items() if d == ds}
        for p in LOADERS[ds]()["prompt"]:
            p = p.strip()      # the corpus stores every prompt with its surrounding whitespace removed
            sha = hashlib.sha256(p.encode()).hexdigest()
            if sha in needed:
                found[sha] = p
        n_found = len(needed & set(found))
        print(f"{ds}: {len(needed)} prompts, {n_found} fetched, {len(needed) - n_found} not found")
        bad += len(needed) - n_found
    if bad:
        print(f"{bad} prompts are not in the datasets as the corpus held them", file=sys.stderr)
        return 1
    with open(a.out, "x") as f:
        for sha, p in found.items():
            f.write(json.dumps({"prompt_sha256": sha, "prompt": p}) + "\n")
    print(f"{a.out}: {len(found)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
