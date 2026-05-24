import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from prompt import DECOMPOSITION_CIRR_PROMPT, build_user_prompt


def encode_image(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    with open(image_path, "rb") as image_file:
        payload = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def get_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for GPT decomposition.") from exc
    if azure_endpoint and azure_key:
        return AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))


def make_messages(image_path: str, instruction: str, dataset: str, detail: str, shared_concept: Optional[str]):
    return [
        {"role": "system", "content": DECOMPOSITION_CIRR_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(instruction, dataset, shared_concept)},
                {"type": "image_url", "image_url": {"url": encode_image(image_path), "detail": detail}},
            ],
        },
    ]


def call_gpt(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: Optional[int],
    json_response_format: bool,
) -> str:
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }
    if json_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def strip_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def clean_phrase(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;,\n\t")


def clean_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for item in values:
        phrase = clean_phrase(item)
        key = phrase.lower()
        if phrase and key not in seen:
            cleaned.append(phrase)
            seen.add(key)
    return cleaned


def clean_pairs(values: Any) -> List[Dict[str, str]]:
    if not isinstance(values, list):
        return []
    pairs = []
    seen = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        desired = clean_phrase(item.get("desired"))
        confusing = clean_phrase(item.get("confusing"))
        key = (desired.lower(), confusing.lower())
        if desired and confusing and key not in seen:
            pairs.append({"desired": desired, "confusing": confusing})
            seen.add(key)
    return pairs


def parse_decomposition(raw_text: str) -> Dict[str, Any]:
    data = json.loads(strip_json_text(raw_text))
    return {
        "reference_image_description": clean_phrase(data.get("reference_image_description")),
        "positive_target": clean_phrase(data.get("positive_target")),
        "negative_constraints": clean_list(data.get("negative_constraints")),
        "contrastive_pairs": clean_pairs(data.get("contrastive_pairs")),
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_split(split: str) -> str:
    return "test1" if split == "test" else split


def load_cirr(dataset_path: Path, split: str) -> List[Dict[str, Any]]:
    split_name = resolve_split(split)
    triplets = read_json(dataset_path / "cirr" / "captions" / f"cap.rc2.{split_name}.json")
    name_to_relpath = read_json(dataset_path / "cirr" / "image_splits" / f"split.rc2.{split_name}.json")
    rows = []
    for index, item in enumerate(triplets):
        reference_name = item["reference"]
        row = {
            "sample_id": str(item.get("pairid", index)),
            "dataset": "cirr",
            "index": index,
            "reference_name": reference_name,
            "reference_image_path": str(dataset_path / name_to_relpath[reference_name]),
            "relative_caption": item["caption"],
            "group_members": item.get("img_set", {}).get("members", []),
        }
        if "target_hard" in item:
            row["target_name"] = item["target_hard"]
        if "pairid" in item:
            row["pair_id"] = item["pairid"]
        rows.append(row)
    return rows


def load_fashioniq(dataset_path: Path, split: str, dress_types: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for dress_type in dress_types:
        triplets = read_json(dataset_path / "captions" / f"cap.{dress_type}.{split}.json")
        for index, item in enumerate(triplets):
            captions = item.get("captions", [])
            instruction = " and ".join(clean_phrase(caption) for caption in captions if clean_phrase(caption))
            reference_name = item["candidate"]
            sample_id_parts = [dress_type, split, reference_name, str(index)]
            if "target" in item:
                sample_id_parts.insert(2, item["target"])
            row = {
                "sample_id": "_".join(sample_id_parts),
                "dataset": "fashioniq",
                "category": dress_type,
                "index": index,
                "reference_name": reference_name,
                "reference_image_path": str(dataset_path / "images" / f"{reference_name}.png"),
                "relative_caption": instruction,
                "relative_captions": captions,
            }
            if "target" in item:
                row["target_name"] = item["target"]
            rows.append(row)
    return rows


def load_circo(dataset_path: Path, split: str) -> List[Dict[str, Any]]:
    annotations = read_json(dataset_path / "annotations" / f"{split}.json")
    image_info_path = dataset_path / "COCO2017_unlabeled" / "annotations" / "image_info_unlabeled2017.json"
    if image_info_path.exists():
        image_info = read_json(image_info_path)
        id_to_file = {str(item["id"]): item["file_name"] for item in image_info["images"]}
    else:
        image_dir = dataset_path / "COCO2017_unlabeled" / "unlabeled2017"
        id_to_file = {str(int(path.stem)): path.name for path in image_dir.glob("*.jpg")}
        id_to_file.update({str(int(path.stem)): path.name for path in image_dir.glob("*.png")})
    rows = []
    for index, item in enumerate(annotations):
        reference_id = str(item["reference_img_id"])
        row = {
            "sample_id": str(item.get("id", index)),
            "dataset": "circo",
            "index": index,
            "query_id": str(item.get("id", index)),
            "reference_name": reference_id,
            "reference_image_path": str(dataset_path / "COCO2017_unlabeled" / "unlabeled2017" / id_to_file[reference_id]),
            "relative_caption": item["relative_caption"],
            "shared_concept": item.get("shared_concept"),
        }
        if "target_img_id" in item:
            row["target_name"] = str(item["target_img_id"])
        if "gt_img_ids" in item:
            row["gt_img_ids"] = [str(value) for value in item["gt_img_ids"]]
        rows.append(row)
    return rows


def load_queries(dataset: str, dataset_path: Path, split: str, fashioniq_types: List[str]) -> List[Dict[str, Any]]:
    key = dataset.lower()
    if key == "cirr":
        return load_cirr(dataset_path, split)
    if key == "circo":
        return load_circo(dataset_path, split)
    if key == "fashioniq":
        return load_fashioniq(dataset_path, split, fashioniq_types)
    if key.startswith("fashioniq_"):
        return load_fashioniq(dataset_path, split, [key.split("_", 1)[1]])
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_done_ids(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return done


def iter_selected(rows: List[Dict[str, Any]], start: int, limit: int) -> Iterable[Dict[str, Any]]:
    selected = rows[start:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def decompose_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    sample_id = str(row["sample_id"])
    image_path = row["reference_image_path"]
    instruction = row["relative_caption"]
    if not Path(image_path).exists():
        return {"sample_id": sample_id, "ok": False, "error": f"Missing image: {image_path}", **row}

    raw_response = ""
    parsed = None
    error = None
    client = get_client()
    messages = make_messages(image_path, instruction, row["dataset"], args.image_detail, row.get("shared_concept"))
    for attempt in range(args.max_retries + 1):
        try:
            raw_response = call_gpt(
                client=client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                json_response_format=args.json_response_format,
            )
            parsed = parse_decomposition(raw_response)
            error = None
            break
        except Exception as exc:
            error = str(exc)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    return {
        "sample_id": sample_id,
        "ok": error is None,
        "error": error,
        "raw_response": raw_response,
        "decomposition": parsed,
        **row,
    }


def run(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_queries(args.dataset, Path(args.dataset_path), args.split, args.fashioniq_types)
    done_ids = load_done_ids(output_path) if args.resume else set()
    selected_rows = [
        row for row in iter_selected(rows, args.start, args.limit)
        if str(row["sample_id"]) not in done_ids
    ]
    total = len(selected_rows)
    written = 0
    with output_path.open("a", encoding="utf-8") as output_file:
        if args.num_workers <= 1:
            for written, row in enumerate(selected_rows, start=1):
                result = decompose_row(row, args)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"[{written}/{total}] {result['sample_id']} ok={result['ok']}", flush=True)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(decompose_row, row, args) for row in selected_rows]
                for future in as_completed(futures):
                    result = future.result()
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    written += 1
                    print(f"[{written}/{total}] {result['sample_id']} ok={result['ok']}", flush=True)
    print(json.dumps({"selected": total, "written": written, "output": str(output_path)}, ensure_ascii=False))


def build_arg_parser():
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser("CoVer-CIR intent decomposition")
    parser.add_argument("--dataset", required=True, choices=["cirr", "circo", "fashioniq", "fashioniq_dress", "fashioniq_shirt", "fashioniq_toptee"])
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", default=cfg.dataset.default_split)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06"))
    parser.add_argument("--fashioniq-types", nargs="+", default=list(cfg.dataset.fashioniq_types), choices=list(cfg.dataset.fashioniq_types))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-detail", default="low", choices=["low", "high", "auto"])
    parser.add_argument("--json-response-format", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
