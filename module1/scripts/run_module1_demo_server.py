"""Small local web server for trying Module 1 in a browser."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import tempfile
from urllib.parse import unquote, urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.concept_coverage.concepts_reference import build_concept_reference_from_model_answers
from module1.module1_features import build_module1_features


MODULE1_ROOT = PROJECT_ROOT / "module1"
STATIC_DIR = MODULE1_ROOT / "web_demo"
MODEL_PATH = MODULE1_ROOT / "models" / "concept_coverage_model"


class Module1DemoHandler(BaseHTTPRequestHandler):
    """Serve the demo page and Module 1 preview API."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        requested_path = parsed.path
        if requested_path in {"", "/"}:
            requested_path = "/index.html"

        static_path = (STATIC_DIR / unquote(requested_path.lstrip("/"))).resolve()
        if not str(static_path).startswith(str(STATIC_DIR.resolve())):
            self.send_error(403)
            return
        if not static_path.exists() or static_path.is_dir():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(static_path)
        body = static_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/module1-preview":
            self.send_error(404)
            return

        try:
            payload = self.read_json_payload()
            response = run_module1_preview(payload)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:  # pragma: no cover - surfaced clearly in the UI.
            self.write_json({"error": f"Module 1 failed: {exc}"}, status=500)
            return

        self.write_json(response)

    def read_json_payload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def write_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[module1-demo] {self.address_string()} {format % args}")


def run_module1_preview(payload: dict[str, object]) -> dict[str, object]:
    student_answer = str(payload.get("student_answer", "")).strip()
    model_answer = str(payload.get("model_answer", "")).strip()
    question = str(payload.get("question", "")).strip() or "What is the concept?"
    use_trained_model = bool(payload.get("use_trained_model", True))

    if not student_answer:
        raise ValueError("Student answer is required.")
    if not model_answer:
        raise ValueError("Schema / model answer is required.")

    demo_row = {
        "question_id": "DEMO_Q1",
        "question": question,
        "synthetic_answer": student_answer,
        "scheme": model_answer,
        "ai_score": 3.0,
    }
    dataframe = pd.DataFrame([demo_row])
    concepts = build_concept_reference_from_model_answers(
        pd.DataFrame(
            [
                {
                    "question_id": "DEMO_Q1",
                    "question": question,
                    "model_answer": model_answer,
                }
            ]
        )
    )
    if concepts.empty:
        raise ValueError("Schema / model answer must contain at least one concept or sentence.")

    concept_backend = "trained-llm" if use_trained_model and MODEL_PATH.exists() else "weak-score"
    fallback_reason = ""
    with tempfile.TemporaryDirectory(prefix="cleanstream_module1_demo_") as temp_dir:
        concept_reference = Path(temp_dir) / "concepts.csv"
        concepts.to_csv(concept_reference, index=False)
        try:
            output = build_module1_features(
                dataframe,
                model_answer_column="scheme",
                student_answer_column="synthetic_answer",
                concept_reference_path=concept_reference,
                concept_backend=concept_backend,
                concept_model_path=MODEL_PATH,
                language_check_backend="simple",
                require_model_answer=True,
            )
        except (FileNotFoundError, ImportError) as exc:
            if concept_backend != "trained-llm":
                raise
            fallback_reason = str(exc)
            output = build_module1_features(
                dataframe,
                model_answer_column="scheme",
                student_answer_column="synthetic_answer",
                concept_reference_path=concept_reference,
                concept_backend="weak-score",
                language_check_backend="simple",
                require_model_answer=True,
            )

    row = normalize_for_json(output.iloc[0].to_dict())
    return {
        "summary": build_result_summary(row, fallback_reason),
        "concepts": {
            "present": split_concept_cell(row.get("concepts_present")),
            "partial": split_concept_cell(row.get("concepts_partial")),
            "missing": split_concept_cell(row.get("concepts_missing")),
        },
        "raw": row,
    }


def build_result_summary(row: dict[str, object], fallback_reason: str) -> dict[str, object]:
    return {
        "concept_backend": row.get("concept_backend"),
        "concept_fallback_reason": fallback_reason,
        "concept_coverage_ratio": row.get("concept_coverage_ratio"),
        "semantic_similarity_score": row.get("semantic_similarity_score"),
        "reasoning_quality": row.get("reasoning_quality"),
        "reasoning_connective_count": row.get("reasoning_connective_count"),
        "contradiction_check_applied": row.get("contradiction_check_applied"),
        "contradiction_detected": row.get("contradiction_detected"),
        "language_quality_score": row.get("language_quality_score"),
        "spelling_error_count": row.get("spelling_error_count"),
        "grammar_error_count": row.get("grammar_error_count"),
        "cross_question_flag": row.get("cross_question_flag"),
        "answer_word_count": row.get("answer_word_count"),
    }


def split_concept_cell(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("; ") if part.strip()]


def normalize_for_json(record: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = None
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Cleanstream Module 1 test page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Module1DemoHandler)
    print(f"Cleanstream Module 1 demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
