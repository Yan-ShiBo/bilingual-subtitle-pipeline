from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LLM_ROLE_NAMES = ("metadata", "translation", "proofread")


@dataclass(frozen=True)
class LlmGenerationProfile:
    temperature: float
    top_p: float
    top_k: int
    seed: int
    num_ctx: int
    max_tokens: int
    think: bool = False


GENERATION_PROFILES = {
    "metadata": LlmGenerationProfile(
        temperature=0.0,
        top_p=0.8,
        top_k=20,
        seed=41,
        num_ctx=8192,
        max_tokens=512,
    ),
    "translation": LlmGenerationProfile(
        temperature=0.15,
        top_p=0.8,
        top_k=20,
        seed=42,
        num_ctx=32768,
        max_tokens=4096,
    ),
    "proofread": LlmGenerationProfile(
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        seed=43,
        num_ctx=65536,
        max_tokens=4096,
    ),
}


def generation_profile(role: str) -> LlmGenerationProfile:
    try:
        return GENERATION_PROFILES[role]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM role: {role}") from exc


def _terminology_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        },
    }


def indexed_subtitle_schema(expected_count: int, role: str) -> dict[str, Any]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if role == "translation":
        properties = {
            "index": {
                "type": "integer",
                "minimum": 0,
                "maximum": expected_count - 1,
            },
            "corrected_text": {"type": "string"},
            "chinese_translation": {"type": "string"},
            "display": {"type": "boolean"},
            "terminology": _terminology_schema(),
        }
        required = [
            "index",
            "corrected_text",
            "chinese_translation",
            "display",
            "terminology",
        ]
    elif role == "proofread":
        properties = {
            "index": {
                "type": "integer",
                "minimum": 0,
                "maximum": expected_count - 1,
            },
            "corrected_english": {"type": "string"},
            "corrected_chinese": {"type": "string"},
            "display": {"type": "boolean"},
            "terminology": _terminology_schema(),
        }
        required = [
            "index",
            "corrected_english",
            "corrected_chinese",
            "display",
            "terminology",
        ]
    else:
        raise ValueError(f"Unsupported indexed subtitle schema role: {role}")

    return {
        "type": "array",
        "minItems": expected_count,
        "maxItems": expected_count,
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def movie_name_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "series_name": {"type": "string"},
            "movie_name": {"type": "string"},
        },
        "required": ["series_name", "movie_name"],
        "additionalProperties": False,
    }


def movie_style_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "register": {"type": "string"},
            "name_policy": {"type": "string"},
            "address_policy": {"type": "string"},
            "sdh_policy": {"type": "string"},
            "punctuation_policy": {"type": "string"},
            "terminology": _terminology_schema(),
        },
        "required": [
            "register",
            "name_policy",
            "address_policy",
            "sdh_policy",
            "punctuation_policy",
            "terminology",
        ],
        "additionalProperties": False,
    }
