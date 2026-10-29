"""Safety Polytope runtime adapter for AdversariaLLM.

This adapter intentionally delegates steering to the official upstream
SafeRepModel. It does not reimplement or approximate the Safety Polytope
algorithm. The upstream package and a trained constraint artifact must be
installed/provided separately.
"""
from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..lm_utils.text_generation import GenerationResult, RetryOverrides
from ..types import Conversation
from .base import TargetSystem

_LOG = logging.getLogger(__name__)


def _last_user_text(conv: Conversation) -> str:
    for message in reversed(conv):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _render_conversation(tokenizer: Any, conv: Conversation) -> str:
    """Render messages with the checkpoint's own chat template when available."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
    # Fallback is deliberately explicit and role-labelled; it avoids silently
    # pretending a missing chat template exists.
    return "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in conv) + "\nASSISTANT:"


def _resolve_torch_dtype(value: Any, *, fallback: torch.dtype) -> Any:
    if value is None:
        return fallback
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "auto":
            return "auto"
        if hasattr(torch, lowered):
            candidate = getattr(torch, lowered)
            if isinstance(candidate, torch.dtype):
                return candidate
    raise ValueError(
        f"Unsupported SafetyPolytope torch_dtype value: {value!r}. "
        "Use 'auto', a torch dtype string like 'float16'/'bfloat16', or a torch.dtype."
    )


def _warn_on_transformers_major_mismatch() -> None:
    try:
        tf_version = version("transformers")
    except PackageNotFoundError:
        return
    except Exception:
        return
    try:
        major = int(tf_version.split(".", 1)[0])
    except Exception:
        return
    if major >= 5:
        _LOG.warning(
            "SafetyPolytope upstream code targets transformers==4.45.2, but this environment has %s. "
            "If generation fails with model-forward API errors, run SafetyPolytope in an isolated compatible "
            "environment or pin a known-compatible transformers version for this defense path.",
            tf_version,
        )


class SafetyPolytopeDefense(TargetSystem):
    """AdversariaLLM adapter around the official SafetyPolytope SafeRepModel.

    Required config keys:
      model_id: base model used to train the constraint
      weights_path: path to the trained upstream Safety Polytope .pth artifact

    Optional keys are forwarded to SafeRepModel or generation as appropriate.
    The model is loaded once by SafeRepModel; the `model` argument supplied by
    AdversariaLLM is retained as the framework model reference but is not used
    for generation. This means the base model is duplicated in memory. This is
    intentional for the compatibility adapter and is documented as a limitation.
    """

    NAME = "safety_polytope"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        safe_model: Any,
        *,
        model_id: str,
        weights_path: str,
        default_generate_kwargs: dict[str, Any] | None = None,
        max_input_tokens: int | None = None,
        max_new_tokens: int = 128,
        steer_layer: int | None = 20,
        steer_first_n_tokens: int = 20,
        lambda_weight: float = 2.0,
        safe_violation_weight: float = 0.0001,
        projection: bool = False,
        use_backup_response: bool = False,
        backup_response: str = "Sorry, I can't help with that request.",
    ) -> None:
        super().__init__(model, tokenizer)
        self.safe_model = safe_model
        self.safe_tokenizer = getattr(safe_model, "tokenizer", tokenizer)
        self.model_id = str(model_id)
        self.weights_path = str(weights_path)
        self.default_generate_kwargs = dict(default_generate_kwargs or {})
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = int(max_new_tokens)
        self.steer_layer = steer_layer
        self.steer_first_n_tokens = int(steer_first_n_tokens)
        self.lambda_weight = float(lambda_weight)
        self.safe_violation_weight = float(safe_violation_weight)
        self.projection = bool(projection)
        self.use_backup_response = bool(use_backup_response)
        self.backup_response = backup_response
        if not callable(getattr(safe_model, "generate", None)):
            raise TypeError("Loaded SafetyPolytope SafeRepModel has no callable generate().")

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_generate_kwargs: dict[str, Any] | None = None,
    ) -> "SafetyPolytopeDefense":
        model_cfg = cfg.get("model") or {}
        model_id = cfg.get("model_id") or model_cfg.get("id")
        weights_path = cfg.get("weights_path") or cfg.get("polytope_weight_path")
        if not model_id:
            raise ValueError("SafetyPolytope config requires 'model_id'.")
        if not weights_path:
            raise ValueError(
                "SafetyPolytope config requires 'weights_path' (or upstream-compatible alias "
                "'polytope_weight_path') pointing to a trained upstream .pth artifact. "
                "The artifact cannot be generated by this runtime adapter."
            )
        weights = Path(os.path.expandvars(os.path.expanduser(str(weights_path)))).resolve()
        if not weights.is_file():
            raise FileNotFoundError(
                f"SafetyPolytope weights not found: {weights}. Set weights_path to the trained artifact."
            )

        try:
            SafeRepModel = import_module(
                "safety_polytope.polytope.safe_rep_model"
            ).SafeRepModel
        except Exception as exc:
            raise ImportError(
                "Could not import the official Safety Polytope SafeRepModel. Install the upstream repository "
                "in this environment and verify its Transformers compatibility. AdversariaLLM pins Transformers "
                "5.9.0 while the upstream SafetyPolytope setup.py pins 4.45.2."
            ) from exc

        model_kwargs = dict(cfg.get("model_kwargs", {}))
        model_kwargs["torch_dtype"] = _resolve_torch_dtype(
            model_kwargs.get("torch_dtype"),
            fallback=getattr(model, "dtype", torch.float16),
        )
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = cfg.get("device_map", "auto")

        _warn_on_transformers_major_mismatch()

        try:
            safe_model = SafeRepModel.from_pretrained(
                pretrained_model_name_or_path=str(model_id),
                polytope_weight_path=str(weights),
                steer_layer=cfg.get("steer_layer", 20),
                steer_first_n_tokens=cfg.get("steer_first_n_tokens", 20),
                use_backup_response=cfg.get("use_backup_response", False),
                lambda_weight=cfg.get("lambda_weight", 2.0),
                safe_violation_weight=cfg.get("safe_violation_weight", 0.0001),
                projection=cfg.get("projection", False),
                backup_response=cfg.get("backup_response", "Sorry, I can't help with that request."),
                **model_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "SafetyPolytope SafeRepModel failed to load. Most likely causes are a Transformers API/model-architecture "
                "mismatch, an incompatible checkpoint, or an incompatible trained weights artifact. Do not proceed to "
                "attack evaluation until this load succeeds. Original error: " + repr(exc)
            ) from exc
        if callable(getattr(safe_model, "eval", None)):
            safe_model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            safe_model=safe_model,
            model_id=str(model_id),
            weights_path=str(weights),
            default_generate_kwargs=default_generate_kwargs,
            max_input_tokens=cfg.get("max_input_tokens"),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            steer_layer=cfg.get("steer_layer", 20),
            steer_first_n_tokens=cfg.get("steer_first_n_tokens", 20),
            lambda_weight=cfg.get("lambda_weight", 2.0),
            safe_violation_weight=cfg.get("safe_violation_weight", 0.0001),
            projection=cfg.get("projection", False),
            use_backup_response=cfg.get("use_backup_response", False),
            backup_response=cfg.get("backup_response", "Sorry, I can't help with that request."),
        )

    @torch.enable_grad()
    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        filters: list[dict] | None = None,
        retry_overrides: RetryOverrides | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        # The official SaP steering path optimizes hidden states and requires
        # gradients even during autoregressive generation. Do not decorate this
        # method with torch.no_grad().
        if not convs:
            return GenerationResult(gen=[], input_ids=[], raw_gen=[], defense_decisions=[])
        if filters:
            raise NotImplementedError(
                "SafetyPolytope adapter does not yet translate AdversariaLLM generation filters into the upstream "
                "SafeRepModel API. Disable filters for this defense rather than silently ignoring them."
            )
        if retry_overrides is not None:
            _LOG.debug("SafetyPolytope adapter currently performs one generation attempt; retry_overrides are not applied.")

        prompts = [_render_conversation(self.safe_tokenizer, conv) for conv in convs]
        tokenized = self.safe_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_input_tokens is not None,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        # With device_map='auto', inputs must start on the embedding device.
        try:
            input_device = self.safe_model.model.embed_tokens.weight.device
        except Exception:
            input_device = next(self.safe_model.parameters()).device
        input_ids = torch.as_tensor(tokenized["input_ids"]).to(input_device)
        attention_mask = torch.as_tensor(tokenized["attention_mask"]).to(input_device)
        input_width = input_ids.shape[1]

        generation_kwargs: dict[str, Any] = dict(self.default_generate_kwargs)
        generation_kwargs.update(kwargs)
        generation_kwargs["max_new_tokens"] = int(max_new_tokens or generation_kwargs.get("max_new_tokens", self.max_new_tokens))
        if temperature is not None:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["do_sample"] = float(temperature) > 0
        if top_p is not None:
            generation_kwargs["top_p"] = float(top_p)
        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = int(num_return_sequences)
        generation_kwargs.setdefault("pad_token_id", self.safe_tokenizer.pad_token_id or self.safe_tokenizer.eos_token_id)
        generation_kwargs.setdefault("eos_token_id", self.safe_tokenizer.eos_token_id)

        # SaP's custom forward includes hidden-state optimization; gradients must
        # remain enabled. The output is expected to be [batch * returns, seq].
        generated_ids = self.safe_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        if hasattr(generated_ids, "sequences"):
            generated_ids = generated_ids.sequences
        if generated_ids.ndim != 2:
            raise RuntimeError(
                f"SafetyPolytope generate() returned tensor with unexpected shape {tuple(generated_ids.shape)}. "
                "Expected [batch * num_return_sequences, sequence_length]."
            )
        continuation_ids = generated_ids[:, input_width:]
        decoded = self.safe_tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
        n = int(num_return_sequences or generation_kwargs.get("num_return_sequences", 1))
        if len(decoded) != len(convs) * n:
            raise RuntimeError(
                f"SafetyPolytope returned {len(decoded)} sequences for {len(convs)} prompts and {n} return sequences. "
                "Check generation API compatibility before using evaluation results."
            )
        grouped = [decoded[i * n:(i + 1) * n] for i in range(len(convs))]
        input_id_rows = [row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)]
        metadata = [[{
            "defense": self.NAME,
            "upstream_model": getattr(self.safe_model, "model_name_or_path", self.model_id),
            "weights_path": self.weights_path,
            "steering_enabled": True,
            "note": "Output generated through upstream SafeRepModel; this metadata is not a per-sample safety verdict.",
        } for _ in row] for row in grouped]
        return GenerationResult(
            gen=grouped,
            input_ids=input_id_rows,
            raw_gen=[list(row) for row in grouped],
            defense_decisions=metadata,
        )
