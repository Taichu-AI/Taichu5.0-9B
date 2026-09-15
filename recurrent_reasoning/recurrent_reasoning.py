# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropy-triggered recurrent reasoning for ZDTaichu-5.0.

The host calls ``_init_recurrent_reasoning`` in its constructor and
``_forward_with_recurrent_reasoning`` after preparing multimodal embeddings and
position IDs. The latter returns (normal_model_outputs, selected_logits).
Decode reasoning always uses the historical KV/recurrent cache.
Settings are read from RECURRENT_REASONING_* environment variables at import time.
"""

import copy
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask


def _read_recurrent_reasoning_layer_indices() -> Tuple[int, ...]:
    raw_value = os.getenv(
        "RECURRENT_REASONING_LAYER_INDICES", "12,13,14,15"
    )
    try:
        layer_indices = tuple(
            int(value.strip())
            for value in raw_value.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "RECURRENT_REASONING_LAYER_INDICES must be a comma-separated "
            f"list of integers, got {raw_value!r}"
        ) from exc

    if not layer_indices:
        raise ValueError(
            "RECURRENT_REASONING_LAYER_INDICES must contain at least one "
            "layer index"
        )
    return layer_indices


# Recurrent-reasoning settings. Environment variables are read at import time;
# layer indices are zero-based.
RECURRENT_REASONING_LAYER_INDICES = (
    _read_recurrent_reasoning_layer_indices()
)
# Number of extra middle-block applications after the original N=1 pass.
# Zero disables recurrent reasoning; MAX_ITERS=1 evaluates up to N=2,
# MAX_ITERS=2 evaluates up to N=3, etc.
RECURRENT_REASONING_MAX_ITERS = int(
    os.getenv("RECURRENT_REASONING_MAX_ITERS", "1")
)
RECURRENT_REASONING_THRESHOLD = float(
    os.getenv("RECURRENT_REASONING_THRESHOLD", "1.0")
)
# Stop when KL(current || previous) falls below this threshold; zero disables it.
RECURRENT_REASONING_KL_THRESHOLD = float(
    os.getenv("RECURRENT_REASONING_KL_THRESHOLD", "1e-4")
)
RECURRENT_REASONING_VERBOSE = os.getenv(
    "RECURRENT_REASONING_VERBOSE", "0"
).strip().lower() in {"1", "true", "on"}

if RECURRENT_REASONING_MAX_ITERS < 0:
    raise ValueError("RECURRENT_REASONING_MAX_ITERS must be non-negative")
if RECURRENT_REASONING_THRESHOLD < 0:
    raise ValueError("RECURRENT_REASONING_THRESHOLD must be non-negative")
if not 0.0 <= RECURRENT_REASONING_KL_THRESHOLD < float("inf"):
    raise ValueError(
        "RECURRENT_REASONING_KL_THRESHOLD must be finite and non-negative"
    )


def _is_global_rank_zero() -> bool:
    """Return whether this process is global rank 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # During launcher startup the process group may not be initialized yet,
    # but torchrun/accelerate already exports the global RANK environment value.
    rank = os.getenv("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return rank == "0"


def _reasoning_print(*args, **kwargs) -> None:
    """Print reasoning diagnostics only when enabled and on global rank 0."""
    if RECURRENT_REASONING_VERBOSE and _is_global_rank_zero():
        print(*args, **kwargs)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy in FP32 for numerical stability."""
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _kl_from_logits(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(current || reference) over the vocabulary in FP32."""
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    reference_log_probs = torch.nn.functional.log_softmax(
        reference_logits.float(), dim=-1
    )
    # Roundoff can make an otherwise zero KL slightly negative.
    return (
        log_probs.exp() * (log_probs - reference_log_probs)
    ).sum(dim=-1).clamp_min(0.0)


class RecurrentReasoningMixin:
    """Add recurrent inference to a host exposing ``self.language_model``.

    This adapter uses Qwen3.5 decoder layers, M-RoPE and hybrid KV/DeltaNet cache
    fields. It adds no model parameters and leaves the outer model responsible
    for vision inputs, position preparation, loss and the public return format.
    """

    def _init_recurrent_reasoning(self) -> None:
        """Initialize the inputs retained for cacheless prefill reasoning."""
        self._recurrent_reasoning_context_embeds = None
        self._recurrent_reasoning_context_attention_mask = None
        self._recurrent_reasoning_context_position_ids = None

    def _use_recurrent_reasoning_cache(self) -> bool:
        """Use a fixed inference cache policy, independent of saved config."""
        return not self.training

    def generate(self, *args, **kwargs):
        """Keep generation's cache preparation aligned with the LLM forward."""
        kwargs["use_cache"] = self._use_recurrent_reasoning_cache()
        return super().generate(*args, **kwargs)

    def _forward_with_recurrent_reasoning(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Tuple[Any, torch.Tensor]:
        """Run the ordinary LLM forward with optional recurrent reasoning.

        Accept the language model's forward arguments. Inference uses cache
        unless the caller disables it; training always disables cache. All
        current input positions' logits are retained.

        Return (ordinary_model_output, selected_logits). The ordinary output
        follows return_dict; the host remains responsible for computing loss
        from the selected logits.
        """
        # Preserve the caller's cache opt-out while disabling it for training.
        use_cache = self._use_recurrent_reasoning_cache() and use_cache is not False
        logits_to_keep = 0
        return_dict = (
            return_dict if return_dict is not None
            else self.language_model.config.use_return_dict
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Record this before the normal forward because an existing cache is
        # mutated in place. A positive past length identifies cached decoding
        # for the recurrent-reasoning comparison below.
        past_length_before_forward = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            and hasattr(past_key_values, "get_seq_length")
            else 0
        )
        is_decode = past_length_before_forward > 0
        reasoning_enabled = RECURRENT_REASONING_MAX_ITERS > 0
        use_cached_decode_reasoning = reasoning_enabled and is_decode
        if use_cached_decode_reasoning and (
            inputs_embeds.shape[1] != 1
            or self.training
            or use_cache is False
        ):
            raise ValueError(
                "Cached recurrent reasoning is inference-only and requires "
                "use_cache=True and exactly one decode token"
            )

        prefill_reasoning_inputs_embeds = None
        prefill_reasoning_attention_mask = None
        prefill_reasoning_position_ids = None
        if not reasoning_enabled:
            # N=0 is the ordinary model forward. Clear any context retained
            # by an earlier enabled run and skip all reasoning bookkeeping.
            self._recurrent_reasoning_context_embeds = None
            self._recurrent_reasoning_context_attention_mask = None
            self._recurrent_reasoning_context_position_ids = None
        elif use_cached_decode_reasoning:
            # Decode reasoning reads history from the model cache, so the
            # raw prefill inputs can be released.
            self._recurrent_reasoning_context_embeds = None
            self._recurrent_reasoning_context_attention_mask = None
            self._recurrent_reasoning_context_position_ids = None
        else:
            # Retain the exact post-vision-injection inputs for optional
            # cacheless prefill reasoning.
            self._store_recurrent_reasoning_prefill_context(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            prefill_reasoning_inputs_embeds = (
                self._recurrent_reasoning_context_embeds
            )
            prefill_reasoning_attention_mask = (
                self._recurrent_reasoning_context_attention_mask
            )
            prefill_reasoning_position_ids = (
                self._recurrent_reasoning_context_position_ids
            )

        cached_linear_state_snapshots = None
        cached_boundary_hidden_states: Dict[str, torch.Tensor] = {}
        reasoning_hook_handles = []
        if use_cached_decode_reasoning:
            (
                text_model,
                num_layers,
                reasoning_start,
                reasoning_end,
            ) = self._get_recurrent_reasoning_layer_span()
            cached_linear_state_snapshots = (
                self._snapshot_recurrent_reasoning_linear_states(
                    past_key_values,
                    range(reasoning_start, num_layers),
                )
            )

            def capture_middle_input(_module, args):
                cached_boundary_hidden_states["middle_input"] = (
                    args[0].detach()
                )

            def capture_middle_output(_module, _args, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(
                        "Expected the reasoning middle block to return a "
                        f"tensor, got {type(output).__name__}"
                    )
                cached_boundary_hidden_states["middle_output"] = (
                    output.detach()
                )

            reasoning_hook_handles = [
                text_model.layers[reasoning_start].register_forward_pre_hook(
                    capture_middle_input
                ),
                text_model.layers[reasoning_end].register_forward_hook(
                    capture_middle_output
                ),
            ]

        try:
            outputs = self.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        finally:
            for hook_handle in reasoning_hook_handles:
                hook_handle.remove()
        logits = outputs.logits

        should_reason = False
        ori_last_token_entropy = None
        if reasoning_enabled:
            with torch.no_grad():
                ori_last_token_entropy = _entropy_from_logits(
                    logits[:, -1, :]
                ).item()
                _reasoning_print(
                    f"ori_last_token_entropy={ori_last_token_entropy}"
                )
                # Batch size is always 1, so entropy is handled as one scalar.
                should_reason = (
                    ori_last_token_entropy > RECURRENT_REASONING_THRESHOLD
                )

        if should_reason:
            current_input_length = inputs_embeds.shape[1]
            if use_cached_decode_reasoning:
                missing_boundaries = {
                    "middle_input",
                    "middle_output",
                } - cached_boundary_hidden_states.keys()
                if missing_boundaries:
                    raise RuntimeError(
                        "Normal decode did not expose reasoning boundary "
                        f"states: missing {sorted(missing_boundaries)}"
                    )
                frozen_history_cache = (
                    self._build_recurrent_reasoning_history_cache(
                        outputs.past_key_values,
                        past_length=past_length_before_forward,
                        linear_state_snapshots=(
                            cached_linear_state_snapshots
                        ),
                    )
                )
                (
                    selected_reasoning_n,
                    selected_entropy,
                    selected_reasoning_logits,
                    candidate_entropies,
                ) = self._run_cached_decode_recurrent_reasoning_candidates(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    frozen_history_cache=frozen_history_cache,
                    base_input_hidden_states=(
                        cached_boundary_hidden_states["middle_input"]
                    ),
                    base_output_hidden_states=(
                        cached_boundary_hidden_states["middle_output"]
                    ),
                    baseline_entropy=ori_last_token_entropy,
                    baseline_last_token_logits=logits[:, -1, :],
                    logits_to_keep=logits_to_keep,
                )
            else:
                # Prefill reasoning recomputes the current input tokens
                # without a KV/recurrent cache.
                (
                    selected_reasoning_n,
                    selected_entropy,
                    selected_reasoning_logits,
                    candidate_entropies,
                ) = self._run_recurrent_reasoning_candidates(
                    inputs_embeds=prefill_reasoning_inputs_embeds,
                    attention_mask=prefill_reasoning_attention_mask,
                    position_ids=prefill_reasoning_position_ids,
                    cache_position=None,
                    current_input_length=current_input_length,
                    baseline_entropy=ori_last_token_entropy,
                    baseline_last_token_logits=logits[:, -1, :],
                    logits_to_keep=logits_to_keep,
                )

            entropy_summary = [
                f"ori={ori_last_token_entropy}"
            ] + [
                f"N={iteration}={candidate_entropy}"
                for iteration, candidate_entropy in candidate_entropies.items()
            ]
            _reasoning_print(
                "reasoning comparison: " + ", ".join(entropy_summary)
            )

            # Only logits affect standard generation. Keep any optionally
            # requested hidden states from the original model path unchanged.
            if selected_reasoning_n is not None:
                logits = selected_reasoning_logits
                _reasoning_print(
                    "accepted reasoning logits from "
                    f"N={selected_reasoning_n}, "
                    f"entropy={selected_entropy}"
                )
            else:
                _reasoning_print(
                    "kept original logits because N=2 increased the entropy"
                )

        return (outputs if return_dict else outputs.to_tuple()), logits

    def _store_recurrent_reasoning_prefill_context(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> None:
        """Save the exact full inputs needed for cacheless reasoning."""
        batch_size, sequence_length = inputs_embeds.shape[:2]

        if attention_mask is None:
            context_attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        else:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Recurrent reasoning expects a 2-D attention_mask, got "
                    f"shape {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape[-1] < sequence_length:
                raise ValueError(
                    "Prefill attention_mask is shorter than inputs_embeds: "
                    f"{attention_mask.shape[-1]} < {sequence_length}"
                )
            context_attention_mask = attention_mask[:, -sequence_length:]

        if position_ids is None:
            if cache_position is not None:
                if cache_position.numel() < sequence_length:
                    raise ValueError(
                        "Prefill cache_position is shorter than "
                        f"inputs_embeds: {cache_position.numel()} < "
                        f"{sequence_length}"
                    )
                context_positions = cache_position.reshape(-1)[
                    -sequence_length:
                ].to(inputs_embeds.device)
            else:
                context_positions = torch.arange(
                    sequence_length,
                    device=inputs_embeds.device,
                )
            # Match Qwen3.5's position_ids=None behavior: one causal-mask
            # channel followed by three identical rotary channels.
            context_position_ids = context_positions.view(
                1, 1, -1
            ).expand(4, batch_size, -1)
        else:
            if position_ids.ndim not in (2, 3):
                raise ValueError(
                    "Recurrent reasoning expects 2-D or 3-D position_ids, "
                    f"got shape {tuple(position_ids.shape)}"
                )
            if position_ids.shape[-1] < sequence_length:
                raise ValueError(
                    "Prefill position_ids is shorter than inputs_embeds: "
                    f"{position_ids.shape[-1]} < {sequence_length}"
                )
            context_position_ids = position_ids[..., -sequence_length:]
            if context_position_ids.ndim == 2:
                context_position_ids = context_position_ids[
                    None, ...
                ].expand(4, -1, -1)
            elif context_position_ids.shape[0] == 1:
                # A single channel is a rotary channel in Qwen3.5. Expand it
                # to the three M-RoPE channels while keeping mask positions
                # implicit, matching the original model path.
                context_position_ids = context_position_ids.expand(
                    3, -1, -1
                )
            elif context_position_ids.shape[0] not in (3, 4):
                raise ValueError(
                    "Recurrent reasoning expects 1, 3, or 4 position "
                    f"channels, got {context_position_ids.shape[0]}"
                )

        self._recurrent_reasoning_context_embeds = inputs_embeds.detach()
        self._recurrent_reasoning_context_attention_mask = (
            context_attention_mask.detach()
        )
        self._recurrent_reasoning_context_position_ids = (
            context_position_ids.detach()
        )

    def _get_recurrent_reasoning_layer_span(
        self,
    ) -> Tuple[nn.Module, int, int, int]:
        """Validate and return the contiguous middle-block layer span."""
        text_model = self.language_model.model
        num_layers = len(text_model.layers)
        invalid_layer_indices = [
            layer_idx
            for layer_idx in RECURRENT_REASONING_LAYER_INDICES
            if layer_idx < 0 or layer_idx >= num_layers
        ]
        if invalid_layer_indices:
            raise ValueError(
                "Invalid recurrent-reasoning layer indices "
                f"{invalid_layer_indices}; the language model has "
                f"{num_layers} layers"
            )

        reasoning_start = RECURRENT_REASONING_LAYER_INDICES[0]
        reasoning_end = RECURRENT_REASONING_LAYER_INDICES[-1]
        expected_layer_indices = tuple(
            range(reasoning_start, reasoning_end + 1)
        )
        if RECURRENT_REASONING_LAYER_INDICES != expected_layer_indices:
            raise ValueError(
                "RECURRENT_REASONING_LAYER_INDICES must be a contiguous "
                "ascending block, got "
                f"{RECURRENT_REASONING_LAYER_INDICES}"
            )
        return text_model, num_layers, reasoning_start, reasoning_end

    @staticmethod
    def _snapshot_recurrent_reasoning_linear_states(
        cache,
        layer_indices,
    ) -> Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Clone pre-forward DeltaNet states that normal decode mutates."""
        required_attributes = (
            "key_cache",
            "value_cache",
            "conv_states",
            "recurrent_states",
        )
        missing_attributes = [
            name for name in required_attributes if not hasattr(cache, name)
        ]
        if missing_attributes:
            raise TypeError(
                "Cached decode reasoning requires Qwen3.5's dynamic cache; "
                f"missing attributes {missing_attributes} on "
                f"{type(cache).__name__}"
            )

        snapshots = {}
        for layer_idx in layer_indices:
            conv_state = cache.conv_states[layer_idx]
            recurrent_state = cache.recurrent_states[layer_idx]
            snapshots[layer_idx] = (
                None if conv_state is None else conv_state.detach().clone(),
                None
                if recurrent_state is None
                else recurrent_state.detach().clone(),
            )
        return snapshots

    @staticmethod
    def _build_recurrent_reasoning_history_cache(
        updated_cache,
        past_length: int,
        linear_state_snapshots: Dict[
            int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        ],
    ):
        """Build an isolated cache view containing history before this token."""
        frozen_cache = copy.copy(updated_cache)
        frozen_cache.key_cache = list(updated_cache.key_cache)
        frozen_cache.value_cache = list(updated_cache.value_cache)
        frozen_cache.conv_states = list(updated_cache.conv_states)
        frozen_cache.recurrent_states = list(updated_cache.recurrent_states)

        # Qwen3.5's dynamic full-attention cache appends with torch.cat. Slice
        # the just-added token from the updated normal cache without copying the
        # potentially long KV history.
        for layer_idx, key_state in enumerate(frozen_cache.key_cache):
            if key_state is None or key_state.numel() == 0:
                continue
            value_state = frozen_cache.value_cache[layer_idx]
            if value_state is None or key_state.ndim < 3:
                raise RuntimeError(
                    "Malformed Qwen3.5 KV cache at full-attention layer "
                    f"{layer_idx}"
                )
            if key_state.shape[2] < past_length:
                raise RuntimeError(
                    "Updated KV cache is shorter than its pre-forward length: "
                    f"layer {layer_idx}, {key_state.shape[2]} < {past_length}"
                )
            frozen_cache.key_cache[layer_idx] = key_state.narrow(
                2, 0, past_length
            )
            frozen_cache.value_cache[layer_idx] = value_state.narrow(
                2, 0, past_length
            )

        for layer_idx, (conv_state, recurrent_state) in (
            linear_state_snapshots.items()
        ):
            frozen_cache.conv_states[layer_idx] = conv_state
            frozen_cache.recurrent_states[layer_idx] = recurrent_state
        return frozen_cache

    @staticmethod
    def _fork_recurrent_reasoning_cache(frozen_cache, layer_indices):
        """Fork scratch state while sharing immutable historical KV tensors."""
        working_cache = copy.copy(frozen_cache)
        working_cache.key_cache = list(frozen_cache.key_cache)
        working_cache.value_cache = list(frozen_cache.value_cache)
        working_cache.conv_states = list(frozen_cache.conv_states)
        working_cache.recurrent_states = list(frozen_cache.recurrent_states)
        for layer_idx in layer_indices:
            conv_state = frozen_cache.conv_states[layer_idx]
            recurrent_state = frozen_cache.recurrent_states[layer_idx]
            # causal_conv1d_update mutates its state tensor in place.
            working_cache.conv_states[layer_idx] = (
                None if conv_state is None else conv_state.clone()
            )
            working_cache.recurrent_states[layer_idx] = (
                None if recurrent_state is None else recurrent_state.clone()
            )
        return working_cache

    def _evaluate_recurrent_reasoning_trajectory(
        self,
        base_input_hidden_states: torch.Tensor,
        base_output_hidden_states: torch.Tensor,
        run_middle_block: Callable[[torch.Tensor], torch.Tensor],
        run_suffix: Callable[[torch.Tensor], torch.Tensor],
        current_input_length: int,
        baseline_entropy: float,
        baseline_last_token_logits: torch.Tensor,
        logits_to_keep: int,
    ) -> Tuple[
        Optional[int],
        float,
        Optional[torch.Tensor],
        Dict[int, float],
    ]:
        """Evaluate N=2... until entropy rises or accepted logits converge."""
        text_model = self.language_model.model
        selected_n = None
        previous_entropy = baseline_entropy
        previous_last_token_logits = baseline_last_token_logits.detach()
        selected_logits = None
        candidate_entropies: Dict[int, float] = {}

        # MAX_ITERS counts extra middle-block applications beyond N=1. Use one
        # fixed damping weight for the complete candidate trajectory.
        max_candidate_n = RECURRENT_REASONING_MAX_ITERS + 1
        damping_weight = 1.0 / max_candidate_n
        base_input_hidden_states = base_input_hidden_states.to(
            base_output_hidden_states.device
        )
        reasoning_hidden_states = (
            (1.0 - damping_weight) * base_input_hidden_states
            + damping_weight * base_output_hidden_states
        )

        for reasoning_iter in range(1, RECURRENT_REASONING_MAX_ITERS + 1):
            candidate_n = reasoning_iter + 1
            iteration_input_hidden_states = reasoning_hidden_states
            iteration_output_hidden_states = run_middle_block(
                iteration_input_hidden_states
            )

            # The suffix is a branch for this depth and never feeds the next
            # middle-block application.
            candidate_hidden_states = run_suffix(
                iteration_output_hidden_states
            )
            candidate_current_hidden_states = candidate_hidden_states[
                :, -current_input_length:, :
            ]
            candidate_logits_hidden_states = (
                candidate_current_hidden_states
                if logits_to_keep == 0
                else candidate_current_hidden_states[
                    :, -logits_to_keep:, :
                ]
            )
            candidate_logits_hidden_states = text_model.norm(
                candidate_logits_hidden_states
            )
            candidate_logits = self.language_model.lm_head(
                candidate_logits_hidden_states
            )

            with torch.no_grad():
                candidate_entropy = _entropy_from_logits(
                    candidate_logits[:, -1, :]
                ).item()
                candidate_kl = (
                    _kl_from_logits(
                        candidate_logits[:, -1, :],
                        previous_last_token_logits,
                    ).item()
                    if RECURRENT_REASONING_KL_THRESHOLD > 0
                    else None
                )
            candidate_entropies[candidate_n] = candidate_entropy

            if candidate_entropy > previous_entropy:
                previous_name = (
                    "ori" if selected_n is None else f"N={selected_n}"
                )
                _reasoning_print(
                    "stopped reasoning at "
                    f"N={candidate_n}: entropy={candidate_entropy} "
                    f"> {previous_name}_entropy={previous_entropy}"
                )
                break

            selected_n = candidate_n
            previous_entropy = candidate_entropy
            selected_logits = candidate_logits
            previous_last_token_logits = candidate_logits[:, -1, :].detach()

            # Keep the accepted candidate, then skip further block applications
            # when its output distribution barely changes from the prior one.
            if (
                candidate_kl is not None
                and candidate_kl < RECURRENT_REASONING_KL_THRESHOLD
            ):
                _reasoning_print(
                    "stopped reasoning at "
                    f"N={candidate_n}: KL(current || previous)={candidate_kl} "
                    f"< {RECURRENT_REASONING_KL_THRESHOLD}; "
                    "accepted current logits"
                )
                break

            if reasoning_iter < RECURRENT_REASONING_MAX_ITERS:
                reasoning_hidden_states = (
                    (1.0 - damping_weight) * iteration_input_hidden_states
                    + damping_weight * iteration_output_hidden_states
                )
            _reasoning_print(
                f"completed reasoning iteration {reasoning_iter}/"
                f"{RECURRENT_REASONING_MAX_ITERS} (N={candidate_n})"
            )

        return (
            selected_n,
            previous_entropy,
            selected_logits,
            candidate_entropies,
        )

    def _run_recurrent_reasoning_candidates(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
        current_input_length: int,
        baseline_entropy: float,
        baseline_last_token_logits: torch.Tensor,
        logits_to_keep: int,
    ) -> Tuple[
        Optional[int],
        float,
        Optional[torch.Tensor],
        Dict[int, float],
    ]:
        """Increase reasoning depth until entropy rises or logits converge.

        Layers before ``RECURRENT_REASONING_LAYER_INDICES`` run once. The
        configured middle block then runs once for the original N=1 depth plus
        at most ``RECURRENT_REASONING_MAX_ITERS`` additional times. Consequently,
        reasoning iteration 1 evaluates N=2, iteration 2 evaluates N=3, and so
        on. N=2 is compared with the original entropy; every later N is compared
        with the previously accepted candidate. The previous result is retained
        as soon as entropy strictly increases. Otherwise, the current result is
        accepted and reasoning also stops if KL(current || previous) is below
        RECURRENT_REASONING_KL_THRESHOLD (when positive).
        """

        (
            text_model,
            num_layers,
            reasoning_start,
            reasoning_end,
        ) = self._get_recurrent_reasoning_layer_span()

        batch_size, sequence_length = inputs_embeds.shape[:2]
        if current_input_length <= 0 or current_input_length > sequence_length:
            raise ValueError(
                "current_input_length must be in [1, sequence_length], got "
                f"{current_input_length} for sequence length {sequence_length}"
            )
        reasoning_cache_position = torch.arange(
            sequence_length, device=inputs_embeds.device
        )

        reasoning_attention_mask = attention_mask
        if reasoning_attention_mask is not None:
            if reasoning_attention_mask.ndim != 2:
                raise ValueError(
                    "Recurrent reasoning expects a 2-D attention_mask, got "
                    f"shape {tuple(reasoning_attention_mask.shape)}"
                )
            if reasoning_attention_mask.shape[-1] < sequence_length:
                raise ValueError(
                    "attention_mask is shorter than the reasoning hidden "
                    f"states: {reasoning_attention_mask.shape[-1]} < "
                    f"{sequence_length}"
                )
            # Align the mask with the current prefill input tokens.
            reasoning_attention_mask = reasoning_attention_mask[
                :, -sequence_length:
            ]

        reasoning_position_ids = position_ids
        if reasoning_position_ids is None:
            rotary_positions = reasoning_cache_position
            if cache_position is not None:
                if cache_position.numel() < sequence_length:
                    raise ValueError(
                        "cache_position is shorter than the reasoning "
                        f"hidden states: {cache_position.numel()} < "
                        f"{sequence_length}"
                    )
                rotary_positions = cache_position.reshape(-1)[
                    -sequence_length:
                ].to(inputs_embeds.device)
            # Only rotary channels are needed here. The causal mask uses the
            # local, cacheless positions created above.
            reasoning_position_ids = rotary_positions.view(
                1, 1, -1
            ).expand(3, batch_size, -1)
        else:
            if reasoning_position_ids.ndim not in (2, 3):
                raise ValueError(
                    "Recurrent reasoning expects 2-D or 3-D position_ids, "
                    f"got shape {tuple(reasoning_position_ids.shape)}"
                )
            if reasoning_position_ids.shape[-1] < sequence_length:
                raise ValueError(
                    "position_ids is shorter than the reasoning hidden "
                    f"states: {reasoning_position_ids.shape[-1]} < "
                    f"{sequence_length}"
                )
            reasoning_position_ids = reasoning_position_ids[
                ..., -sequence_length:
            ]
            if reasoning_position_ids.ndim == 2:
                reasoning_position_ids = reasoning_position_ids[
                    None, ...
                ].expand(4, -1, -1)

        # Qwen3.5 uses four M-RoPE channels on prefill: the first builds the
        # causal mask and the remaining three build rotary embeddings.
        if (
            reasoning_position_ids.ndim == 3
            and reasoning_position_ids.shape[0] == 4
        ):
            text_position_ids = reasoning_position_ids[0]
            rotary_position_ids = reasoning_position_ids[1:]
        else:
            text_position_ids = None
            rotary_position_ids = reasoning_position_ids

        causal_mask = create_causal_mask(
            config=text_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=reasoning_attention_mask,
            cache_position=reasoning_cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attention_mask = reasoning_attention_mask
        if (
            linear_attention_mask is not None
            and torch.all(linear_attention_mask == 1)
        ):
            linear_attention_mask = None

        position_embeddings = text_model.rotary_emb(
            inputs_embeds, rotary_position_ids
        )

        def run_decoder_layers(
            current_hidden_states: torch.Tensor,
            layer_indices,
        ) -> torch.Tensor:
            for layer_idx in layer_indices:
                decoder_layer = text_model.layers[layer_idx]
                layer_attention_mask = (
                    linear_attention_mask
                    if decoder_layer.layer_type == "linear_attention"
                    else causal_mask
                )
                current_hidden_states = decoder_layer(
                    current_hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    position_ids=rotary_position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=reasoning_cache_position,
                )
            return current_hidden_states

        # Run the prefix and the original N=1 middle block once.
        base_input_hidden_states = run_decoder_layers(
            inputs_embeds,
            range(reasoning_start),
        )
        base_output_hidden_states = run_decoder_layers(
            base_input_hidden_states,
            RECURRENT_REASONING_LAYER_INDICES,
        )
        return self._evaluate_recurrent_reasoning_trajectory(
            base_input_hidden_states=base_input_hidden_states,
            base_output_hidden_states=base_output_hidden_states,
            run_middle_block=lambda hidden_states: run_decoder_layers(
                hidden_states,
                RECURRENT_REASONING_LAYER_INDICES,
            ),
            run_suffix=lambda hidden_states: run_decoder_layers(
                hidden_states,
                range(reasoning_end + 1, num_layers),
            ),
            current_input_length=current_input_length,
            baseline_entropy=baseline_entropy,
            baseline_last_token_logits=baseline_last_token_logits,
            logits_to_keep=logits_to_keep,
        )

    def _run_cached_decode_recurrent_reasoning_candidates(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
        frozen_history_cache,
        base_input_hidden_states: torch.Tensor,
        base_output_hidden_states: torch.Tensor,
        baseline_entropy: float,
        baseline_last_token_logits: torch.Tensor,
        logits_to_keep: int,
    ) -> Tuple[
        Optional[int],
        float,
        Optional[torch.Tensor],
        Dict[int, float],
    ]:
        """Run reasoning for one decode token against immutable cached history.

        Historical token states stay fixed at their normal-model values. Every
        middle-block repetition and suffix branch receives a scratch cache fork,
        so candidate evaluation cannot update the normal generation cache or
        feed the current token into itself as cached history.
        """
        (
            text_model,
            num_layers,
            _,
            reasoning_end,
        ) = self._get_recurrent_reasoning_layer_span()
        batch_size, current_length = inputs_embeds.shape[:2]
        if current_length != 1:
            raise ValueError(
                "Cached decode reasoning supports exactly one current token, "
                f"got sequence length {current_length}"
            )

        past_length = frozen_history_cache.get_seq_length()
        if cache_position is None:
            reasoning_cache_position = torch.arange(
                past_length,
                past_length + current_length,
                device=inputs_embeds.device,
            )
        else:
            if cache_position.numel() < current_length:
                raise ValueError(
                    "cache_position is shorter than the current decode input: "
                    f"{cache_position.numel()} < {current_length}"
                )
            reasoning_cache_position = cache_position.reshape(-1)[
                -current_length:
            ].to(inputs_embeds.device)

        reasoning_position_ids = position_ids
        if reasoning_position_ids is None:
            reasoning_position_ids = reasoning_cache_position.view(
                1, 1, -1
            ).expand(4, batch_size, -1)
        else:
            if reasoning_position_ids.ndim not in (2, 3):
                raise ValueError(
                    "Recurrent reasoning expects 2-D or 3-D position_ids, "
                    f"got shape {tuple(reasoning_position_ids.shape)}"
                )
            if reasoning_position_ids.shape[-1] < current_length:
                raise ValueError(
                    "position_ids is shorter than the current decode input: "
                    f"{reasoning_position_ids.shape[-1]} < {current_length}"
                )
            reasoning_position_ids = reasoning_position_ids[
                ..., -current_length:
            ].to(inputs_embeds.device)
            if reasoning_position_ids.ndim == 2:
                reasoning_position_ids = reasoning_position_ids[
                    None, ...
                ].expand(4, -1, -1)

        if reasoning_position_ids.shape[0] == 4:
            text_position_ids = reasoning_position_ids[0]
            rotary_position_ids = reasoning_position_ids[1:]
        else:
            text_position_ids = None
            rotary_position_ids = reasoning_position_ids

        causal_mask = create_causal_mask(
            config=text_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=reasoning_cache_position,
            past_key_values=frozen_history_cache,
            position_ids=text_position_ids,
        )
        linear_attention_mask = text_model._update_linear_attn_mask(
            attention_mask,
            reasoning_cache_position,
        )
        position_embeddings = text_model.rotary_emb(
            inputs_embeds,
            rotary_position_ids,
        )

        def run_decoder_layers(
            current_hidden_states: torch.Tensor,
            layer_indices,
        ) -> torch.Tensor:
            layer_indices = tuple(layer_indices)
            working_cache = self._fork_recurrent_reasoning_cache(
                frozen_history_cache,
                layer_indices,
            )
            for layer_idx in layer_indices:
                decoder_layer = text_model.layers[layer_idx]
                layer_attention_mask = (
                    linear_attention_mask
                    if decoder_layer.layer_type == "linear_attention"
                    else causal_mask
                )
                current_hidden_states = decoder_layer(
                    current_hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    position_ids=rotary_position_ids,
                    past_key_values=working_cache,
                    use_cache=True,
                    cache_position=reasoning_cache_position,
                )
            return current_hidden_states

        return self._evaluate_recurrent_reasoning_trajectory(
            base_input_hidden_states=base_input_hidden_states,
            base_output_hidden_states=base_output_hidden_states,
            run_middle_block=lambda hidden_states: run_decoder_layers(
                hidden_states,
                RECURRENT_REASONING_LAYER_INDICES,
            ),
            run_suffix=lambda hidden_states: run_decoder_layers(
                hidden_states,
                range(reasoning_end + 1, num_layers),
            ),
            current_input_length=current_length,
            baseline_entropy=baseline_entropy,
            baseline_last_token_logits=baseline_last_token_logits,
            logits_to_keep=logits_to_keep,
        )
