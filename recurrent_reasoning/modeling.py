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

# ============================================================================
#  ZDTaichu-5.0  —  Main Model
#
#  Architecture
#  ────────────
#    Vision encoder : C-RADIOv4-H (ViT-H/16, 653 M)
#    Projector      : RMSNorm → Linear(5120→20480) → SquaredReLU → Linear(20480→H)
#    LLM decoder    : Qwen3.5 hybrid (Gated DeltaNet + full attention, 3:1 ratio)
#
#  Position encoding (M-RoPE)
#  ──────────────────────────
#  Vision tokens receive 3D position IDs (temporal, height, width) computed
#  from the InternVL-style tile grid via ``get_rope_index()``.  Text tokens
#  receive standard 1D positions (all three M-RoPE channels are identical).
#
#  This matches the official Qwen3.5 VL pipeline where ``Qwen3_5Model.forward()``
#  calls ``compute_3d_position_ids()`` → ``get_rope_index()`` before forwarding
#  to ``Qwen3_5TextModel``.  The resulting ``position_ids`` of shape ``(3, B, S)``
#  are consumed directly by ``Qwen3_5TextRotaryEmbedding``, which applies
#  interleaved M-RoPE across temporal / height / width frequency bands.
#
#  Generation
#  ──────────
#  This model inherits from ``GenerationMixin``, owning the generation loop
#  (like ``Qwen3_5ForConditionalGeneration``).  Key overrides:
#    - ``_prepare_position_ids_for_generation``: computes 3D ``position_ids``
#      on the prefill step and caches ``rope_deltas``; applies ``rope_deltas``
#      on subsequent decode steps.
#    - ``prepare_inputs_for_generation``: clears ``pixel_values`` /
#      ``pixel_values_videos`` after the first step (vision features are
#      already embedded in the KV cache).
#
#  Cache handling
#  ──────────────
#  ``Qwen3_5DynamicCache`` is created internally by ``Qwen3_5TextModel`` when
#  ``use_cache=True``.  It stores KV states for full-attention layers and
#  ``conv_states`` + ``recurrent_states`` for Gated DeltaNet layers.
# ============================================================================

import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration import ZDTaichu5_0_Config
from .cradio_model import RADIOModel
from .recurrent_reasoning import RecurrentReasoningMixin

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Import Qwen3.5 model classes — requires transformers >= 5.3.0
# ---------------------------------------------------------------------------

_MIN_TRANSFORMERS = "5.3.0"

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from transformers.cache_utils import DynamicCache as Qwen3_5DynamicCache
    _HAS_QWEN3_5 = True
except Exception as e:
    _HAS_QWEN3_5 = False
    Qwen3_5ForCausalLM = None
    Qwen3_5DynamicCache = None
    logger.warning(
        f"Could not import Qwen3_5ForCausalLM from transformers. "
        f"Import error: {e!r}"
    )


def _version_ge(v1, v2):
    """Check if version v1 >= v2."""
    from packaging import version
    return version.parse(v1) >= version.parse(v2)


# ─────────────────────────────────────────────────────────────────────────────
# Projector components
# ─────────────────────────────────────────────────────────────────────────────

class SquaredReLU(nn.Module):
    """Squared ReLU activation — same non-linearity used in the projector."""
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    """
    Standard RMSNorm for the projector (NOT the Qwen3.5 LLM variant).

    Qwen3.5's internal ``Qwen3_5RMSNorm`` uses zero-initialized weight with
    ``output * (1 + weight)``.  The projector uses ones-initialized weight
    with ``output * weight`` — the standard formulation.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class ZDTaichu5_0_ForConditionalGeneration(RecurrentReasoningMixin, PreTrainedModel, GenerationMixin):
    """
    ZDTaichu-5.0: C-RADIOv4-H vision encoder + Qwen3.5 language decoder.

    Architecture overview::

        pixel_values
          └─► C-RADIOv4-H (ViT-H/16, 653 M)
                └─► pixel_shuffle(0.5)
                      └─► mlp1: RMSNorm → Linear → SquaredReLU → Linear
                            └─► inject into Qwen3.5 embeddings at <image> positions
                                  └─► Qwen3.5 (hybrid DeltaNet / Transformer)
    """

    config_class = ZDTaichu5_0_Config
    main_input_name = "input_ids"
    _tied_weights_keys = None#["language_model.lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [
        # The RADIO input_conditioner registers norm_mean / norm_std as
        # buffers, but make_preprocessor_external() removes the conditioner
        # at init time (normalization is handled by the image processor).
        # The build script still saves these from the source checkpoint, so
        # they appear as unexpected keys during loading — safe to ignore.
        r"vision_model\.radio_model\.input_conditioner\..*",
        r"^mtp\..*",
    ]

    _supports_flash_attn_2 = True
    _supports_flash_attention_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _is_stateful = True
    supports_gradient_checkpointing = True

    def __init__(self, config: ZDTaichu5_0_Config):
        super().__init__(config)

        # Guard for bleeding-edge transformers (>= 4.57.0.dev) where
        # _finalize_model_loading reads all_tied_weights_keys but
        # PreTrainedModel.__init__ may not yet initialise it.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

        assert _version_ge(transformers.__version__, _MIN_TRANSFORMERS), (
            f"Qwen3.5 support requires transformers >= {_MIN_TRANSFORMERS} "
            f"(found {transformers.__version__})"
        )
        assert _HAS_QWEN3_5, (
            "Qwen3_5ForCausalLM is not available. "
            f"Ensure transformers >= {_MIN_TRANSFORMERS} is installed."
        )

        image_size = config.force_image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id

        # Per-tile token dimensions (e.g. 14×14 for 448px, patch=16, ds=0.5)
        self.tile_h = int((image_size // patch_size) * config.downsample_ratio)
        self.tile_w = self.tile_h

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"tile_h={self.tile_h}, tile_w={self.tile_w}")
        logger.info(f"ps_version: {self.ps_version}")
        logger.info(f"Vision encoder: {config.vision_config.version}")
        logger.info(
            f"LLM: Qwen3.5 ({config.llm_config.num_hidden_layers} layers, "
            f"hidden={config.llm_config.hidden_size}, "
            f"hybrid="
            f"{sum(1 for t in config.llm_config.layer_types if t == 'linear_attention')} linear + "
            f"{sum(1 for t in config.llm_config.layer_types if t == 'full_attention')} full)"
        )

        # ── Language model ───────────────────────────────────────────────────
        self.language_model = Qwen3_5ForCausalLM(config.llm_config)

        # ── Vision encoder ───────────────────────────────────────────────────
        self.vision_model = RADIOModel(config.vision_config)
        self.vision_model.model._initialize_weights = (
            self.vision_model.model._init_weights
        )
        self.vision_model.radio_model.make_preprocessor_external()
        self.vision_model = self.vision_model.to(
            self.language_model.config.torch_dtype
        )

        self.drop_vision_class_token = True

        # ── MLP projector ────────────────────────────────────────────────────
        vit_hidden_size = config.vit_hidden_size
        proj_hidden = config.projector_hidden_size
        llm_hidden = config.llm_config.hidden_size
        pixel_shuffle_dim = vit_hidden_size * int(1 / self.downsample_ratio) ** 2

        self.mlp1 = nn.Sequential(
            RMSNorm(pixel_shuffle_dim, eps=1e-5),
            nn.Linear(pixel_shuffle_dim, proj_hidden, bias=False),
            SquaredReLU(),
            nn.Linear(proj_hidden, llm_hidden, bias=False),
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.torch_dtype)

        # Cached rope_deltas for multi-step generation
        self.rope_deltas = None

        self._init_recurrent_reasoning()

    # ── Embedding accessors (required by GenerationMixin) ─────────────────

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # 大头在 LLM:直接委托给内层 Qwen3.5(它原生支持 GC)
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        # 视觉塔可选:支持就开,不支持就跳过(不影响主显存)
        vm = getattr(self, "vision_model", None)
        if vm is not None and getattr(vm, "supports_gradient_checkpointing", False):
            try:
                vm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except Exception:
                pass

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()
        vm = getattr(self, "vision_model", None)
        if vm is not None and hasattr(vm, "gradient_checkpointing_disable"):
            try:
                vm.gradient_checkpointing_disable()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Vision helpers
    # ─────────────────────────────────────────────────────────────────────────

    def pixel_shuffle(
        self, x: torch.Tensor, scale_factor: float = 0.5
    ) -> torch.Tensor:
        """Space-to-depth rearrangement (ps_version='v2' = corrected layout)."""
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "ps_version='v1' produces a transposed spatial layout. "
                "Use ps_version='v2' for correct output."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run pixels through C-RADIOv4-H → pixel_shuffle → MLP projector."""
        vit_embeds = self.vision_model(pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, vit_embeds.shape[-1]
        )
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    # ─────────────────────────────────────────────────────────────────────────
    # 3D M-RoPE position IDs
    # ─────────────────────────────────────────────────────────────────────────

    def get_vision_position_ids(
        self,
        start_position: int,
        tile_rows: int,
        tile_cols: int,
        has_thumbnail: bool = True,
        device: torch.device = None,
    ) -> torch.LongTensor:
        """
        Compute 3D (temporal, height, width) position IDs for vision tokens
        from a single InternVL-style tiled image.

        Token layout (flattened order expected by the model):
          1. Grid tiles in raster order: tile(0,0), tile(0,1), …, tile(R-1,C-1).
             Each tile has ``tile_h × tile_w`` tokens in raster order.
          2. Thumbnail tile (optional): a single tile covering the full image
             at reduced resolution.

        Args:
            start_position: Offset added to all positional indices.
            tile_rows: Number of tile rows in the image grid.
            tile_cols: Number of tile columns in the image grid.
            has_thumbnail: Whether a thumbnail tile is appended after grid tiles.
            device: Target device.

        Returns:
            ``torch.LongTensor`` of shape ``(3, num_vision_tokens)``.
        """
        tile_h, tile_w = self.tile_h, self.tile_w
        npt = tile_h * tile_w  # num tokens per tile

        # ── Grid tiles ───────────────────────────────────────────────────────
        num_grid_tiles = tile_rows * tile_cols
        tile_idx = torch.arange(num_grid_tiles, device=device)
        tr = tile_idx // tile_cols
        tc = tile_idx % tile_cols

        local_idx = torch.arange(npt, device=device)
        lr = local_idx // tile_w
        lc = local_idx % tile_w

        # (num_grid_tiles, npt) → flatten
        global_h = (tr[:, None] * tile_h + lr[None, :]).reshape(-1).long()
        global_w = (tc[:, None] * tile_w + lc[None, :]).reshape(-1).long()

        total_grid = num_grid_tiles * npt
        pos_t = torch.full(
            (total_grid,), start_position, device=device, dtype=torch.long
        )
        pos_h = start_position + global_h
        pos_w = start_position + global_w

        # ── Thumbnail tile ───────────────────────────────────────────────────
        if has_thumbnail:
            # Map thumbnail local(r, c) → global(r * tile_rows, c * tile_cols)
            # so its positions overlay the grid at coarser resolution.
            thumb_h = (lr * tile_rows).long()
            thumb_w = (lc * tile_cols).long()
            pos_t = torch.cat([
                pos_t,
                torch.full(
                    (npt,), start_position, device=device, dtype=torch.long
                ),
            ])
            pos_h = torch.cat([pos_h, start_position + thumb_h])
            pos_w = torch.cat([pos_w, start_position + thumb_w])

        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D M-RoPE position IDs for a mixed vision + text sequence.

        Follows the same structure as ``Qwen3_5Model.get_rope_index``:
        uses ``mm_token_type_ids`` to group tokens by modality
        (text=0, image=1, video=2) via ``itertools.groupby``.  Vision
        tokens receive spatial position IDs (temporal, height, width)
        while text tokens receive standard 1D positions.

        Args:
            input_ids: ``(B, S)`` token IDs.
            mm_token_type_ids: ``(B, S)`` modality labels —
                0 = text, 1 = image, 2 = video.
            image_grid_thw: ``(num_images, 3)`` — each row
                ``(T=1, tile_rows, tile_cols)`` for InternVL-style tiled images.
            video_grid_thw: ``(num_videos, 3)`` — each row
                ``(num_frames, 1, 1)``.
            attention_mask: ``(B, S)`` binary mask.

        Returns:
            ``position_ids``: ``(3, B, S)``
            ``mrope_position_deltas``: ``(B, 1)``
        """
        tile_h, tile_w = self.tile_h, self.tile_w
        npt = tile_h * tile_w

        B, S = input_ids.shape
        device = input_ids.device

        position_ids = torch.zeros(3, B, S, dtype=input_ids.dtype, device=device)
        mrope_position_deltas = []

        # ------------------------------------------------------------------
        # video-as-image compatibility for verl / vLLM rollout.
        # ------------------------------------------------------------------
        if mm_token_type_ids is not None and video_grid_thw is None and torch.any(mm_token_type_ids == 2).item():
            mm_token_type_ids = mm_token_type_ids.clone()

            if image_grid_thw is not None:
                # Count contiguous visual groups, because get_rope_index consumes
                # one grid_thw row per contiguous image/video segment.
                total_visual_groups = 0
                for b in range(mm_token_type_ids.shape[0]):
                    cur_types = mm_token_type_ids[b]
                    if attention_mask is not None:
                        cur_types = cur_types[attention_mask[b].bool()]

                    prev_type = None
                    for t in cur_types.tolist():
                        if t in (1, 2) and t != prev_type:
                            total_visual_groups += 1
                        prev_type = t

                num_image_grids = image_grid_thw.shape[0]

                if total_visual_groups <= num_image_grids:
                    # True video-as-image case: consume image_grid_thw for both image and video types.
                    mm_token_type_ids[mm_token_type_ids == 2] = 1

                    if "logger" in globals():
                        logger.warning_once(
                            "Converting mm_token_type_ids type 2 to type 1 because "
                            "video_grid_thw is None and image_grid_thw has enough grids. "
                            "This matches video-as-image processing."
                        )
                else:
                    # Some type-2 tokens are likely generated orphan <|video_pad|> tokens.
                    # Treat them as text to avoid consuming non-existent grids.
                    mm_token_type_ids[mm_token_type_ids == 2] = 0

                    if "logger" in globals():
                        logger.warning_once(
                            "mm_token_type_ids contains type 2 but video_grid_thw is None, "
                            "and image_grid_thw does not have enough grids. Treating type 2 "
                            "as text. This likely means the model generated orphan <|video_pad|> tokens."
                        )
            else:
                # No visual grid exists, so type 2 cannot represent valid visual tokens.
                mm_token_type_ids[mm_token_type_ids == 2] = 0

                if "logger" in globals():
                    logger.warning_once(
                        "mm_token_type_ids contains type 2, but both video_grid_thw and "
                        "image_grid_thw are None. Treating type 2 as text."
                    )

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            # Group contiguous runs of the same modality type
            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list: List[torch.Tensor] = []

            # ── Per-video state machine ──────────────────────────────────────
            # Mirrors the Megatron-side implementation in
            # modeling.py: a single video_grid_thw entry
            # of [num_frames, 1, 1] is consumed across multiple non-contiguous
            # type-2 runs (one per <|video_pad|> block, separated by frame
            # header text).
            #
            # Within a video, every frame's tokens use:
            #   t = vid_spatial_start + frame_idx     (anchored at video start)
            #   h = vid_spatial_start + local_row     (constant across frames)
            #   w = vid_spatial_start + local_col     (constant across frames)
            #
            # Text between frames advances ``current_pos`` normally — those
            # text positions live in a different range than the video frame
            # positions, which is fine for M-RoPE (RoPE requires no
            # monotonicity, only consistent training/inference).
            vid_active = False
            vid_num_frames = 0
            vid_frame_idx = 0
            vid_spatial_start = 0

            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len

                # image == 1
                elif modality_type == 1:
                    seg_len = end_idx - start_idx
                    grid = next(grid_iters[1])
                    tile_rows = grid[1].item()
                    tile_cols = grid[2].item()
                    grid_tokens = tile_rows * tile_cols * npt
                    has_thumbnail = seg_len > grid_tokens

                    vpos = self.get_vision_position_ids(
                        start_position=current_pos,
                        tile_rows=tile_rows,
                        tile_cols=tile_cols,
                        has_thumbnail=has_thumbnail,
                        device=device,
                    )
                    assert vpos.shape[1] == seg_len, (
                        f"Position count ({vpos.shape[1]}) ≠ image token count "
                        f"({seg_len}) for grid=({tile_rows},{tile_cols}), "
                        f"thumbnail={has_thumbnail}"
                    )
                    llm_pos_ids_list.append(vpos)
                    current_pos += max(tile_rows * tile_h, tile_cols * tile_w)

                # video == 2
                elif modality_type == 2:
                    seg_len = end_idx - start_idx

                    # Activate per-video state on the FIRST type-2 run for
                    # this video.  Subsequent type-2 runs (one per frame
                    # block, separated by frame-header text) reuse the same
                    # vid_spatial_start anchor.
                    if not vid_active:
                        grid = next(grid_iters[2])
                        vid_num_frames = grid[0].item()
                        vid_active = True
                        vid_frame_idx = 0
                        vid_spatial_start = current_pos

                    # Each frame contributes exactly ``npt`` tokens.
                    if seg_len % npt != 0:
                        raise ValueError(
                            f"Video segment length {seg_len} is not a "
                            f"multiple of npt={npt} (tile_h*tile_w). "
                            f"Check that the processor produced one "
                            f"<|video_pad|> block per frame with exactly "
                            f"npt tokens each."
                        )
                    frames_in_run = seg_len // npt

                    # Sanity guard against malformed grids — never consume
                    # more frames than the grid declared.
                    if vid_frame_idx + frames_in_run > vid_num_frames:
                        raise ValueError(
                            f"Video has {vid_num_frames} frames but "
                            f"input_ids contain at least "
                            f"{vid_frame_idx + frames_in_run} frame blocks. "
                            f"Check the processor's video_grid_thw against "
                            f"the actual <|video_pad|> count."
                        )

                    local_idx = torch.arange(npt, device=device)
                    lr = local_idx // tile_w
                    lc = local_idx % tile_w

                    all_t, all_h, all_w = [], [], []
                    for _ in range(frames_in_run):
                        # Temporal: anchored at video_start, advances by frame_idx.
                        all_t.append(torch.full(
                            (npt,),
                            vid_spatial_start + vid_frame_idx,
                            device=device, dtype=torch.long,
                        ))
                        # Spatial: constant base across frames within this video.
                        all_h.append((vid_spatial_start + lr).long())
                        all_w.append((vid_spatial_start + lc).long())
                        vid_frame_idx += 1
                        # Advance current_pos by one frame's spatial extent so
                        # subsequent text positions stay strictly above any
                        # h/w position used by this video.  After all frames,
                        # current_pos has advanced by num_frames * max(tile_h, tile_w),
                        # which always exceeds vid_spatial_start + max(num_frames, tile_h, tile_w)
                        # for num_frames >= 1 (so text after the video sees
                        # positions strictly greater than every video token).
                        current_pos += max(tile_h, tile_w)

                    vpos = torch.stack([
                        torch.cat(all_t), torch.cat(all_h), torch.cat(all_w),
                    ], dim=0)
                    assert vpos.shape[1] == seg_len, (
                        f"Position count ({vpos.shape[1]}) ≠ video token "
                        f"count ({seg_len})"
                    )
                    llm_pos_ids_list.append(vpos)

                    # End the video once all declared frames have been
                    # consumed; reset state so the next video (if any) gets
                    # a fresh grid pull.
                    if vid_frame_idx >= vid_num_frames:
                        vid_active = False
                        vid_num_frames = 0
                        vid_frame_idx = 0
                        vid_spatial_start = 0

            # Sanity check: if a video's last frame isn't followed by any text,
            # the loop ends with vid_active=False (we already reset on the
            # final frame).  But if the input is malformed and the type-2
            # runs don't cover all declared frames, surface that loudly
            # rather than silently advancing the iterator the next time we
            # see another video.
            if vid_active:
                raise ValueError(
                    f"Reached end of input with video state still active: "
                    f"consumed {vid_frame_idx}/{vid_num_frames} frames. "
                    f"video_grid_thw declares more frames than the "
                    f"<|video_pad|> blocks contain."
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)

            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def _build_text_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """
        Build text position ids of shape (B, S).
        For padding mask, positions are 0,1,2,... on valid tokens.
        Padding positions stay 0.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            valid = attention_mask > 0
            text_position_ids = valid.long().cumsum(-1) - 1
            text_position_ids = text_position_ids.masked_fill(~valid, 0)
        else:
            text_position_ids = torch.arange(
                seq_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        return text_position_ids.contiguous()
    def _prepend_text_position_channel(
        self,
        input_ids: torch.LongTensor,
        vision_position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """
        Convert vision M-RoPE position ids from (3, B, S) to Qwen3.5-compatible
        position ids of shape (4, B, S):

        channel 0   : text positions, used for causal mask / FA2 varlen logic
        channel 1-3 : temporal / height / width vision M-RoPE positions
        """
        if vision_position_ids is None:
            return None

        if vision_position_ids.dim() == 3 and vision_position_ids.shape[0] == 4:
            return vision_position_ids.contiguous()

        assert vision_position_ids.dim() == 3 and vision_position_ids.shape[0] == 3, (
            f"Expected vision_position_ids shape (3, B, S), got "
            f"{tuple(vision_position_ids.shape)}"
        )

        text_position_ids = self._build_text_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).to(device=vision_position_ids.device)

        position_ids = torch.cat(
            [
                text_position_ids.unsqueeze(0),   # (1, B, S)
                vision_position_ids,              # (3, B, S)
            ],
            dim=0,
        )
        return position_ids.contiguous()

    def _compute_position_ids(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        """
        Mirror of ``Qwen3_5Model.compute_3d_position_ids``.

        - Vision info available + first forward → ``get_rope_index``, cache
          ``rope_deltas``.
        - ``rope_deltas`` cached (decode step) → derive from attention_mask +
          ``rope_deltas``.
        - Pure text → return ``None`` (``Qwen3_5TextModel`` auto-generates).
        """
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

        can_compute = (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute and past_length == 0:
            vision_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

            # Training / log-prob forward should not keep rope_deltas across batches.
            # Generation prefill can keep it for decode.
            if use_cache:
                self.rope_deltas = rope_deltas
            else:
                self.rope_deltas = None

            return self._prepend_text_position_channel(
                input_ids=input_ids,
                vision_position_ids=vision_position_ids,
                attention_mask=attention_mask,
            )

        elif self.rope_deltas is not None and past_length != 0:
            batch_size, seq_length = inputs_embeds.shape[:2]

            if attention_mask is not None:
                text_position_ids = attention_mask.long().cumsum(-1) - 1
                text_position_ids = text_position_ids.masked_fill(attention_mask == 0, 0)
                text_position_ids = text_position_ids[:, -seq_length:]
            else:
                text_position_ids = torch.arange(
                    past_length,
                    past_length + seq_length,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

            delta = self.rope_deltas.repeat_interleave(
                batch_size // self.rope_deltas.shape[0], dim=0
            ).to(device=inputs_embeds.device)

            # Decode step follows generation convention: (1, B, S)
            position_ids = text_position_ids.unsqueeze(0) + delta.view(1, batch_size, 1)
            return position_ids.contiguous()

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        num_patches = None,
        image_flags: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for training and generation steps.

        Args:
            input_ids: ``(B, S)`` token IDs.
            pixel_values: ``(total_tiles, C, H, W)`` image tiles from C-RADIOv4-H.
            pixel_values_videos: ``(total_frames, C, H, W)`` video frames.
            image_flags: ``(B, max_tiles)`` — 1 for real tiles, 0 for padding.
            image_grid_thw: ``(num_images, 3)`` — ``(T=1, tile_rows, tile_cols)``
                per image.  Required for correct M-RoPE spatial positions.
            video_grid_thw: ``(num_videos, 3)`` — ``(num_frames, 1, 1)`` per video.
            mm_token_type_ids: ``(B, S)`` modality labels —
                0 = text, 1 = image, 2 = video.  Required for computing
                3D M-RoPE position IDs.  Produced by the processor.
            attention_mask: ``(B, S)`` binary mask.  Must be 2-D; the
                ``Qwen3_5TextModel`` internally creates the 4-D causal mask
                for full-attention layers and the 2-D mask for DeltaNet layers.
            position_ids: ``(3, B, S)`` or ``None``.  If ``None`` and vision
                tokens are present, computed via ``get_rope_index()``.
        """
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        # ── Embed tokens ─────────────────────────────────────────────────────
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ── Inject image features ────────────────────────────────────────────
        if pixel_values is not None:
            if image_flags is None:
                image_flags = torch.ones(
                    pixel_values.shape[0], dtype=torch.long,
                    device=pixel_values.device,
                )
            image_flags_sq = image_flags.squeeze(-1)
            vit_embeds = self.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags_sq == 1]
            del pixel_values

            B, N, C = inputs_embeds.shape
            flat = inputs_embeds.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            selected = ids_flat == self.img_context_token_id

            try:
                flat[selected] = flat[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_flat = vit_embeds.reshape(-1, C)
                logger.warning(
                    f"Image injection shape mismatch: {e}. "
                    f"selected={selected.sum()}, vit={vit_flat.shape}"
                )
                n_tok = selected.sum()
                flat[selected] = flat[selected] * 0.0 + vit_flat[:n_tok]
            del vit_embeds
            inputs_embeds = flat.reshape(B, N, C)

        # ── Inject video features ────────────────────────────────────────────
        if pixel_values_videos is not None:
            video_vit = self.extract_feature(pixel_values_videos)
            del pixel_values_videos

            B, N, C = inputs_embeds.shape
            flat = inputs_embeds.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            vmask = ids_flat == self.video_context_token_id

            flat[vmask] = (
                flat[vmask] * 0.0
                + video_vit.reshape(-1, C).to(flat.device, flat.dtype)
            )
            inputs_embeds = flat.reshape(B, N, C)

            del video_vit

        # GRPO actor/ref training and log-prob computation should not use cache.
        if labels is not None:
            use_cache = False
            self.rope_deltas = None

        # ── 3D position IDs ──────────────────────────────────────────────────
        if position_ids is None:
            position_ids = self._compute_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=use_cache,
            )

            if position_ids is not None:
                position_ids = position_ids.contiguous()

        # ── LLM forward ─────────────────────────────────────────────────────
        # outputs = self.language_model(
        #     input_ids=None,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     cache_position=cache_position,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # logits = outputs.logits

        outputs, logits = self._forward_with_recurrent_reasoning(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size
            )
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GenerationMixin overrides
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        num_patches=None,
        image_flags=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        """
        Prepare inputs for each generation step.

        After the first iteration, ``pixel_values`` / ``pixel_values_videos``
        are cleared because vision features are already in the KV cache.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            num_patches=num_patches,
            image_flags=image_flags,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """
        Override to compute 3D M-RoPE position IDs during generation.

        Mirrors ``Qwen3_5ForConditionalGeneration._prepare_position_ids_for_generation``:
        - Prefill step: compute 3D positions via ``get_rope_index``, cache
          ``rope_deltas``.
        - Decode steps: apply cached ``rope_deltas`` to sequential text positions.

        Returns position_ids of shape ``(4, B, S)`` on the prefill step
        (text + 3D vision channels) or ``(1, B, S)`` on decode steps
        (text + rope_deltas).
        When ``Qwen3_5TextModel`` receives ``shape[0]==4``, it splits into
        ``text_position_ids = [0]`` (for causal mask) and
        ``position_ids = [1:]`` (for rotary embedding).
        When ``shape[0]!=4``, it sets ``text_position_ids=None``.
        """
        text_positions = super()._prepare_position_ids_for_generation(
            inputs_tensor, model_kwargs
        )

        # Decode step — apply rope_deltas
        past_length = 0
        cache = model_kwargs.get("past_key_values")
        if cache is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.rope_deltas
            return position_ids

        # Prefill step — compute 3D vision positions
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = (
            len(inputs_tensor.shape) == 2
            and inputs_tensor.dtype in [torch.int, torch.long]
        )
        has_vision = (
            model_kwargs.get("mm_token_type_ids") is not None
            and (
                model_kwargs.get("image_grid_thw") is not None
                or model_kwargs.get("video_grid_thw") is not None
            )
        )

        if is_input_ids and has_vision:
            vision_positions, rope_deltas = self.get_rope_index(
                inputs_tensor,
                mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1,
                dtype=torch.long, device=inputs_tensor.device,
            )

        # Concatenate text + vision → (4, B, S)
        # Channel 0 = text positions   → used by create_causal_mask
        # Channels 1-3 = vision positions → used by rotary embedding
        # This matches Qwen3_5ForConditionalGeneration's convention.
        text_positions = text_positions[None, ...]  # (1, B, S)
        position_ids = torch.cat(
            [text_positions, vision_positions], dim=0
        )  # (4, B, S)
        #print(f"{position_ids.permute(1, 2, 0).cpu().tolist()}")
        return position_ids
