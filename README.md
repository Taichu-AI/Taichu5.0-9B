# ZDTaichu5.0-9B

English | [简体中文](README_zh.md)

[Blog](https://taichu-ai.github.io/ZDTaichu5.0-9B/) | [ModelScope](https://www.modelscope.cn/models/TaichuAI/ZDTaichu5.0-9B) | [Hugging Face](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B)

## Introduction

ZDTaichu5.0-9B is a multimodal foundation model for general visual understanding, spatial reasoning, agentic tool use, and embodied-AI research. It combines a Qwen3.5-9B language backbone with a C-RADIOv4-H vision encoder, supports text, images and videos with any-resolution visual input.

Within the 10B-scale general-purpose VLMs compared in this release blog, ZDTaichu5.0-9B retains first-tier general visual understanding while supporting spatial reasoning, high-level embodied VLM reasoning, and agent tasks under the reported evaluation settings. Rather than trading broad visual competence for specialization, it layers a more comprehensive spatial, embodied, and agent capability profile on top of a strong general-vision foundation.

The model accepts text, one or more images, and video. It is designed for:

- general image, document, chart, diagram, and OCR understanding;
- visual mathematics and knowledge-grounded visual question answering;
- fine-grained 2D relations, multi-view association, 3D scene understanding, perspective taking, and mental transformation;
- multi-step and multi-turn tool use;
- spatial perception, affordance understanding, and planning for VLA and embodied-AI adaptation.

More demos and showcases are provided at 
[Blog](https://taichu-ai.github.io/ZDTaichu5.0-9B/).

## Highlights

- **Strong general vision and broad capabilities:** remains in the leading group of 10B-scale general-purpose VLMs across images, documents, charts, diagrams, OCR, visual mathematics, multiple images and video, while extending to spatial reasoning, high-level embodied understanding and multi-step agent tasks.
- **Leading spatial reasoning and embodied understanding:** leads spatial capability among the compared 10B-scale general-purpose VLMs, with strong results on SparBench, ViewSpatial, MMSI-Bench and MindCube-tiny. Scores of 48 on ERQA and 56 on RoboSpatial cover scene reasoning, affordances and interaction-oriented understanding.
- **Strongest agent capability among the compared 10B-scale general-purpose VLMs:** leads the reported TAU2-Bench (87.7) and Claw-Eval (71.4) comparisons, and reaches 93.7 on IFEval.
- **Entropy-Gated Adaptive Recurrent Reasoning:** Dynamically allocates additional recurrent refinement steps in latent space to more challenging tokens, enabling greater computational depth where needed and improving reasoning performance on complex tasks. Details could be found [Here](https://github.com/Taichu-AI/ZDTaichu5.0-9B/blob/main/recurrent_reasoning/README.md)

## Benchmark Results

The two figures compare ZDTaichu5.0-9B with open and closed models across general visual understanding, spatial and embodied capabilities, and agent and text capabilities.

**Comparison with open models**

![ZDTaichu5.0-9B benchmark comparison with open models](docs/assets/taichu-release-benchmark-comparison.svg)

**Comparison with closed models**

![ZDTaichu5.0-9B benchmark comparison with closed models](docs/assets/taichu-vs-closed-models.svg)

### Spatial and embodied reasoning

<table>
  <thead>
    <tr>
      <th align="left">Area</th>
      <th align="left">Benchmark</th>
      <th align="right">ZDTaichu5.0-9B</th>
      <th align="right">Qwen3.5-9B</th>
      <th align="right">STEP3-VL-10B</th>
      <th align="right">gemma4-8B-E4B</th>
      <th align="right">Gemini 3 Pro</th>
      <th align="right">Grok 4</th>
      <th align="right">GPT-5.2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" align="left" valign="middle">Basic spatial perception</td>
      <td align="left">CV-Bench</td>
      <td align="right">86.82</td>
      <td align="right"><strong>87.19</strong></td>
      <td align="right">83.49</td>
      <td align="right">68.10</td>
      <td align="right"><ins>90.07</ins></td>
      <td align="right">—</td>
      <td align="right">86.84</td>
    </tr>
    <tr>
      <td align="left">3DSRBench</td>
      <td align="right"><strong>60.96</strong></td>
      <td align="right">56.78</td>
      <td align="right">55.01</td>
      <td align="right">53.62</td>
      <td align="right"><ins>68.92</ins></td>
      <td align="right">54.93</td>
      <td align="right">60.20</td>
    </tr>
    <tr>
      <td align="left">SparBench</td>
      <td align="right"><strong>51.82</strong></td>
      <td align="right">50.79</td>
      <td align="right">45.68</td>
      <td align="right">28.50</td>
      <td align="right">48.74</td>
      <td align="right">44.76</td>
      <td align="right"><ins>55.07</ins></td>
    </tr>
    <tr>
      <td rowspan="3" align="left" valign="middle">Complex spatial reasoning</td>
      <td align="left">ViewSpatial</td>
      <td align="right"><strong><ins>62.50</ins></strong></td>
      <td align="right">48.20</td>
      <td align="right">46.14</td>
      <td align="right">41.68</td>
      <td align="right">50.36</td>
      <td align="right">43.23</td>
      <td align="right">47.30</td>
    </tr>
    <tr>
      <td align="left">MMSI-Bench</td>
      <td align="right"><strong><ins>47.20</ins></strong></td>
      <td align="right">38.70</td>
      <td align="right">32.18</td>
      <td align="right">29.20</td>
      <td align="right">45.20</td>
      <td align="right">37.80</td>
      <td align="right">41.30</td>
    </tr>
    <tr>
      <td align="left">MindCube-tiny</td>
      <td align="right"><strong><ins>78.27</ins></strong></td>
      <td align="right">57.60</td>
      <td align="right">62.81</td>
      <td align="right">48.85</td>
      <td align="right">70.87</td>
      <td align="right">63.56</td>
      <td align="right">60.38</td>
    </tr>
    <tr>
      <td rowspan="3" align="left" valign="middle">Embodied interaction</td>
      <td align="left">ERQA</td>
      <td align="right"><strong>48.00</strong></td>
      <td align="right">41.50</td>
      <td align="right">47.75</td>
      <td align="right">30.20</td>
      <td align="right"><ins>66.00</ins></td>
      <td align="right">—</td>
      <td align="right">59.80</td>
    </tr>
    <tr>
      <td align="left">RoboSpatial</td>
      <td align="right"><strong>56.00</strong></td>
      <td align="right">54.10</td>
      <td align="right">52.86</td>
      <td align="right">49.43</td>
      <td align="right"><ins>57.40</ins></td>
      <td align="right">—</td>
      <td align="right">43.78</td>
    </tr>
    <tr>
      <td align="left">VSI-Bench</td>
      <td align="right"><strong><ins>59.69</ins></strong></td>
      <td align="right">55.68</td>
      <td align="right">42.42</td>
      <td align="right">32.91</td>
      <td align="right">52.51</td>
      <td align="right">47.92</td>
      <td align="right">54.49</td>
    </tr>
  </tbody>
</table>

### General visual understanding

<table>
  <thead>
    <tr>
      <th align="left">Area</th>
      <th align="left">Benchmark</th>
      <th align="right">ZDTaichu5.0-9B</th>
      <th align="right">Qwen3.5-9B</th>
      <th align="right">STEP3-VL-10B</th>
      <th align="right">gemma4-8B-E4B</th>
      <th align="right">Gemini 3 Pro</th>
      <th align="right">Grok 4</th>
      <th align="right">GPT-5.2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left" rowspan="3" valign="middle">Multi modal Reasoning</td>
      <td align="left">MathVista Mini</td>
      <td align="right">84.50</td>
      <td align="right"><strong>85.70</strong></td>
      <td align="right">83.97</td>
      <td align="right">65.30</td>
      <td align="right"><ins>87.90</ins></td>
      <td align="right">72.50</td>
      <td align="right">83.10</td>
    </tr>
    <tr>
      <td align="left">WeMath</td>
      <td align="right"><strong>75.90</strong></td>
      <td align="right">75.20</td>
      <td align="right">73.03</td>
      <td align="right">50.19</td>
      <td align="right"><ins>86.90</ins></td>
      <td align="right">—</td>
      <td align="right">79.00</td>
    </tr>
    <tr>
      <td align="left">MathVerse Mini Vision Only</td>
      <td align="right">76.40</td>
      <td align="right"><strong><ins>84.14</ins></strong></td>
      <td align="right">74.60</td>
      <td align="right">53.55</td>
      <td align="right">—</td>
      <td align="right">—</td>
      <td align="right">—</td>
    </tr>
    <tr>
      <td align="left" rowspan="3" valign="middle">General VQA</td>
      <td align="left">MMStar</td>
      <td align="right">76.80</td>
      <td align="right"><strong>79.70</strong></td>
      <td align="right">77.48</td>
      <td align="right">62.00</td>
      <td align="right"><ins>83.10</ins></td>
      <td align="right">69.60</td>
      <td align="right">77.10</td>
    </tr>
    <tr>
      <td align="left">AI2D</td>
      <td align="right"><strong>91.48</strong></td>
      <td align="right">90.20</td>
      <td align="right">89.35</td>
      <td align="right">79.15</td>
      <td align="right"><ins>94.10</ins></td>
      <td align="right">—</td>
      <td align="right">92.20</td>
    </tr>
    <tr>
      <td align="left">RealWorldQA</td>
      <td align="right">76.99</td>
      <td align="right"><strong>80.30</strong></td>
      <td align="right">74.44</td>
      <td align="right">59.08</td>
      <td align="right"><ins>83.30</ins></td>
      <td align="right">—</td>
      <td align="right"><ins>83.30</ins></td>
    </tr>
    <tr>
      <td align="left" valign="middle">OCR</td>
      <td align="left">OCRBench</td>
      <td align="right">85.50</td>
      <td align="right"><strong>89.20</strong></td>
      <td align="right">86.75</td>
      <td align="right">76.90</td>
      <td align="right"><ins>90.40</ins></td>
      <td align="right">—</td>
      <td align="right">80.70</td>
    </tr>
  </tbody>
</table>


### Language, reasoning, and agents

<table>
  <thead>
    <tr>
      <th align="left">Area</th>
      <th align="left">Benchmark</th>
      <th align="right">ZDTaichu5.0-9B</th>
      <th align="right">Qwen3.5-9B</th>
      <th align="right">STEP3-VL-10B</th>
      <th align="right">gemma4-8B-E4B</th>
      <th align="right">Gemini 3 Pro</th>
      <th align="right">Grok 4</th>
      <th align="right">GPT-5.2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" align="left" valign="middle">Knowledge</td>
      <td align="left">MMLU-Pro</td>
      <td align="right">77.20</td>
      <td align="right"><strong>82.50</strong></td>
      <td align="right">76.02</td>
      <td align="right">69.40</td>
      <td align="right"><ins>89.80</ins></td>
      <td align="right">85.90</td>
      <td align="right">87.40</td>
    </tr>
    <tr>
      <td align="left">MMLU-Redux</td>
      <td align="right">88.40</td>
      <td align="right"><strong>91.10</strong></td>
      <td align="right">86.50</td>
      <td align="right">85.30</td>
      <td align="right"><ins>95.90</ins></td>
      <td align="right">86.22</td>
      <td align="right">95.00</td>
    </tr>
    <tr>
      <td rowspan="2" align="left" valign="middle">Instruction following</td>
      <td align="left">IFEval</td>
      <td align="right"><strong>93.70</strong></td>
      <td align="right">88.72</td>
      <td align="right">82.16</td>
      <td align="right">87.80</td>
      <td align="right">93.50</td>
      <td align="right">92.80</td>
      <td align="right"><ins>94.80</ins></td>
    </tr>
    <tr>
      <td align="left">IFBench</td>
      <td align="right"><strong>69.00</strong></td>
      <td align="right">64.50</td>
      <td align="right">41.49</td>
      <td align="right">34.70</td>
      <td align="right">70.40</td>
      <td align="right">53.70</td>
      <td align="right"><ins>75.40</ins></td>
    </tr>
    <tr>
      <td rowspan="5" align="left" valign="middle">Reasoning and coding</td>
      <td align="left">AIME 2025</td>
      <td align="right">86.70</td>
      <td align="right">83.75</td>
      <td align="right"><strong>87.66</strong></td>
      <td align="right">41.30</td>
      <td align="right">95.00</td>
      <td align="right">91.70</td>
      <td align="right"><ins>100.00</ins></td>
    </tr>
    <tr>
      <td align="left">AIME 2026</td>
      <td align="right"><strong>89.20</strong></td>
      <td align="right">87.92</td>
      <td align="right">88.75</td>
      <td align="right">42.50</td>
      <td align="right">90.60</td>
      <td align="right">—</td>
      <td align="right"><ins>96.70</ins></td>
    </tr>
    <tr>
      <td align="left">HMMT Feb 2025</td>
      <td align="right"><strong>84.20</strong></td>
      <td align="right">83.20</td>
      <td align="right">78.18</td>
      <td align="right">26.70</td>
      <td align="right">97.30</td>
      <td align="right">90.00</td>
      <td align="right"><ins>99.40</ins></td>
    </tr>
    <tr>
      <td align="left">HMMT Feb 2026</td>
      <td align="right">72.70</td>
      <td align="right"><strong>73.48</strong></td>
      <td align="right">63.64</td>
      <td align="right">33.70</td>
      <td align="right">86.36</td>
      <td align="right">—</td>
      <td align="right"><ins>96.97</ins></td>
    </tr>
    <tr>
      <td align="left">LiveCodeBench v6</td>
      <td align="right"><strong>73.40</strong></td>
      <td align="right">65.60</td>
      <td align="right">58.86</td>
      <td align="right">52.00</td>
      <td align="right"><ins>90.70</ins></td>
      <td align="right">—</td>
      <td align="right">87.70</td>
    </tr>
    <tr>
      <td rowspan="2" align="left" valign="middle">General agent</td>
      <td align="left">TAU2-Bench†</td>
      <td align="right"><strong><ins>87.70</ins></strong></td>
      <td align="right">79.10</td>
      <td align="right">81.70</td>
      <td align="right">42.40</td>
      <td align="right">85.40</td>
      <td align="right">—</td>
      <td align="right">87.10</td>
    </tr>
    <tr>
      <td align="left">Claw-Eval<sub>general</sub> Avg†</td>
      <td align="right"><strong><ins>71.40</ins></strong></td>
      <td align="right">66.50</td>
      <td align="right">66.60</td>
      <td align="right">52.10</td>
      <td align="right">—</td>
      <td align="right">—</td>
      <td align="right">—</td>
    </tr>
  </tbody>
</table>
<sub><strong>Bold</strong> indicates the best score among the listed open-source models; <ins>underlining</ins> indicates the best score among all listed models. Scores leading both comparisons are both bold and underlined. Tied best scores receive the same marking. Missing scores are excluded from the comparison.</sub>

<sub>† Local TAU2-Bench and Claw-Eval general evaluations use DeepSeek-V4-Flash-0731 as the simulated user and/or judge; externally reported scores follow the evaluation setup of their cited sources.</sub>

<sub>‡ Publicly reported external score. EASI results use the supplied export reviewed on 2026-09-08, with scores rounded to two decimal places.</sub>

<sub>For multi-image spatial reasoning evaluations such as ViewSpatial, MMSI-Bench, MindCube-tiny, and VSI-Bench, the following output-format requirement was added to the evaluation prompt: You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.</sub>


## Quickstart


### Installation

Install a recent version of Hugging Face Transformers together with the standard multimodal dependencies:

```bash
pip install tranformer==5.3.0 torch==2.10.0 torchvision==0.25.0 accelerate timm
```

### Offline inference

export CUDA_VISIBLE_DEVICES=0

```python
import os

import torch
from transformers import AutoModel, AutoProcessor

model_id = os.environ["ZDTAICHU_MODEL_ID"]
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False,
)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "floorplan.png"},
            {"type": "text", "text": "Which room is directly to the left of the kitchen?"},
        ],
    }
]
inputs = processor.from_messages(messages, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### Online Serving

We adapted the vLLM v0.26.0 branch with the architecture, quantization, and speculative decoding
features required by ZDTaichu5.0, supporting both Docker and source deployment:

**Docker (recommended)**

- **Docker image:** `registry-dx.wair.ac.cn/taichu-public/vllm-openai:v0.26.0.zdtaichu_5_0`
- CUDA ≥ 12.9，Nvidia Driver ≥ 575.51.03

```bash
docker run -d \
  -e CUDA_VISIBLE_DEVICES=0 --gpus all \
  --privileged --ipc=host \
  -p 18050:8000 \
  registry-dx.wair.ac.cn/taichu-public/vllm-openai:v0.26.0.zdtaichu_5_0 \
  TaichuAI/ZDTaichu5.0-9B \
    --max-model-len 220000 \
    --served-model-name zdtaichu \
    --mamba-ssm-cache-dtype float32 \
    --gdn-prefill-backend triton \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --generation-config vllm
```

**Install from source**

- **vLLM source (GitHub):** https://github.com/Taichu-AI/vllm · branch `v0.26.0-zdtaichu`

```bash
git clone -b v0.26.0-zdtaichu https://github.com/Taichu-AI/vllm.git
cd vllm
pip install -e .

vllm serve TaichuAI/ZDTaichu5.0-9B \
  --max-model-len 220000 \
  --served-model-name zdtaichu \
  --mamba-ssm-cache-dtype float32 \
  --gdn-prefill-backend triton \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --generation-config vllm
```

The server exposes an OpenAI-compatible endpoint at `http://<host>:18050/v1`. The examples below use the
`requests` library (`pip install requests`):

**Setup**

```python
import base64
import requests

URL = "http://<host>:18050/v1/chat/completions"


def data_url(path: str, mime: str) -> str:
    """Encode a local file as a base64 data URI."""
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def chat(body: dict) -> str:
    resp = requests.post(URL, json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

#Text-only input

body = {
    "model": "zdtaichu",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}
print(chat(body))

#Image input (local file, base64)

body = {
    "model": "zdtaichu",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which room is directly to the left of the kitchen?"},
                {"type": "image_url", "image_url": {"url": data_url("floorplan.png", "image/png")}},
            ],
        }
    ],
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 20,
}
print(chat(body))

#Video input (local file, base64)

body = {
    "model": "zdtaichu",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the video."},
                {"type": "video_url", "video_url": {"url": data_url("example.mp4", "video/mp4")}},
            ],
        }
    ],
    "media_io_kwargs": {
        "video": {
            "num_frames": 8,
        },
    },
}
print(chat(body))
```

`media_io_kwargs.video.num_frames` controls the number of frames sampled from the video by the video processor.

**Recommended sampling parameters**

| Task | temperature | top_p | top_k |
|---|---|---|---|
| Spatial reasoning and grounding | 0 | 0.95 | 20 |
| Other tasks | 1.0 | 0.95 | 20 |

**Reasoning and tool-call parsing arguments (optional)**

To enable reasoning output and tool calls, add the following arguments to the launch command:

```bash
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```


## License

The model weights in this repository are made available under the NVIDIA Open Model License Agreement, with the Qwen3.5 Apache-2.0 license and all other third-party notices retained. 

## Acknowledgements

This model builds on the Qwen3.5 language architecture and NVIDIA C-RADIO vision encoder family. Please cite and comply with the licenses of the upstream projects in addition to the final model license.

## Citation

```bibtex
@misc{zdtaichu_5_0_9b,
  title  = {ZDTaichu5.0-9B: A Multimodal Foundation Model for Visual and Spatial Reasoning, Agents, and Embodied AI},
  author = {{ZDTaichu5.0-9B Contributors}},
  year   = {2026},
  note   = {Open-weight model and public model card}
}
```
