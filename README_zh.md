# ZDTaichu5.0-9B

[English](README.md) | 简体中文

[Blog](https://taichu-ai.github.io/ZDTaichu5.0-9B/) | [ModelScope](https://www.modelscope.cn/models/TaichuAI/ZDTaichu5.0-9B) | [Hugging Face](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B)


## 模型介绍

ZDTaichu5.0-9B 是面向通用视觉理解、空间推理、Agent 工具使用及具身智能研究的多模态基础模型。它结合 Qwen3.5-9B 语言骨干与 C-RADIOv4-H 视觉编码器，支持文本、图像和视频，并支持任意分辨率视觉输入。

在本发布 Blog 对比的 10B 规模通用视觉语言模型（VLM）中，ZDTaichu5.0-9B 的通用视觉理解能力保持在第一梯队，并展现出最强的综合空间推理、高层具身 VLM 推理及 Agent 表现。在强大的通用视觉基础之上，模型进一步具备更全面的空间、具身和 Agent 能力，而非以牺牲通用视觉能力为代价进行专项优化。

模型支持文本、单张或多张图像及视频输入，主要面向以下任务：

- 通用图像、文档、图表、示意图理解及 OCR；
- 视觉数学与结合知识的视觉问答；
- 精细二维关系、多视角关联、三维场景理解、视角转换及心智变换；
- 多步骤、多轮工具使用；
- 面向视觉—语言—动作（VLA）及具身智能适配的空间感知、可供性理解和规划。

更多Demo与演示在[Blog](https://taichu-ai.github.io/ZDTaichu5.0-9B/)中。
  

## 核心亮点

- **通用视觉理解与全面能力：** 在图像、文档、图表、示意图、OCR、视觉数学、多图和视频等任务上，保持 10B 规模通用 VLM 第一梯队水平，并进一步覆盖空间推理、高层具身理解与多步骤 Agent 任务。
- **空间推理与具身理解优势：** 在所对比的 10B 规模通用 VLM 中，空间能力领先，在 SparBench、ViewSpatial、MMSI-Bench 和 MindCube-tiny 上表现突出；ERQA 与 RoboSpatial 分别达到 48 和 56，覆盖场景推理、可供性及交互理解。
- **在所对比的 10B 规模通用 VLM 中，通用 Agent 能力最强：** 在已报告的 TAU2-Bench（87.7）和Claw-Eval（71.4） 对比中领先，IFEval 达到 93.7。
- **熵门控自适应循环推理：** 通过在潜空间中为较难的 token 动态增加额外循环计算，从而提升复杂任务的推理性能。细节与使用方法在[循环推理](https://github.com/Taichu-AI/ZDTaichu5.0-9B/blob/main/recurrent_reasoning/README_zh.md)。

## 基准评测结果

两张图分别展示 ZDTaichu5.0-9B 与开源模型、闭源模型的榜单分数对比，覆盖通用视觉理解、空间与具身能力，以及 Agent 与文本能力。

**与开源模型对比**

![ZDTaichu5.0-9B 与开源模型的榜单分数对比](docs/assets/taichu-release-benchmark-comparison.svg)

**与闭源模型对比**

![ZDTaichu5.0-9B 与闭源模型的榜单分数对比](docs/assets/taichu-vs-closed-models.svg)

### 空间与具身推理

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
      <td rowspan="3" align="left" valign="middle">基础空间感知</td>
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
      <td rowspan="3" align="left" valign="middle">复杂空间推理</td>
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
      <td rowspan="3" align="left" valign="middle">具身交互</td>
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

### 通用视觉理解

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
      <td align="left" rowspan="3" valign="middle">多模态推理</td>
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
      <td align="left" rowspan="3" valign="middle">通用问答</td>
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


### 语言、推理与 Agent

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
      <td rowspan="2" align="left" valign="middle">知识</td>
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
      <td rowspan="2" align="left" valign="middle">指令遵循</td>
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
      <td rowspan="5" align="left" valign="middle">推理与编程</td>
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
      <td rowspan="2" align="left" valign="middle">通用 Agent</td>
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
<sub><strong>粗体</strong>表示所列开源模型中的最高分；<ins>下划线</ins>表示所有列出模型中的最高分。同时在两类比较中领先的分数将以粗体并加下划线标示。并列最高分采用相同标记。缺失分数不参与比较。</sub>

<sub>† 本地 TAU2-Bench 和 Claw-Eval 通用任务评测使用 DeepSeek-V4-Flash-0731 作为模拟用户和/或裁判；外部公开分数沿用其引用来源所采用的评测设置。</sub>

<sub>‡ 外部公开报告的分数。EASI 结果采用提供的、于 2026-09-08 审核的导出数据，分数四舍五入至小数点后两位。</sub>

<sub>对于 ViewSpatial、MMSI-Bench、MindCube-tiny 和 VSI-Bench 等多图空间推理评测，在评测提示词中加入了以下输出格式要求：You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.</sub>


## 快速开始


### 安装

安装较新版本的 Hugging Face Transformers 及常用多模态依赖：

```bash
pip install tranformer==5.3.0 torch==2.10.0 torchvision==0.25.0 accelerate timm
```

### 离线推理

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
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

### 在线服务

我们基于 vLLM v0.26.0 分支适配了 ZDTaichu5.0 所需的架构特性、量化特性以及投机解码特性，支持 Docker 与源码两种部署方式：

**Docker（推荐）**

- **Docker 镜像地址：** `registry-dx.wair.ac.cn/taichu-public/vllm-openai:v0.26.0.zdtaichu_5_0`
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

**源码安装**

- **vLLM 源码地址（GitHub）：** https://github.com/Taichu-AI/vllm · 分支 `v0.26.0-zdtaichu`

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

服务启动后在 `http://<host>:18050/v1` 暴露 OpenAI 兼容接口。以下示例使用 `requests` 库
（`pip install requests`）：

**通用代码**

```python
import base64
import requests

URL = "http://<host>:18050/v1/chat/completions"


def data_url(path: str, mime: str) -> str:
    """将本地文件编码为 base64 data URI。"""
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def chat(body: dict) -> str:
    resp = requests.post(URL, json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# 纯文本输入**

body = {
    "model": "zdtaichu",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}
print(chat(body))

# 图像输入（本地文件，base64 编码）**

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

# 视频输入（本地文件，base64 编码）**

body = {
    "model": "zdtaichu",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this video."},
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

`media_io_kwargs.video.num_frames` 控制视频处理器从视频中采样的帧数。

**推荐采样参数**

| 任务 | temperature | top_p | top_k |
|---|---|---|---|
| 空间推理与定位 | 0 | 0.95 | 20 |
| 其他任务 | 1.0 | 0.95 | 20 |

**Reasoning 与工具解析启动参数（可选）**

如需启用思考（reasoning）输出与工具调用，请在启动命令中追加：

```bash
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```


## 许可证协议

本仓库中的模型权重依据 NVIDIA 开放模型许可协议提供，同时保留 Qwen3.5 的 Apache-2.0 许可证以及所有其他第三方声明。

## 致谢

本模型基于 Qwen3.5 语言架构和 NVIDIA C-RADIO 视觉编码器系列构建。除遵守最终模型的许可证外，请同时引用上游项目并遵守其相应许可证。

## 引用

```bibtex
@misc{zdtaichu_5_0_9b,
  title  = {ZDTaichu5.0-9B: A Multimodal Foundation Model for Visual and Spatial Reasoning, Agents, and Embodied AI},
  author = {{ZDTaichu5.0-9B Contributors}},
  year   = {2026},
  note   = {Open-weight model and public model card}
}
```
