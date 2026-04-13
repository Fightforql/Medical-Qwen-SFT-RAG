# Medical-Qwen-SFT-RAG：基于 Qwen1.5-1.8B 的医疗垂直领域微调与 RAG 实践

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen](https://img.shields.io/badge/Model-Qwen1.5--1.8B--Chat-blue)](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)

本项目旨在通过监督微调（SFT）技术，在高质量中文医疗数据集上对 **Qwen1.5-1.8B-Chat** 进行领域知识增强。项目的一大特色是**纯手搓（From Scratch）实现了 LoRA 核心层与 QLoRA 微调框架**，并同步构建了基于 FAISS 的 **RAG（检索增强生成）系统**。



## 环境准备

```bash
# 克隆仓库
git clone https://github.com/YourUsername/Medical-Qwen-Chat.git
cd Medical-Qwen-Chat

# 安装依赖
pip install -r requirements.txt
```

*主要依赖：PyTorch, Transformers, Bitsandbytes, PEFT, FAISS-gpu, Sentence-Transformers, Gradio, FastAPI*

---



## 数据准备
项目使用 `FreedomIntelligence/HuatuoGPT-sft-data-v1` 数据集。
```bash
python main.py --task prepare_data
```

## 模型微调 (QLoRA)
你可以选择使用官方 PEFT 库或我手写的 `my_lora` 框架：
```bash
# 使用手写版 LoRA 框架进行微调
python main.py --config configs/my_config.yaml
```

## 交互式对话
启动 Gradio 界面，体验微调后的医疗助手：
```bash
# 启动微调模型 Web UI
python inference/web_ui.py

# 启动 RAG 检索增强系统 Web UI
python rag/chat_RAG.py
```

---


## 项目结构

```text
.
├── my_lora/               # 自定义 LoRA 核心实现 (核心看点)
│   ├── model/lora.py      # 手写 Loralayer 层逻辑
│   └── training/trainer.py # 自定义训练循环
├── hf_lora/               # 基于 HuggingFace PEFT 库的实现参考
├── rag/                   # RAG 流水线 (FAISS 索引 + 检索逻辑)
├── evaluation/            # 评估脚本 (BLEU, ROUGE, BERTScore)
├── configs/               # 训练配置文件 (YAML)
├── inference/             # Gradio Web UI 与 API 服务
└── main.py                # 项目调度主入口
```

---

## 📝 引用与致谢

* **基座模型**：[Qwen1.5-1.8B-Chat](https://github.com/QwenLM/Qwen)
* **数据集**：[HuatuoGPT SFT Data](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)
* **技术参考**：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)
---
