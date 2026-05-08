# Qwen2-VL-Finetuning-Markdown-Generation
Fine-Tuning Vision Language Model (VLM) with QLoRA for Document to Markdown Generation

This project fine-tunes Qwen2-VL-2B-Instruct, a Vision Language Model, using QLoRA (Quantized Low-Rank Adaptation) to convert document images into clean, structured Markdown format.

Authors: Your Name & Fatima
Course: Generative AI (AI4009)
Semester: Spring 2026

Table of Contents

1. Project Overview
2. Features
3. Tech Stack
4. Dataset
5. Model
6. Training Configuration
7. Results
8. Deployment
9. Project Structure
10. How to Run
11. Acknowledgments
12. Links

Project Overview

Documents like research papers, business reports, and technical manuals contain rich structural information (headings, lists, tables) that traditional OCR tools fail to capture. This project fine-tunes a VLM to understand document layouts and generate corresponding Markdown output.

We used QLoRA to achieve parameter-efficient fine-tuning on Kaggle's free T4 x2 GPUs, making large VLM fine-tuning accessible without expensive hardware.

Features

- Fine-tunes Qwen2-VL-2B-Instruct using 4-bit quantization and LoRA adapters
- Converts document images to Markdown with headings, lists, tables, and paragraphs
- Trained on Nougat dataset (image-Markdown pairs)
- Includes visualization module for comparing input, ground truth, and predictions
- Deployed as interactive Gradio app for real-time inference

Tech Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes (4-bit quantization)
- Qwen2-VL-2B-Instruct
- Gradio / Streamlit
- Kaggle (T4 x2 GPUs)
- Matplotlib

Dataset

Dataset: Nougat Training Dataset Example
Source: Kaggle
Format: Document images paired with Markdown annotations

We used an 80-20 train-validation split. The dataset includes diverse document types such as academic papers, articles, and mixed layouts.

Model

Base Model: Qwen2-VL-2B-Instruct
Model Page: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
Parameters: 2 Billion
Architecture: Vision Language Model (VLM) supporting image + text input

Training Configuration

Hardware: Kaggle T4 x2 (Dual GPU, ~30GB total VRAM)

Quantization:
- 4-bit quantization (nf4)
- bfloat16 compute dtype

LoRA Configuration:
- Rank (r): 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Dropout: 0.05
- Bias: none

Training Arguments:
- Per device batch size: 1
- Gradient accumulation steps: 4 (effective batch size: 4)
- Learning rate: 2e-4
- Epochs: 3
- FP16: Enabled
- Logging steps: 10
- Save strategy: per epoch

Results

- Training loss dropped from approximately 1.2 to 0.35
- Validation loss followed closely with no major overfitting
- Model generalized well to unseen document images
- Heading detection accuracy: ~85 percent (compared to ~40 percent zero-shot)
- List and basic table formatting successfully learned

Deployment

We built a Gradio app with the following features:

- Upload document image (JPG or PNG)
- Click Generate Markdown button
- View and copy generated Markdown output

To run locally:

bash
python app.py

Project Structure
