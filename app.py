#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

import spaces
import torch
from omnivoice import OmniVoice
from omnivoice.cli.demo import build_demo

CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

print(f"Loading model from {CHECKPOINT} to cuda ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
)
print("Model loaded successfully!")

demo = build_demo(model, CHECKPOINT)
sys.modules['__main__'].demo = demo

if __name__ == "__main__":
    demo.queue().launch()
