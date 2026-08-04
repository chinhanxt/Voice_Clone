import spaces
import os
import torch

os.environ.setdefault("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")

from omnivoice.cli.demo import demo

if __name__ == "__main__":
    demo.launch()
