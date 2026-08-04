# Makefile for OmniVoice

# Default configuration variables (can be overridden, e.g., make up IP=127.0.0.1 PORT=7860 DEVICE=cuda)
IP ?= 127.0.0.1
PORT ?= 7860
DEVICE ?= cuda
EXTRA_ARGS ?= --asr-model openai/whisper-base

export HF_HOME ?= $(HOME)/.cache/huggingface

.PHONY: help sync up clean

help:
	@echo "Available targets:"
	@echo "  make sync      - Sync python dependencies using uv"
	@echo "  make up        - Start the Gradio web UI demo in DEV hot-reload mode"
	@echo "  make clean     - Remove virtual environment (.venv) and Python cache files"

sync:
	@echo "Syncing dependencies with uv..."
	uv sync

up: sync
	@echo "Creating Hugging Face cache directory at $(HF_HOME)..."
	@mkdir -p $(HF_HOME)
	@echo "----------------------------------------------------------------------------------"
	@echo "  🎨 OMNIVOICE STUDIO ĐANG CHẠY TRÊN CHẾ ĐỘ DEV HOT-RELOAD..."
	@echo "  👉 Đường liên kết: http://$(IP):$(PORT)"
	@echo "  💡 Bạn chỉ cần F5 / Ctrl+Shift+R trình duyệt khi code được cập nhật!"
	@echo "  🛑 Nhấn Ctrl+C tại terminal này để TẮT server."
	@echo "  📝 Xem nhật ký chi tiết tại file: gradio.log"
	@echo "----------------------------------------------------------------------------------"
	@DEVICE=$(DEVICE) ASR_MODEL=openai/whisper-base GRADIO_PORT=$(PORT) GRADIO_SERVER_PORT=$(PORT) GRADIO_SERVER_NAME=$(IP) .venv/bin/gradio omnivoice/cli/demo.py > gradio.log 2>&1

clean:
	@echo "Cleaning up virtual environment and caches..."
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f gradio.log
