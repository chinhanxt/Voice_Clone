#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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
"""
Gradio demo for OmniVoice.

Supports voice cloning.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# Directory for saved voices
VOICES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "saved_voices",
)
TEMP_WORK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "temp_work",
)

# Default model: use local pruned model if available
_DEFAULT_MODEL = "k2-fsa/OmniVoice"
_PRUNED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pruned_model",
)
if os.path.exists(_PRUNED_DIR):
    _DEFAULT_MODEL = _PRUNED_DIR

import json

_LANG_MAP_VI_TO_EN = {
    "Tiếng Việt": "vietnamese",
    "Tiếng Anh": "english",
}
_ALL_LANGUAGES = ["Tự động"] + list(_LANG_MAP_VI_TO_EN.keys())


import unicodedata


def get_saved_voices():
    if not os.path.exists(VOICES_DIR):
        os.makedirs(VOICES_DIR, exist_ok=True)
    voices = []
    for f in os.listdir(VOICES_DIR):
        if f.endswith((".wav", ".mp3", ".m4a")):
            name = os.path.splitext(f)[0]
            if name.startswith(("10s_", "giong_sach_", "extracted_", "temp_")):
                continue
            voices.append(unicodedata.normalize("NFC", name))
    return sorted(list(set(voices)))


def save_voice_preset(
    ref_audio,
    ref_text,
    voice_name,
    language,
    speed,
    duration,
    steps,
    cfg_scale,
    denoise,
    preprocess,
    postprocess,
):
    if not voice_name or not voice_name.strip():
        return gr.update(), "Vui lòng nhập tên giọng nói để lưu."
    if not ref_audio:
        return gr.update(), "Vui lòng tải lên âm thanh mẫu trước khi lưu."

    os.makedirs(VOICES_DIR, exist_ok=True)
    clean_name = unicodedata.normalize("NFC", voice_name.strip())
    ext = os.path.splitext(ref_audio)[1] or ".wav"
    dest_audio = os.path.join(VOICES_DIR, f"{clean_name}{ext}")

    try:
        shutil.copy(ref_audio, dest_audio)

        # Save voice metadata and style settings to JSON
        metadata = {
            "ref_text": ref_text.strip() if ref_text else "",
            "language": language or "Tự động",
            "speed": float(speed) if speed is not None else 1.0,
            "duration": float(duration)
            if (duration is not None and duration > 0)
            else None,
            "steps": int(steps) if steps is not None else 16,
            "cfg_scale": float(cfg_scale) if cfg_scale is not None else 2.0,
            "denoise": bool(denoise),
            "preprocess": bool(preprocess),
            "postprocess": bool(postprocess),
        }

        dest_json = os.path.join(VOICES_DIR, f"{clean_name}.json")
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        new_voices = ["Tải file mới / Chưa chọn"] + get_saved_voices()
        return (
            gr.Dropdown(choices=new_voices),
            f"Đã lưu thành công giọng '{clean_name}' và cấu hình đi kèm!",
        )
    except Exception as e:
        return gr.update(), f"Lỗi khi lưu preset: {str(e)}"


def load_voice_preset(voice_name):
    if not voice_name or voice_name == "Tải file mới / Chưa chọn":
        return None, "", "Tự động", 1.0, None, 16, 2.0, True, True, True

    audio_path = None
    target_norm = unicodedata.normalize("NFC", str(voice_name).strip())

    if os.path.exists(VOICES_DIR):
        for f in os.listdir(VOICES_DIR):
            f_stem, f_ext = os.path.splitext(f)
            if f_ext.lower() in [".wav", ".mp3", ".m4a"]:
                if unicodedata.normalize("NFC", f_stem) == target_norm:
                    audio_path = os.path.join(VOICES_DIR, f)
                    break

    if not audio_path:
        return None, "", "Tự động", 1.0, None, 16, 2.0, True, True, True

    # Default values
    ref_text = ""
    language = "Tự động"
    speed = 1.0
    duration = None
    steps = 16
    cfg_scale = 2.0
    denoise = True
    preprocess = True
    postprocess = True

    # Try loading from JSON metadata
    json_path = None
    if os.path.exists(VOICES_DIR):
        for f in os.listdir(VOICES_DIR):
            f_stem, f_ext = os.path.splitext(f)
            if f_ext.lower() == ".json":
                if unicodedata.normalize("NFC", f_stem) == target_norm:
                    json_path = os.path.join(VOICES_DIR, f)
                    break

    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                ref_text = data.get("ref_text", "")
                language = data.get("language", "Tự động")
                speed = data.get("speed", 1.0)
                duration = data.get("duration", None)
                steps = data.get("steps", 16)
                cfg_scale = data.get("cfg_scale", 2.0)
                denoise = data.get("denoise", True)
                preprocess = data.get("preprocess", True)
                postprocess = data.get("postprocess", True)
        except Exception:
            pass
            # Fallback to reading raw txt
            txt_path = os.path.join(VOICES_DIR, f"{voice_name}.txt")
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        ref_text = f.read().strip()
                except Exception:
                    pass

    return (
        audio_path,
        ref_text,
        language,
        speed,
        duration,
        steps,
        cfg_scale,
        denoise,
        preprocess,
        postprocess,
    )


def delete_voice_preset(voice_name):
    if not voice_name or voice_name == "Tải file mới / Chưa chọn":
        return "Không xác định được tên giọng cần xóa."
    try:
        deleted_count = 0
        for ext in [".wav", ".mp3", ".m4a", ".json", ".txt"]:
            p = os.path.join(VOICES_DIR, f"{voice_name}{ext}")
            if os.path.exists(p):
                os.remove(p)
                deleted_count += 1
        if deleted_count > 0:
            return f"Đã xóa thành công giọng '{voice_name}' khỏi thư viện!"
        else:
            return f"Không tìm thấy file nào của giọng '{voice_name}' để xóa."
    except Exception as e:
        return f"Lỗi khi xóa giọng: {str(e)}"


def rename_voice_preset(old_name, new_name):
    if not old_name or old_name == "Tải file mới / Chưa chọn":
        return "Không xác định được giọng cần đổi tên."
    if not new_name or not new_name.strip():
        return "Vui lòng nhập tên mới hợp lệ."
    new_name = new_name.strip()
    if old_name == new_name:
        return "Tên mới trùng với tên cũ."
    try:
        renamed_count = 0
        for ext in [".wav", ".mp3", ".m4a", ".json", ".txt"]:
            old_p = os.path.join(VOICES_DIR, f"{old_name}{ext}")
            new_p = os.path.join(VOICES_DIR, f"{new_name}{ext}")
            if os.path.exists(old_p):
                shutil.move(old_p, new_p)
                renamed_count += 1
        if renamed_count > 0:
            return f"Đã đổi tên giọng từ '{old_name}' thành '{new_name}' thành công!"
        else:
            return f"Không tìm thấy file nào của giọng '{old_name}' để đổi tên."
    except Exception as e:
        return f"Lỗi khi đổi tên giọng: {str(e)}"


def trim_10s_audio(audio_path, start_time):
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ Lỗi: Vui lòng chọn hoặc tải lên tệp âm thanh nguồn trước."
    try:
        start_t = float(start_time or 0)
        end_t = start_t + 10.0
        
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(TEMP_WORK_DIR, f"10s_{base}_{int(start_t)}s.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_t),
            "-to", str(end_t),
            "-i", audio_path,
            "-ar", "24000",
            "-ac", "1",
            out_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None, f"❌ Lỗi cắt audio bằng FFmpeg: {res.stderr}"
            
        return out_path, f"🎉 ĐÃ CẮT THÀNH CÔNG KHUNG CỨNG 10.0 GIÂY!\n⏱️ Phân đoạn: {start_t:.1f}s đến {end_t:.1f}s\n📁 Tệp: {out_path}"
    except Exception as e:
        return None, f"❌ Lỗi cắt audio: {str(e)}"


def clean_voice_audio(audio_path, do_demucs, do_denoise):
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ Lỗi: Vui lòng chọn hoặc tải lên tệp âm thanh nguồn trước."
    try:
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
        curr_audio = audio_path
        temp_dir = os.path.join(TEMP_WORK_DIR, "temp_clean_work")
        os.makedirs(temp_dir, exist_ok=True)
        
        logs = []
        
        # Step 1: Demucs Vocal Separation
        if do_demucs:
            logs.append("🤖 Đang dùng AI Demucs bóc tách loại bỏ 100% nhạc nền & nhạc cụ...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cmd_demucs = [
                sys.executable, "-m", "demucs.separate",
                "-d", device,
                "--two-stems", "vocals",
                "-n", "htdemucs",
                "-o", temp_dir,
                "--filename", "{stem}.{ext}",
                audio_path
            ]
            res = subprocess.run(cmd_demucs, capture_output=True, text=True)
            if res.returncode == 0:
                vocals_found = None
                for root, dirs, files in os.walk(temp_dir):
                    for f in files:
                        if f.startswith("vocals"):
                            vocals_found = os.path.join(root, f)
                            break
                if vocals_found and os.path.exists(vocals_found):
                    curr_audio = vocals_found
                    logs.append("✅ Đã tách giọng hát thành công (loại bỏ hoàn toàn nhạc nền).")
                else:
                    logs.append("⚠️ Không tìm thấy tệp vocals sau tách, tiếp tục lọc noise với audio gốc.")
            else:
                logs.append(f"⚠️ Demucs thông báo: {res.stderr[:120]}... Tiếp tục lọc noise.")

        # Step 2: FFmpeg Spectral Denoise & Filter
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_clean = os.path.join(TEMP_WORK_DIR, f"giong_sach_{base}.wav")
            
        if do_denoise:
            logs.append("🧹 Đang triệt tiêu tiếng rè microphone + lọc nhiễu tĩnh FFT + cắt tần số ù/rít...")
            filter_str = "afftdn=nf=-25,highpass=f=80,lowpass=f=12000,loudnorm=I=-16:TP=-1.5:LRA=11"
            cmd_ffmpeg = [
                "ffmpeg", "-y",
                "-i", curr_audio,
                "-af", filter_str,
                "-ar", "24000",
                "-ac", "1",
                out_clean
            ]
            res_ff = subprocess.run(cmd_ffmpeg, capture_output=True, text=True)
            if res_ff.returncode != 0:
                cmd_ffmpeg[4] = "afftdn=nf=-25,highpass=f=80,lowpass=f=12000"
                subprocess.run(cmd_ffmpeg, capture_output=True, text=True)
            logs.append("✅ Đã lọc noise & chuẩn hóa âm lượng giọng nói thành công.")
        else:
            cmd_ffmpeg = [
                "ffmpeg", "-y",
                "-i", curr_audio,
                "-ar", "24000",
                "-ac", "1",
                out_clean
            ]
            subprocess.run(cmd_ffmpeg, capture_output=True, text=True)

        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logs.append("🎉 ĐÃ TÁCH GIỌNG & LỌC NOISE HOÀN TẤT CHUẨN VOICE CLONE!")
        return out_clean, "\n".join(logs)
        
    except Exception as e:
        return None, f"❌ Lỗi xử lý làm sạch giọng: {str(e)}"


def extract_audio_from_video(video_path, output_format="wav", sample_rate=24000):
    if not video_path or not os.path.exists(video_path):
        return None, "❌ Lỗi: Vui lòng tải lên hoặc chọn tệp video nguồn trước."
    try:
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        fmt = (output_format or "wav").lower()
        out_path = os.path.join(TEMP_WORK_DIR, f"extracted_{base}.{fmt}")
        
        sr = str(int(sample_rate or 24000))
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ar", sr,
            "-ac", "1",
            out_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None, f"❌ Lỗi trích xuất âm thanh bằng FFmpeg: {res.stderr}"
            
        return out_path, f"🎉 ĐÃ TRÍCH XUẤT ÂM THANH THÀNH CÔNG!\n📄 Định dạng: .{fmt.upper()} ({sr}Hz, Mono)\n📁 Tệp: {out_path}"
    except Exception as e:
        return None, f"❌ Lỗi trích xuất âm thanh: {str(e)}"


# Language list is defined above


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice Voice Cloning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-base",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-base).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        transcribed_text = ref_text
        try:
            if not text or not text.strip():
                return None, "Vui lòng nhập văn bản cần chuyển sang giọng nói.", gr.update()

            gen_config = OmniVoiceGenerationConfig(
                num_step=int(num_step or 16),
                guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
                denoise=bool(denoise) if denoise is not None else True,
                preprocess_prompt=bool(preprocess_prompt),
                postprocess_output=bool(postprocess_output),
            )

            if language and language != "Tự động":
                lang = _LANG_MAP_VI_TO_EN.get(language)
            else:
                lang = None

            kw: Dict[str, Any] = dict(
                text=text.strip(), language=lang, generation_config=gen_config
            )

            if speed is not None and float(speed) != 1.0:
                kw["speed"] = float(speed)
            if duration is not None and float(duration) > 0:
                kw["duration"] = float(duration)

            if mode == "clone":
                if not ref_audio:
                    return None, "Vui lòng tải lên âm thanh tham chiếu.", gr.update()
                
                # If ref_text is empty, pass None to trigger ASR
                ref_txt_arg = ref_text.strip() if (ref_text and ref_text.strip()) else None
                prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_txt_arg,
                )
                kw["voice_clone_prompt"] = prompt
                transcribed_text = prompt.ref_text

            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

            audio = model.generate(**kw)
            waveform = (audio[0] * 32767).astype(np.int16)
            return (sampling_rate, waveform), "Hoàn thành.", transcribed_text if (mode == "clone") else gr.update()
        except Exception as e:
            logging.exception("Error during audio generation")
            return None, f"Lỗi: {type(e).__name__}: {e}", gr.update()

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    if generate_fn is not None:
        _gen = generate_fn
    else:
        try:
            import spaces
            _gen = spaces.GPU(duration=60)(_gen_core)
        except Exception:
            _gen = _gen_core

    theme = gr.themes.Default(
        primary_hue="green",
        secondary_hue="green",
        font=["Inter", "Arial", "sans-serif"],
    )
    
    css = """
    html, body {
        min-height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #f4f7f5 !important;
    }
    .gradio-container {
        background-color: #f4f7f5 !important;
        color: #1e293b !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        max-width: 100% !important;
        padding: 10px 14px !important;
        box-sizing: border-box !important;
    }
    
    /* Enforce non-wrapping text on ALL buttons so text NEVER breaks onto a second line */
    button, .btn-sidebar, .btn-studio-generate, .btn-studio-secondary, .gr-button {
        white-space: nowrap !important;
        word-break: keep-all !important;
        flex-shrink: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .studio-card {
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 12px rgba(11, 110, 79, 0.03) !important;
        margin-bottom: 12px !important;
        box-sizing: border-box !important;
    }
    .studio-title {
        font-size: 0.9em !important;
        font-weight: 700 !important;
        color: #0B6E4F !important;
        margin-bottom: 10px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #e6f1ed;
        padding-bottom: 6px;
    }
    .studio-badge {
        background: #e6f1ed !important;
        color: #0B6E4F !important;
        border-radius: 6px !important;
        width: 20px !important;
        height: 20px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.85em !important;
        font-weight: 800 !important;
    }
    
    /* Remove background from labels/titles in Gradio components */
    .block span, label span, label, .gr-label, .focusable, .block-label, div.block-label, span.focusable, .gr-label-container {
        background: transparent !important;
        background-color: transparent !important;
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.88em !important;
        border: none !important;
        margin-bottom: 3px !important;
    }
    
    /* Input styling */
    input[type="text"], input[type="number"], textarea, select, .gr-dropdown {
        border-color: #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #1f2937 !important;
        padding: 6px 10px !important;
        font-size: 0.88em !important;
    }
    input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus, .gr-dropdown:focus {
        border-color: #0B6E4F !important;
        box-shadow: 0 0 0 2px rgba(11, 110, 79, 0.1) !important;
    }

    .btn-studio-generate {
        background-color: #0B6E4F !important;
        color: #ffffff !important;
        border: 1px solid #0B6E4F !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
        padding: 10px 18px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(11, 110, 79, 0.15) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .btn-studio-generate:hover {
        background-color: #08543c !important;
        border-color: #08543c !important;
    }
    .btn-studio-secondary {
        background-color: #e6f1ed !important;
        color: #0B6E4F !important;
        border: 1px solid #d1e7dd !important;
        font-weight: 600 !important;
        font-size: 0.88em !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    .btn-studio-secondary:hover {
        background-color: #d1e7dd !important;
    }
    
    /* Console log styling */
    .console-log textarea {
        background-color: #f9fafb !important;
        color: #374151 !important;
        font-family: ui-monospace, SFMono-Regular, SF Pro Text, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.82em !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
    }
 
    /* Hide Gradio footer */
    footer {
        display: none !important;
    }
 
    /* Checkbox styling */
    input[type="checkbox"] {
        appearance: checkbox !important;
        -webkit-appearance: checkbox !important;
        width: 15px !important;
        height: 15px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        border-radius: 4px !important;
        cursor: pointer !important;
    }
    input[type="checkbox"]:checked {
        accent-color: #0B6E4F !important;
    }
 
    /* Sidebar layout & Collapsible behavior */
    .sidebar-container {
        border-right: 1px solid #e2e8f0 !important;
        padding: 14px 10px !important;
        background-color: #ffffff !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 6px rgba(11, 110, 79, 0.03) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .sidebar-container.collapsed {
        min-width: 68px !important;
        max-width: 68px !important;
        padding: 14px 6px !important;
    }
    .sidebar-header {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 14px !important;
        padding: 0 4px !important;
    }
    .sidebar-container.collapsed .sidebar-header {
        justify-content: center !important;
        margin-bottom: 18px !important;
    }
    .sidebar-brand {
        font-size: 1.0em !important;
        font-weight: 800 !important;
        color: #0B6E4F !important;
        letter-spacing: -0.5px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
    }
    .sidebar-container.collapsed .sidebar-brand {
        display: none !important;
    }
    .btn-toggle-sidebar {
        background: #e6f1ed !important;
        color: #0B6E4F !important;
        border: 1px solid #d1e7dd !important;
        border-radius: 8px !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.05em !important;
        font-weight: bold !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    .btn-toggle-sidebar:hover {
        background: #0B6E4F !important;
        color: #ffffff !important;
    }
    .menu-group-label {
        font-size: 0.68em !important;
        font-weight: 700 !important;
        color: #9ca3af !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin: 14px 0 6px 6px !important;
        white-space: nowrap !important;
    }
    .sidebar-container.collapsed .menu-group-label {
        display: none !important;
    }
    
    .btn-sidebar {
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.88em !important;
        border-radius: 10px !important;
        padding: 9px 12px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        margin-bottom: 5px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        flex-shrink: 0 !important;
    }
    /* Collapsed sidebar buttons: Hide button text completely and show clean icon */
    .sidebar-container.collapsed .btn-sidebar {
        font-size: 0 !important;
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        padding: 0 !important;
        margin: 0 auto 6px auto !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
        white-space: nowrap !important;
    }
    .sidebar-container.collapsed #btn-cloner::after { content: "⚡"; font-size: 1.2rem !important; }
    .sidebar-container.collapsed #btn-library::after { content: "📂"; font-size: 1.2rem !important; }
    .sidebar-container.collapsed #btn-extractor::after { content: "🎥"; font-size: 1.2rem !important; }
    .sidebar-container.collapsed #btn-cleaner::after { content: "🧹"; font-size: 1.2rem !important; }
    .sidebar-container.collapsed #btn-trimmer::after { content: "✂️"; font-size: 1.2rem !important; }

    .btn-sidebar.primary-btn {
        background-color: #0B6E4F !important;
        color: #ffffff !important;
        box-shadow: 0 2px 5px rgba(11, 110, 79, 0.15) !important;
    }
    .btn-sidebar.secondary-btn {
        background-color: transparent !important;
        color: #4b5563 !important;
    }
    .btn-sidebar.secondary-btn:hover {
        background-color: #e6f1ed !important;
        color: #0B6E4F !important;
    }
    
    /* Voice list item styling */
    .voice-item-row {
        border-bottom: 1px solid #e6f1ed !important;
        padding: 10px 0 !important;
        align-items: center !important;
    }
    .compact-audio-preview audio {
        height: 35px !important;
    }
    """

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Studio - AI Voice Cloning") as demo:
        library_trigger = gr.State(value=0)
        # Main layout: Sidebar on the left, Content on the right
        with gr.Row():
            # SIDEBAR COLUMN (Left)
            with gr.Column(scale=2, min_width=180, elem_classes="sidebar-container"):
                gr.HTML("""
                <div class="sidebar-header">
                    <div class="sidebar-brand">🎙️ OMNIVOICE STUDIO</div>
                    <button type="button" id="btn-toggle-sidebar" class="btn-toggle-sidebar" title="Thu gọn Sidebar" onclick="(function(btn){
                        var sb = document.querySelector('.sidebar-container');
                        if(sb) {
                            var isCol = sb.classList.toggle('collapsed');
                            btn.innerHTML = isCol ? '▶' : '◀';
                            btn.title = isCol ? 'Mở rộng Sidebar' : 'Thu gọn Sidebar';
                        }
                    })(this)">◀</button>
                </div>
                <div class="menu-group-label">🎙️ GIỌNG NÓI</div>
                """)
                btn_tab_cloner = gr.Button("⚡ Nhân bản Giọng", elem_id="btn-cloner", elem_classes="btn-sidebar primary-btn")
                btn_tab_library = gr.Button("📂 Quản lý Giọng", elem_id="btn-library", elem_classes="btn-sidebar secondary-btn")
                
                gr.HTML("""<div class="menu-group-label">🛠️ BỘ CÔNG CỤ ÂM THANH</div>""")
                btn_tab_extractor = gr.Button("🎥 Trích Âm Thanh MP4", elem_id="btn-extractor", elem_classes="btn-sidebar secondary-btn")
                btn_tab_cleaner = gr.Button("🧹 Lọc Noise & Tách Giọng", elem_id="btn-cleaner", elem_classes="btn-sidebar secondary-btn")
                btn_tab_trimmer = gr.Button("✂️ Cắt mẫu 10s Giọng", elem_id="btn-trimmer", elem_classes="btn-sidebar secondary-btn")
                
            # CONTENT COLUMN (Right)
            with gr.Column(scale=10):
                
                # Tab 2: Nhân bản Giọng nói (Cloner) - Defined first so vc_preset is defined before Tab 1 references it!
                with gr.Column(visible=True) as col_tab_cloner:
                    with gr.Row():
                        # Cột 1: GIỌNG NÓI THAM CHIẾU (Reference Audio)
                        with gr.Column(scale=4, elem_classes="studio-card"):
                            gr.HTML(
                                """
                                <div class="studio-title">
                                    <span class="studio-badge">1</span>
                                    <span>Giọng Nói Tham Chiếu</span>
                                </div>
                                """
                            )
                            
                            saved_list = ["Tải file mới / Chưa chọn"] + get_saved_voices()
                            vc_preset = gr.Dropdown(
                                label="📂 Chọn giọng đã lưu (Presets)",
                                choices=saved_list,
                                value="Tải file mới / Chưa chọn",
                                interactive=True,
                            )
                            
                            with gr.Row():
                                vc_save_name = gr.Textbox(
                                    label="Tên để lưu giọng mới",
                                    placeholder="Ví dụ: Giọng Chí Nhân",
                                    lines=1,
                                    scale=3,
                                )
                                vc_save_btn = gr.Button(
                                    "💾 Lưu giọng",
                                    elem_classes="btn-studio-secondary",
                                    scale=1,
                                )
                            
                            vc_ref_audio = gr.Audio(
                                label="Tải lên tệp âm thanh ghi âm mẫu (.wav, .mp3, .m4a...)",
                                type="filepath",
                            )
                            
                            vc_ref_text = gr.Textbox(
                                label="Văn bản giọng nói mẫu (Nhập để bỏ qua ASR và tăng tốc 10x)",
                                placeholder="Nhập nội dung giọng nói mẫu...",
                                lines=1,
                            )
                            
                        # Cột 2: KỊCH BẢN & BỘ TRỘN (Script & Audio Mixer)
                        with gr.Column(scale=4, elem_classes="studio-card"):
                            gr.HTML(
                                """
                                <div class="studio-title">
                                    <span class="studio-badge">2</span>
                                    <span>Kịch Bản & Bộ Trộn (Mixer)</span>
                                </div>
                                """
                            )
                            
                            vc_text = gr.Textbox(
                                label="Nội dung văn bản muốn nhân bản nói",
                                placeholder="Nhập văn bản cần đọc tại đây...",
                                lines=5,
                            )
                            
                            vc_lang = gr.Dropdown(
                                label="Ngôn ngữ mục tiêu (Target Language)",
                                choices=_ALL_LANGUAGES,
                                value="Tự động",
                                allow_custom_value=False,
                                interactive=True,
                            )
                            
                            vc_sp = gr.Slider(
                                0.5,
                                1.5,
                                value=1.0,
                                step=0.05,
                                label="Tốc độ nói (Speed)",
                            )
                            
                            with gr.Accordion("🎛️ Bảng Điều Khiển Tần Số (Style Adjustments)", open=False):
                                vc_du = gr.Number(
                                    value=None,
                                    label="Độ dài cố định (Duration - giây)",
                                )
                                vc_ns = gr.Slider(
                                    4,
                                    64,
                                    value=16,
                                    step=1,
                                    label="Số bước suy luận (Steps)",
                                )
                                vc_gs = gr.Slider(
                                    0.0,
                                    4.0,
                                    value=2.0,
                                    step=0.1,
                                    label="Thang đo hướng dẫn (CFG)",
                                )
                                vc_dn = gr.Checkbox(
                                    label="Khử nhiễu nền (Denoise)",
                                    value=True,
                                )
                                vc_pp = gr.Checkbox(
                                    label="Tiền xử lý giọng mẫu",
                                    value=True,
                                )
                                vc_po = gr.Checkbox(
                                    label="Hậu xử lý kết quả",
                                    value=True,
                                )

                        # Cột 3: KẾT QUẢ & GIÁM SÁT (Output & Monitor)
                        with gr.Column(scale=4, elem_classes="studio-card"):
                            gr.HTML(
                                """
                                <div class="studio-title">
                                    <span class="studio-badge">3</span>
                                    <span>Đầu Ra & Giám Sát</span>
                                </div>
                                """
                            )
                            
                            vc_btn = gr.Button("⚡ KÍCH HOẠT NHÂN BẢN GIỌNG NÓI", elem_classes="btn-studio-generate")
                            
                            vc_status = gr.Textbox(
                                label="🖥️ Nhật ký giám sát",
                                value="Hệ thống sẵn sàng...",
                                interactive=False,
                                lines=2,
                                elem_classes="console-log",
                            )
                            
                            vc_audio = gr.Audio(
                                label="🎵 Âm thanh kết quả (Studio Waveform)",
                                type="numpy",
                            )
                
                # Tab Trimmer: Cắt mẫu 10s Giọng
                with gr.Column(visible=False) as col_tab_trimmer:
                    gr.HTML("<h2 style='font-size:1.3em;font-weight:700;color:#0B6E4F;margin-bottom:15px;padding-bottom:8px;border-bottom:1px solid #e6f1ed;'>✂️ Cắt Phân Đoạn Mẫu 10 Giây Khung Cứng cho Voice Clone</h2>")
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes="studio-card"):
                            trim_file = gr.Audio(
                                label="Tải lên hoặc chọn tệp âm thanh nguồn cần cắt",
                                type="filepath",
                            )
                            trim_start = gr.Slider(
                                minimum=0,
                                maximum=300,
                                value=0,
                                step=0.1,
                                label="Thời điểm bắt đầu cắt (Giây)",
                                info="Vùng cắt sẽ tự động cố định đúng 10.0 giây (Từ Start ➡️ Start + 10s)"
                            )
                            trim_info = gr.Textbox(
                                label="📌 Khung cắt cố định",
                                value="Cố định: 0.0s ➡️ 10.0s (Tổng thời lượng: 10.0 giây)",
                                interactive=False,
                            )
                            trim_btn = gr.Button("✂️ CẮT NGAY MẪU 10s GIỌNG", elem_classes="btn-studio-generate")

                        with gr.Column(scale=6, elem_classes="studio-card"):
                            trim_status = gr.Textbox(
                                label="🖥️ Trạng thái xử lý",
                                value="Sẵn sàng...",
                                lines=3,
                                elem_classes="console-log",
                            )
                            trim_output = gr.Audio(
                                label="🎵 Kết quả đoạn 10s đã cắt",
                                type="filepath",
                            )
                            with gr.Row():
                                trim_use_cloner_btn = gr.Button("📋 Đưa vào Giọng Tham Chiếu Cloner", elem_classes="btn-studio-secondary")

                # Tab Cleaner: Lọc Noise & Tách Giọng Sạch
                with gr.Column(visible=False) as col_tab_cleaner:
                    gr.HTML("<h2 style='font-size:1.3em;font-weight:700;color:#0B6E4F;margin-bottom:15px;padding-bottom:8px;border-bottom:1px solid #e6f1ed;'>🧹 Lọc Noise & Tách Giọng Sạch (Vocal Cleaner & Noise Reduction)</h2>")
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes="studio-card"):
                            clean_file = gr.Audio(
                                label="Tải lên hoặc chọn tệp âm thanh nguồn bị lẫn nhạc / nhiễu",
                                type="filepath",
                            )
                            clean_do_demucs = gr.Checkbox(
                                label="AI Demucs: Bóc tách loại bỏ 100% Nhạc nền & Nhạc cụ",
                                value=True,
                            )
                            clean_do_denoise = gr.Checkbox(
                                label="FFmpeg DSP: Khử nhiễu micro, triệt tiếng ù rè & tạp âm môi trường",
                                value=True,
                            )
                            clean_btn = gr.Button("🪄 BẮT ĐẦU TÁCH GIỌNG & LỌC NOISE SẠCH", elem_classes="btn-studio-generate")

                        with gr.Column(scale=6, elem_classes="studio-card"):
                            clean_status = gr.Textbox(
                                label="🖥️ Nhật ký làm sạch giọng",
                                value="Sẵn sàng...",
                                lines=5,
                                elem_classes="console-log",
                            )
                            clean_output = gr.Audio(
                                label="🎵 Bản thu giọng đọc sạch sau khi lọc",
                                type="filepath",
                            )
                            with gr.Row():
                                clean_use_cloner_btn = gr.Button("📋 Đưa vào Giọng Tham Chiếu Cloner", elem_classes="btn-studio-secondary")

                # Tab Extractor: Trích Âm Thanh Từ Video MP4
                with gr.Column(visible=False) as col_tab_extractor:
                    gr.HTML("<h2 style='font-size:1.3em;font-weight:700;color:#0B6E4F;margin-bottom:15px;padding-bottom:8px;border-bottom:1px solid #e6f1ed;'>🎥 Trích Xuất Âm Thanh Từ Video (MP4 / MKV / MOV ➡️ Audio)</h2>")
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes="studio-card"):
                            extract_file = gr.File(
                                label="1. Tải lên tệp Video nguồn (.mp4, .mkv, .mov, .avi, .webm...)",
                                file_types=[".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"],
                            )
                            extract_fmt = gr.Radio(
                                choices=["wav", "mp3", "m4a", "flac"],
                                value="wav",
                                label="2. Chọn định dạng âm thanh xuất ra",
                            )
                            extract_sr = gr.Slider(
                                minimum=8000,
                                maximum=48000,
                                value=24000,
                                step=4000,
                                label="3. Tần số lấy mẫu (Sample Rate - Hz)",
                                info="Mô hình OmniVoice khuyến nghị chuẩn 24000Hz"
                            )
                            extract_btn = gr.Button("🎥 TRÍCH XUẤT ÂM THANH NGAY", elem_classes="btn-studio-generate")

                        with gr.Column(scale=6, elem_classes="studio-card"):
                            extract_status = gr.Textbox(
                                label="🖥️ Trạng thái trích xuất",
                                value="Sẵn sàng...",
                                lines=3,
                                elem_classes="console-log",
                            )
                            extract_output = gr.Audio(
                                label="🎵 Âm thanh đã trích xuất từ Video",
                                type="filepath",
                            )
                            with gr.Row():
                                extract_use_cloner_btn = gr.Button("📋 Đưa vào Giọng Cloner", elem_classes="btn-studio-secondary")
                                extract_use_trimmer_btn = gr.Button("✂️ Đưa sang Cắt 10s Giọng", elem_classes="btn-studio-secondary")
                                extract_use_cleaner_btn = gr.Button("🧹 Đưa sang Lọc Noise", elem_classes="btn-studio-secondary")

                # Tab 1: Thư viện Giọng nói (Library) - Defined second but visible=False by default
                with gr.Column(visible=False) as col_tab_library:
                    gr.HTML("<h2 style='font-size:1.3em;font-weight:700;color:#0B6E4F;margin-bottom:15px;padding-bottom:8px;border-bottom:1px solid #e6f1ed;'>📂 Quản lý Giọng đã lưu</h2>")
                    
                    @gr.render(inputs=[vc_preset, library_trigger])
                    def render_voices(preset_val, trigger_val):
                        voices = get_saved_voices()
                        if not voices:
                            gr.Markdown("### Chưa có giọng nói nào được lưu.\n\nHãy chuyển sang mục **⚡ Nhân bản Giọng** để bắt đầu tạo mới!")
                            return
                        
                        for voice in voices:
                            audio_path = None
                            for ext in [".wav", ".mp3", ".m4a"]:
                                p = os.path.join(VOICES_DIR, f"{voice}{ext}")
                                if os.path.exists(p):
                                    audio_path = p
                                    break
                            
                            # Read metadata if exists to show info
                            meta_info = ""
                            json_path = os.path.join(VOICES_DIR, f"{voice}.json")
                            if os.path.exists(json_path):
                                try:
                                    with open(json_path, "r", encoding="utf-8") as f:
                                        data = json.load(f)
                                        lang = data.get("language", "Tự động")
                                        sp = data.get("speed", 1.0)
                                        ns = data.get("steps", 16)
                                        meta_info = f"Ngôn ngữ: {lang} | Tốc độ: {sp}x | Số bước: {ns}"
                                except Exception:
                                    pass

                            with gr.Row(elem_classes="voice-item-row"):
                                with gr.Column(scale=4):
                                    gr.HTML(f"<div style='font-weight:700;color:#374151;font-size:1.05em;'>{voice}</div>"
                                            f"<div style='font-size:0.8em;color:#6b7280;margin-top:2px;'>{meta_info}</div>")
                                
                                with gr.Column(scale=5):
                                    if audio_path:
                                        gr.Audio(value=audio_path, interactive=False, show_label=False, container=False, elem_classes="compact-audio-preview")
                                    else:
                                        gr.HTML("<span style='color:#9ca3af;font-size:0.9em;'>Không tìm thấy âm thanh</span>")
                                
                                with gr.Column(scale=3):
                                    with gr.Row():
                                        use_btn = gr.Button("Sử dụng", elem_classes="btn-studio-secondary", size="sm")
                                        rename_btn = gr.Button("Đổi tên", variant="secondary", size="sm")
                                        delete_btn = gr.Button("Xóa", variant="stop", size="sm")
                                    
                                    # Rename Row (hidden by default)
                                    with gr.Row(visible=False) as rename_row:
                                        new_name_input = gr.Textbox(placeholder="Tên mới...", show_label=False, container=False, scale=2)
                                        confirm_btn = gr.Button("✓", variant="primary", size="sm", scale=1)
                                        cancel_btn = gr.Button("✗", variant="secondary", size="sm", scale=1)
                                    
                                    # Callback when use button is clicked
                                    def _use_voice_fn(v=voice):
                                        return (
                                            gr.update(visible=False),   # col_tab_library
                                            gr.update(visible=True),    # col_tab_cloner
                                            gr.update(visible=False),   # col_tab_trimmer
                                            gr.update(visible=False),   # col_tab_cleaner
                                            gr.update(visible=False),   # col_tab_extractor
                                            gr.update(elem_classes="btn-sidebar secondary-btn"), # btn_tab_library
                                            gr.update(elem_classes="btn-sidebar primary-btn"),   # btn_tab_cloner
                                            gr.update(elem_classes="btn-sidebar secondary-btn"), # btn_tab_trimmer
                                            gr.update(elem_classes="btn-sidebar secondary-btn"), # btn_tab_cleaner
                                            gr.update(elem_classes="btn-sidebar secondary-btn"), # btn_tab_extractor
                                            v                           # select the preset in dropdown
                                        )
                                    
                                    # Callback when delete button is clicked
                                    def _delete_voice_fn(trig, v=voice):
                                        status = delete_voice_preset(v)
                                        new_choices = ["Tải file mới / Chưa chọn"] + get_saved_voices()
                                        return (
                                            gr.Dropdown(choices=new_choices, value="Tải file mới / Chưa chọn"),
                                            status,
                                            trig + 1,
                                        )
                                    
                                    # Callback when rename is confirmed
                                    def _confirm_rename_fn(trig, new_val, v=voice):
                                        status = rename_voice_preset(v, new_val)
                                        new_choices = ["Tải file mới / Chưa chọn"] + get_saved_voices()
                                        return (
                                            gr.Dropdown(choices=new_choices, value="Tải file mới / Chưa chọn"),
                                            status,
                                            trig + 1,
                                        )
                                    
                                    use_btn.click(
                                        _use_voice_fn,
                                        inputs=[],
                                        outputs=[col_tab_library, col_tab_cloner, col_tab_trimmer, col_tab_cleaner, col_tab_extractor, btn_tab_library, btn_tab_cloner, btn_tab_trimmer, btn_tab_cleaner, btn_tab_extractor, vc_preset],
                                    )

                                    delete_btn.click(
                                        _delete_voice_fn,
                                        inputs=[library_trigger],
                                        outputs=[vc_preset, vc_status, library_trigger],
                                    )

                                    rename_btn.click(
                                        lambda: gr.update(visible=True),
                                        inputs=[],
                                        outputs=[rename_row],
                                    )

                                    cancel_btn.click(
                                        lambda: gr.update(visible=False),
                                        inputs=[],
                                        outputs=[rename_row],
                                    )

                                    confirm_btn.click(
                                        _confirm_rename_fn,
                                        inputs=[library_trigger, new_name_input],
                                        outputs=[vc_preset, vc_status, library_trigger],
                                    )

        # Wire tab switching buttons (5 tabs)
        def _show_cloner():
            return (
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(elem_classes="btn-sidebar primary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn")
            )

        def _show_trimmer():
            return (
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar primary-btn"),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn")
            )

        def _show_cleaner():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"),
                gr.update(elem_classes="btn-sidebar primary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn")
            )

        def _show_extractor():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar primary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn")
            )

        def _show_library():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"),
                gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar secondary-btn"), gr.update(elem_classes="btn-sidebar primary-btn")
            )

        all_cols_and_btns = [
            col_tab_cloner, col_tab_trimmer, col_tab_cleaner, col_tab_extractor, col_tab_library,
            btn_tab_cloner, btn_tab_trimmer, btn_tab_cleaner, btn_tab_extractor, btn_tab_library
        ]

        btn_tab_cloner.click(_show_cloner, inputs=[], outputs=all_cols_and_btns)
        btn_tab_trimmer.click(_show_trimmer, inputs=[], outputs=all_cols_and_btns)
        btn_tab_cleaner.click(_show_cleaner, inputs=[], outputs=all_cols_and_btns)
        btn_tab_extractor.click(_show_extractor, inputs=[], outputs=all_cols_and_btns)
        btn_tab_library.click(_show_library, inputs=[], outputs=all_cols_and_btns)

        # Wire Trimmer tab events
        def _update_trim_info(start_val):
            st = float(start_val or 0)
            return f"Cố định: {st:.1f}s ➡️ {st + 10.0:.1f}s (Tổng thời lượng: 10.0 giây)"

        trim_start.change(_update_trim_info, inputs=[trim_start], outputs=[trim_info])
        trim_btn.click(trim_10s_audio, inputs=[trim_file, trim_start], outputs=[trim_output, trim_status])

        def _use_trimmed_in_cloner(aud):
            cloner_updates = _show_cloner()
            return cloner_updates + (aud,)

        trim_use_cloner_btn.click(
            _use_trimmed_in_cloner,
            inputs=[trim_output],
            outputs=all_cols_and_btns + [vc_ref_audio]
        )

        # Wire Cleaner tab events
        clean_btn.click(clean_voice_audio, inputs=[clean_file, clean_do_demucs, clean_do_denoise], outputs=[clean_output, clean_status])

        def _use_cleaned_in_cloner(aud):
            cloner_updates = _show_cloner()
            return cloner_updates + (aud,)

        clean_use_cloner_btn.click(
            _use_cleaned_in_cloner,
            inputs=[clean_output],
            outputs=all_cols_and_btns + [vc_ref_audio]
        )

        # Wire Extractor tab events
        extract_btn.click(
            extract_audio_from_video,
            inputs=[extract_file, extract_fmt, extract_sr],
            outputs=[extract_output, extract_status]
        )

        def _use_extracted_in_cloner(aud):
            return _show_cloner() + (aud,)

        extract_use_cloner_btn.click(
            _use_extracted_in_cloner,
            inputs=[extract_output],
            outputs=all_cols_and_btns + [vc_ref_audio]
        )

        def _use_extracted_in_trimmer(aud):
            return _show_trimmer() + (aud,)

        extract_use_trimmer_btn.click(
            _use_extracted_in_trimmer,
            inputs=[extract_output],
            outputs=all_cols_and_btns + [trim_file]
        )

        def _use_extracted_in_cleaner(aud):
            return _show_cleaner() + (aud,)

        extract_use_cleaner_btn.click(
            _use_extracted_in_cleaner,
            inputs=[extract_output],
            outputs=all_cols_and_btns + [clean_file]
        )

        def _clone_fn(
            text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po
        ):
            return _gen(
                text,
                lang,
                ref_aud,
                None, # instruct is always None in clone-only UI
                ns,
                gs,
                dn,
                sp,
                du,
                pp,
                po,
                mode="clone",
                ref_text=ref_text,
            )

        vc_btn.click(
            _clone_fn,
            inputs=[
                vc_text,
                vc_lang,
                vc_ref_audio,
                vc_ref_text,
                vc_ns,
                vc_gs,
                vc_dn,
                vc_sp,
                vc_du,
                vc_pp,
                vc_po,
            ],
            outputs=[vc_audio, vc_status, vc_ref_text],
        )

        def _save_preset_cb(trig, audio, name, ref_text, lang, sp, du, ns, gs, dn, pp, po):
            new_dropdown, status = save_voice_preset(
                audio, ref_text, name, lang, sp, du, ns, gs, dn, pp, po
            )
            return new_dropdown, status, trig + 1

        vc_save_btn.click(
            _save_preset_cb,
            inputs=[
                library_trigger,
                vc_ref_audio,
                vc_save_name,
                vc_ref_text,
                vc_lang,
                vc_sp,
                vc_du,
                vc_ns,
                vc_gs,
                vc_dn,
                vc_pp,
                vc_po,
            ],
            outputs=[vc_preset, vc_status, library_trigger],
        )

        def _load_preset_cb(name):
            audio, ref_text, language, speed, duration, steps, cfg_scale, denoise, preprocess, postprocess = load_voice_preset(name)
            return audio, ref_text, language, speed, duration, steps, cfg_scale, denoise, preprocess, postprocess

        vc_preset.change(
            _load_preset_cb,
            inputs=[vc_preset],
            outputs=[
                vc_ref_audio,
                vc_ref_text,
                vc_lang,
                vc_sp,
                vc_du,
                vc_ns,
                vc_gs,
                vc_dn,
                vc_pp,
                vc_po,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        allowed_paths=[VOICES_DIR, TEMP_WORK_DIR],
    )
    return 0


# Global variable for Gradio hot reloading
demo = None

if __name__ == "__main__":
    main()
elif os.getenv("GRADIO_WATCH_MODE") == "1":
    import os
    import sys
    device = os.getenv("DEVICE", "cuda")
    checkpoint = os.getenv("MODEL", _DEFAULT_MODEL)
    asr_model = os.getenv("ASR_MODEL", "openai/whisper-base")
    no_asr = os.getenv("NO_ASR", "0") == "1"
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")

    # Cache model in sys to survive Gradio CLI hot-reloads and avoid CUDA OOM
    if not hasattr(sys, "_omnivoice_model"):
        print(f"[Gradio Reload] Loading model {checkpoint} on {device} (first time)...")
        sys._omnivoice_model = OmniVoice.from_pretrained(
            checkpoint,
            device_map=device,
            dtype=torch.float16,
            load_asr=not no_asr,
            asr_model_name=asr_model,
        )
    else:
        print("[Gradio Reload] Reusing cached OmniVoice model.")

    model = sys._omnivoice_model
    demo = build_demo(model, checkpoint)
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        show_api=False,
        allowed_paths=[VOICES_DIR, TEMP_WORK_DIR],
    )
