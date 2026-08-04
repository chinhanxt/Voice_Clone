#!/usr/bin/env python3
import os
import re
import json
import shutil
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file, save_file
from omnivoice.models.omnivoice import OmniVoiceConfig

def prune():
    print("Starting backbone model pruning for English and Vietnamese...")
    
    # 1. Resolve HF snapshot path
    from huggingface_hub import snapshot_download
    print("Locating OmniVoice checkpoint snapshot...")
    resolved_path = snapshot_download("k2-fsa/OmniVoice")
    print(f"Snapshot path: {resolved_path}")
    
    # 2. Load tokenizer and get vocab
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # 3. Identify kept tokens
    vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    vietnamese_chars += vietnamese_chars.upper()
    allowed_pattern = re.compile(rf'^[\x00-\x7F{re.escape(vietnamese_chars)}]+$')
    
    kept_indices = []
    vocab_mapping = {} # old_id_str -> new_id
    
    new_idx = 0
    for token, old_idx in sorted_vocab:
        # Decode the token
        try:
            decoded = tokenizer.convert_tokens_to_string([token])
        except Exception:
            decoded = token
            
        is_special = token.startswith('<|') or token in tokenizer.all_special_tokens
        
        if is_special or allowed_pattern.match(decoded):
            kept_indices.append(old_idx)
            vocab_mapping[str(old_idx)] = new_idx
            new_idx += 1
            
    print(f"Original vocab size: {len(sorted_vocab)}")
    print(f"Pruned vocab size: {len(kept_indices)} ({len(kept_indices)/len(sorted_vocab)*100:.2f}% kept)")
    
    # Create pruned directory
    output_dir = "./pruned_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vocab mapping
    mapping_path = os.path.join(output_dir, "vocab_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(vocab_mapping, f, indent=2)
    print(f"Saved vocabulary mapping to {mapping_path}")
    
    # 4. Load state dict and prune llm.embed_tokens.weight
    print("Loading model weights (model.safetensors)...")
    safetensors_path = os.path.join(resolved_path, "model.safetensors")
    state_dict = load_file(safetensors_path)
    
    print("Pruning llm.embed_tokens.weight tensor...")
    old_embed = state_dict["llm.embed_tokens.weight"]
    # Prune rows using kept indices
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long)
    new_embed = old_embed[kept_tensor]
    
    state_dict["llm.embed_tokens.weight"] = new_embed
    print(f"New embedding shape: {new_embed.shape}")
    
    # Save pruned state dict
    pruned_safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, pruned_safetensors_path)
    print(f"Saved pruned weights to {pruned_safetensors_path}")
    
    # 5. Modify config.json
    print("Modifying model configuration...")
    config_path = os.path.join(resolved_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        
    config_data["vocab_size"] = len(kept_indices)
    config_data["llm_config"]["vocab_size"] = len(kept_indices)
    
    pruned_config_path = os.path.join(output_dir, "config.json")
    with open(pruned_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to {pruned_config_path}")
    
    # 6. Copy tokenizer config, jinja template, and audio_tokenizer folder
    print("Copying tokenizer and helper files...")
    for filename in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        src = os.path.join(resolved_path, filename)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, filename))
            
    # Copy audio_tokenizer folder
    src_audio_tok = os.path.join(resolved_path, "audio_tokenizer")
    dest_audio_tok = os.path.join(output_dir, "audio_tokenizer")
    if os.path.exists(src_audio_tok):
        if os.path.exists(dest_audio_tok):
            shutil.rmtree(dest_audio_tok)
        shutil.copytree(src_audio_tok, dest_audio_tok)
        
    print("\nBackbone pruning completed successfully!")
    print(f"Pruned model is stored in: {os.path.abspath(output_dir)}")
    print("You can run the demo using:")
    print(f"python omnivoice/cli/demo.py --model {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    prune()
