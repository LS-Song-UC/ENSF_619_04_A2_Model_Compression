# Assignment 2 – Version 1: Baseline (FP16)
# CodeLlama-7B-Instruct, prompt P-2, greedy decoding, 5 runs, CodeCarbon per run.
# Colab T4. Install once per session:  !pip install codecarbon

import os
import csv
import glob
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from codecarbon import OfflineEmissionsTracker

model_name = "codellama/CodeLlama-7b-Instruct-hf"
version = "baseline_fp16"
country = "USA"        # pinned grid intensity: identical for all four versions, whatever region Colab assigns
num_runs = 5
max_new_tokens = 300

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU found. Set the Colab runtime to T4.")
hardware = torch.cuda.get_device_name(0)

# Load only if this exact model + version isn't already in memory
if globals().get("loaded_config") != (model_name, version):
    if "model" in globals():
        raise RuntimeError("A different model/version is loaded. Restart the session first.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # device_map="cuda" streams weights straight to the GPU; loading to CPU first can exceed Colab's RAM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
    model.eval()
    loaded_config = (model_name, version)
else:
    print("Model already loaded, skipping")

# Model size = storage size of the weight files on disk (already cached by from_pretrained)
weights_dir = snapshot_download(model_name, allow_patterns=["*.safetensors"])
size_gb = sum(os.path.getsize(f) for f in glob.glob(os.path.join(weights_dir, "*.safetensors"))) / 1e9

print("Version:", version)
print("Hardware:", hardware)
print("Dtype:", model.dtype)
print(f"Model size on disk: {size_gb:.2f} GB")
print(f"GPU memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

Code = '''from typing import List
def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations
    on a bank account that starts with zero balance. Your task is to
    detect if at any point the balance of account falls below zero,
    and at that point function should return True.
    """
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False'''

prompt = ("Generate Google-style docstring for the following Python function: {code}. "
          "Only output the docstring and function code.").format(code=Code)   # P-2

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)
input_tokens = inputs["input_ids"].shape[1]

results_csv = f"results_{version}.csv"
outputs_file = f"docstrings_{version}.txt"

# Warm up once (untimed, untracked)
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)

outputs = []
with open(results_csv, "w", newline="") as f, open(outputs_file, "w") as out_f:
    writer = csv.writer(f)
    writer.writerow(["run", "input_tokens", "output_tokens", "latency_s",
                     "emissions_kg", "energy_kwh", "accuracy_PF"])

    for run in range(1, num_runs + 1):
        # Fresh tracker per run, wrapped around generation only
        tracker = OfflineEmissionsTracker(country_iso_code=country, measure_power_secs=1,
                                          save_to_file=False, log_level="error")
        tracker.start()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        emissions_kg = tracker.stop()
        energy_kwh = tracker.final_emissions_data.energy_consumed

        output_tokens = generated_ids.shape[1] - input_tokens
        text = tokenizer.decode(generated_ids[0][input_tokens:], skip_special_tokens=True)
        outputs.append(text)
        out_f.write(f"--- Run {run} ---\n{text}\n\n")

        writer.writerow([run, input_tokens, output_tokens, f"{latency:.3f}",
                         f"{emissions_kg:.4e}", f"{energy_kwh:.4e}", ""])
        f.flush()

        print(f"Run {run}: {latency:.3f}s | {output_tokens} tokens | "
              f"{emissions_kg:.4e} kg CO2eq | {energy_kwh:.4e} kWh")

print("Identical output across runs:", len(set(outputs)) == 1)
print(f"Docstrings saved to {outputs_file}")
print(f"Metrics saved to {results_csv}")
