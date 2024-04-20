
SEED = 1

MODEL='meta-llama/Llama-2-7b-chat-hf'

FOLDER_PATH = ''

# Defining a prompt suffix
SUFFIX = '. Generate the text providing the only correct option.'

import torch

torch.cuda.empty_cache()

## vLLM code
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, trust_remote_code=True, seed = SEED, dtype="float16", download_dir="weights")
sampling_params = SamplingParams(n=1, temperature=0.01, top_p=0.95, top_k=40, max_tokens=60)

import os
import pandas as pd

temperatures = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8]
top_p_values = [0.9, 0.92, 0.95, 0.96, 0.97]
top_k_values = [20, 40, 60, 80]

for temp in temperatures:
    for top_p in top_p_values:
        for top_k in top_k_values:
            sampling_params = SamplingParams(n=1, temperature=temp, top_p=top_p, top_k=top_k, max_tokens=60)

            for i in os.listdir(FOLDER_PATH):
                for j in os.listdir(os.path.join(FOLDER_PATH, i)):
                    llm.seed = SEED
                    path = os.path.join(FOLDER_PATH, i, j)
                    print(f"-------Processing file: {os.path.join(i, j)}-------")
                    prompts_df = pd.read_csv(path)
                    prompts_df['modified_prompt'] = prompts_df['query'].apply(lambda x: x + SUFFIX)
                    prompts = prompts_df['modified_prompt'].tolist()
                    
                    outputs = llm.generate(prompts, sampling_params)
                    prompts_df['generated_text'] = [output.outputs[0].text for output in outputs]
                    
                    result_folder = os.path.join("results_temp", FOLDER_PATH, i)
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    
                    result_file_name = f"{j}_temp_{temp}_top_p_{top_p}_top_k_{top_k}.csv"
                    result_file_path = os.path.join(result_folder, result_file_name)
                    prompts_df.to_csv(result_file_path)
