# Model
MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'
#MODEL='meta-llama/Llama-2-7b-chat-hf'
# Batch size for inference
BATCH_SIZE = 5
# Location of the csv files
FOLDER_PATH = ''

#SUFFIX = '. Generate the text providing the only correct option.'
SUFFIX = '. Provide the only correct option, without explanation.'

# Seeding everything
def seed_everything(seed):
    import random, os
    import numpy as np
    import torchscre
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# Creating a model, tokenizer, and a pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Setting the padding token id
pipe.tokenizer.pad_token_id = model.config.eos_token_id

import os
import pandas as pd


for i in os.listdir(FOLDER_PATH):
    for j in os.listdir(os.path.join(FOLDER_PATH, i)):
        llm.seed = SEED
        path = os.path.join(FOLDER_PATH, i, j)
        print(f"-------Processing file: {os.path.join(i,j)}-------")
        prompts_df = pd.read_csv(path)
        prompts_df['modified_prompt'] = prompts_df['query'].apply(lambda x: x + SUFFIX)
        prompts = prompts_df['modified_prompt'].tolist()
        outputs = pipe(
            prompts,
            do_sample=True,
            max_new_tokens=50,
            temperature=0.2,
            top_k=60,
            top_p=0.96,
            num_return_sequences=1,
            batch_size=BATCH_SIZE
        )
        outputs = llm.generate(prompts, sampling_params)
        prompts_df['generated_text'] = [output.outputs[0].text for output in outputs]
        if not os.path.exists(os.path.join("results_mistral", FOLDER_PATH, i)):
            os.makedirs(os.path.join("results_mistral", FOLDER_PATH, i))
        prompts_df.to_csv(os.path.join("results_mistral", FOLDER_PATH, i, j))