import torch
from easyeditor import BaseEditor
from easyeditor import GraceHyperParams

hparams=GraceHyperParams.from_hparams('./hparams/GRACE/gpt2-xl.yaml')

import pandas as pd

# Load the CSV file
df = pd.read_csv('first_100_rows.csv')

df['target_new'] = df['target_new'].astype(str)
prompts = df['prompts'].tolist()
target_new = df['target_new'].tolist()
subject = df['subject'].tolist()

editor=BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=None,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)

print(type(edited_model))

# Assuming edited_model is your model
edited_model.save_pretrained("")