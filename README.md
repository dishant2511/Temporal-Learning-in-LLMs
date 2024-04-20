# Temporal Learning in Large Language Models (LLM)

This github repo contains files which were used during the experimental part of the research:
`**Remember This Event That Year? Assessing Temporal Information and Reasoning in Large Language Models**
Himanshu Beniwal, Kowsik Nandagopan D, Mayank Singh`

## Project Overview

This project focuses on the challenges faced by Large Language Models (LLMs) in handling dynamic and temporal data. We aim to conduct a comprehensive analysis of temporal data in key categories, including energy and climate, food and agriculture, human rights, health, poverty, wars, migration, and innovation. Our analysis will focus on how this data is handled by currently available LLM models, which include mistral-instruct, llama-2, phi-2, Google’s Gemini-Pro, OpenAI’s GPT3.5 Turbo, and GPT4.

## Key Features

- Streamlining the inference process on the temporal dataset using pre-trained LLM models with vLLM.
- Fine-tuning models using a continual learning approach with year-wise prompts.
- Employing PEFT fine-tuning methods alongside QLoRA.
- Enhancing the factual knowledge base of various LLM models through advanced EasyEdit methodologies such as ROME, GRACE, MEND, among others.

## Files

- `inf_with_transformers.py`: This script can be used to get inference from LLMs using the transformers library.
- `inf_with_vLLM.py`: This script utilizes the vLLM library to get inference from LLMs.
- `ROME.py`, `GRACE.py`: These scripts contain the model editing methods using EasyEdit methodologies ROME and GRACE respectively.

## Results

The results of our research can be accessed from here.

## Acknowledgements

This research was conducted under the guidance of Prof. Mayank Singh at IIT Gandhinagar and was granted by Microsoft’s Accelerate Foundation Models Research grant.
