# Temporal Learning in Large Language Models (LLM)

This github repo contains files which were used during the experimental part of the research: **'Remember This Event That Year? Assessing Temporal Information and Reasoning in Large Language Models'**
by Himanshu Beniwal, Kowsik Nandagopan D, Mayank Singh. Repo also has files for the next phase of the research which is to edit the factual knowledge in LLMs and compare it with pre-trained models.

## Project Overview

Large Language Models (LLMs) play a crucial role in AI due to their wide-ranging uses and ability to tackle intricate challenges, propelling advancements and efficiency in various sectors. However, given the dynamic nature of details like the current champion in a sports league or the current CEO of the company, along with specific data that is only produced over time (like population statistics), lead to problems like Temporal model deterioration. The project was focused finding the challenges faced by Large Language Models (LLMs) in handling dynamic and temporal data. The aim was to conduct a comprehensive analysis of temporal data in key categories, including energy and climate, food and agriculture, human rights, health, poverty, wars, migration, and innovation. Our analysis involved focus on how this data is handled by currently available LLM models, which include mistral-instruct, llama-2, phi-2, Google’s Gemini-Pro, OpenAI’s GPT3.5 Turbo, and GPT4.

## Files

- `inf_with_transformers.py`: This script can be used to get inference from LLMs using the transformers library.
- `inf_with_vLLM.py`: This script utilizes the vLLM library to get inference from LLMs.
- `ROME.py`, `GRACE.py`: These scripts contain the model editing methods using EasyEdit methodologies ROME and GRACE respectively.

## Results

The results of our research can be accessed from research paper `arXiv:2402.11997 [cs.CL]`.

## Acknowledgements

This research was conducted under the guidance of Prof. Mayank Singh at IIT Gandhinagar and was granted by Microsoft’s Accelerate Foundation Models Research grant.

## References

Himanshu Beniwal, Kowsik Nandagopan D, Mayank Singh (2024). Remember This Event That Year? Assessing Temporal Information and Reasoning in Large Language Models. arXiv preprint arXiv:2402.11997 [cs.CL].
