# sthenno-chatbot

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This framework is just currently a frontend used to provide a Human-AI interaction
environment. Endpoints are hosted using [vLLM](https://github.com/vllm-project/vllm).
For more details, please check [neoheartbeats-kernel](https://github.com/neoheartbeats/neoheartbeats-kernel).

## Current hosted models:

sthenno-gm-04 is a fine-tuned version of DeepMind's gemma2-9b-it.

This model is optimized by KTO(Kahneman-Tversky Optimization) using custom data.

This model is designed to output more naturally that to align human's preferences, but
NOT including to instruct the model to generate human-like outputs such as emotions.

One part of this design is to discover how LLMs implement mental models for
continual-learning and long-term memory's constructions.

Model's safetensors and training data have NOT been disclosed yet but planned to be by
publishing to platforms such as HuggingFace once reliable data is collected under
replicated evaluations.
