# Jarvis-Py

## Description

Jarvis-Py is intended to be a very basic Python-based virtual assistant. So far, it is capable of question and answer based on contexts, indexing documents for semantic search and speech recognition. I want Jarvis-Py to be as minimalistic as possible so the use of external libraries has been minimized. Inferences of the LLM models has been implemented solely using numpy to minimize external dependencies and also as a helpful excercise to further deep my understanding of LLMs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/1554cN3wt0n/jarvis-py.git
   ```
2. Navigate to the project directory:
   ```bash
   cd jarvis-py
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites

To ensure the application runs correctly, you need to create a .env file in the root directory of the project. This file should contain the necessary environment variables definitions. Here is an example of what the .env file should look like:

```sh
BERT_MODEL_PATH=/path/to/tinybert/tinybert_converted.bin
BERT_TOKENIZER_PATH=/path/to/tinybert/bert_tokenizer.json

BERT_EMB_MODEL_PATH=/path/to/bert_emb/pytorch_model_converted.bin
BERT_EMB_TOKENIZER_PATH=/path/to/bert_emb/tokenizer.json

GPT2_MODEL_PATH=/path/to/gpt2/gpt2_converted.bin
GPT2_TOKENIZER_PATH=/path/to/gpt2/tokenizer.json

WHISPER_MODEL_PATH=/path/to/whisper/pytorch_model_converted.bin
WHISPER_TOKENIZER_PATH=/path/to/whisper/tokenizer.json

VIT_MODEL_PATH=/path/to/vit/pytorch_model_converted.bin
VIT_CONFIG_PATH=/path/to/vit/config.json

YOLOS_MODEL_PATH=/path/to/yolos/pytorch_model_converted.bin
YOLOS_CONFIG_PATH=/path/to/yolos/config.json
```

Make sure to replace the placeholder paths with the actual paths where your models and tokenizers are stored.

## Usage

Run the main script to start with the virtual assistant:

NOTE: You might have to install gradio

```sh
pip install gradio
```

```bash
python main.py
```

Or you can run the api server by running the following commnad:

```sh
uvicorn api:app --port 8080
```

## Docker

If you don't want to install the dependencies because it may interfere with your current environment or if you just want to avoid the hassel of install the dependencies and deal with the possible issues, you can also use the docker image. First we build the docker image using the following command.

```sh
docker build -t tools/jarvis -f ./deployment/Dockerfile .
```

`The resulting image is just around **~ 200Mb**`

Before we run the image (i.e. start the container), To ensure the application runs correctly, you need to create a .env file under the deployment directory. This file should contain the necessary environment variables definitions. Here is an example of what the .env file should look like:

```sh
BASE_PATH_MOUNT=/path/to/models
```

Make sure to replace the placeholder paths with the actual paths where all your models and tokenizers are stored.

Then we can start the containers by running the following command:

```sh
docker compose -f ./deployment/docker-compose.yaml up
```

## References

Implementation of the inference for the following models that use transformers were based on the [picoGPT implementation ](https://github.com/jaymody/picoGPT)

- BERT models (used for QA and Text Embedding)
  - Weights and Tokenizers:
    - QA Model: [Intel/dynamic_tinybert](https://huggingface.co/Intel/dynamic_tinybert)
    - Text Embedding model:[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Whisper
  - Weights and Tokenizers:
    - Speech Recognition : [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)
- ViT
  - Weights and Config:
    - Image Classification : [WinKawaks/vit-tiny-patch16-224](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)
- YOLOs
  - Weights and Tokenizers:
    - Object Detection : [hustvl/yolos-tiny](https://huggingface.co/hustvl/yolos-tiny)
