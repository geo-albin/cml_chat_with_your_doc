#!/bin/bash

if [[ "$USE_ONLY_CPU" == "true" ]]; then
    echo "installing CPU only version of llama-cpp"
    pip install llama-cpp-python
else
    echo "installing GPU version of llama-cpp"
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
fi

pip install llama_index llama_index.core llama_index.vector_stores.pinecone llama_index.core  llama_index.llms.openai llama_index.embeddings.openai llama_index.embeddings.huggingface llama_index.llms.huggingface transformers==4.37.2 llama_index.vector_stores.milvus huggingface_hub llama_index.llms.llama_cpp gradio torch milvus unstructured spacy llama_index.readers.nougat_ocr "unstructured[pdf]"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"