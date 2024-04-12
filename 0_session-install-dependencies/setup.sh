#!/bin/bash
# set CMAKE_ARGS=-DLLAMA_CUBLAS=on
# set FORCE_CMAKE=1
# pip install transformers==4.37.2 --force-reinstall --upgrade --no-cache-dir
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --upgrade --force-reinstall llama-cpp-python
pip3 install https://${CDSW_DOMAIN}/api/v2/python.tar.gz
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
pip install llama_index llama_index.core llama_index.vector_stores.pinecone llama_index.core  llama_index.llms.openai llama_index.embeddings.openai llama_index.embeddings.huggingface llama_index.llms.huggingface transformers==4.37.2 llama_index.vector_stores.milvus huggingface_hub llama_index.llms.llama_cpp gradio torch milvus unstructured spacy llama_index.readers.nougat_ocr "unstructured[pdf]"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"