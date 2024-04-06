set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip install llama_index llama_index.core llama_index.vector_stores.pinecone llama_index.core  llama_index.llms.openai llama_index.embeddings.openai llama_index.embeddings.huggingface llama_index.llms.huggingface transformers llama_index.vector_stores.milvus huggingface_hub llama_index.llms.llama_cpp gradio torch milvus unstructured spacy llama_index.readers.nougat_ocr "unstructured[pdf]"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"