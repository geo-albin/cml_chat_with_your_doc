import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import UnstructuredReader
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from huggingface_hub import hf_hub_download
import time
import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from utils.duplicate_preprocessing import DuplicateRemoverNodePostprocessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from utils.upload import Upload_files
import torch
import logging
import sys
import gradio as gr
import atexit
import utils.vectordb as vectordb


def exit_handler():
    print("cmlllmapp is exiting!")
    vectordb.stop_vector_db()


atexit.register(exit_handler)

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])


MODELS_PATH = "./models"
EMBEDSS_PATH = "./embed_models"

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,
    local_files_only=True,
)

embed_model = "thenlper/gte-large"

n_gpu_layers = -1
if torch.cuda.is_available():
    n_gpu_layers = 1

Settings.llm = LlamaCPP(
    model_path=model_path,
    temperature=0.0,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": n_gpu_layers},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=embed_model,
    cache_folder=EMBEDSS_PATH,
)

Settings.callback_manager = callback_manager

node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=20)
Settings.node_parser = node_parser

melvus_start = vectordb.reset_vector_db()
print(f"melvus_start = {melvus_start}")
collection = vectordb.create_vector_db_collection()
print(f"melvus collection created = {collection}")

vector_store = MilvusVectorStore(
    dim=1024, overwrite=True, collection_name="cml_rag_collection"
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

postprocessor = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    threshold_cutoff=0.7,
)

chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT,
    verbose=True,
    postprocessor=[postprocessor, DuplicateRemoverNodePostprocessor()],
)


def Infer2(history):
    query_text = history[-1][0]

    if len(query_text) == 0:
        history[-1][1] = "Please ask some questions"
    else:
        history[-1][1] = ""
        streaming_response = chat_engine.stream_chat(query_text)
        for token in streaming_response.response_gen:
            history[-1][1] = history[-1][1] + token
            yield history


def Infer(query, history=None):
    print(f"Albin : query = {query}")

    query_text = ""
    if isinstance(query, dict) and query["text"]:
        query_text = query["text"]
    elif isinstance(query, str):
        query_text = query
    else:
        return ""

    if len(query_text) == 0:
        return "Please ask some questions"

    streaming_response = chat_engine.stream_chat(query_text)
    generated_text = ""
    for token in streaming_response.response_gen:
        generated_text = generated_text + token
        yield generated_text


def Ingest(ingest_via_cml_job=False, progress=gr.Progress()):
    file_extractor = {
        ".html": UnstructuredReader(),
        ".pdf": UnstructuredReader(),
        ".txt": UnstructuredReader(),
    }

    if torch.cuda.is_available():
        file_extractor[".pdf"] = PDFNougatOCR()

    progress(0.3, desc="loading the document reader...")

    reader = SimpleDirectoryReader(
        input_dir="./assets/doc_list",
        recursive=True,
        file_extractor={
            ".html": UnstructuredReader(),
            ".pdf": UnstructuredReader(),
            ".txt": UnstructuredReader(),
        },
    )
    documents = reader.load_data(num_workers=16, show_progress=True)

    progress(0.4, desc="done loading the document reader...")

    vector_store = MilvusVectorStore(
        dim=1024, overwrite=True, collection_name="cml_rag_collection"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    progress(0.45, desc="done starting the vector db and set the storage context...")

    start_time = time.time()
    progress(0.5, desc="start indexing the document...")
    index = VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context, show_progress=True
    )

    op = "Completed data ingestion. took " + str(time.time() - start_time) + " seconds."

    print(f"{op}")
    progress(0.6, desc=op)

    start_time = time.time()
    progress(0.7, desc="start dataset generation from the document...")
    data_generator = DatasetGenerator.from_documents(documents)

    dataset_op = (
        "Completed data set generation. took "
        + str(time.time() - start_time)
        + " seconds."
    )
    op += "\n" + dataset_op

    progress(0.75, desc=dataset_op)
    progress(0.8, desc="start generating questions from the document...")
    eval_questions = data_generator.generate_questions_from_nodes(num=5)

    i = 1
    for q in eval_questions:
        op += "\nQuestion " + str(i) + " - " + str(q) + "."
        i += 1

    write_list_to_file(eval_questions, "questions.txt")
    progress(0.9, desc="done generating questions from the document...")

    if ingest_via_cml_job:
        melvus_stop = vectordb.stop_vector_db()
        print(f"melvus_stop = {melvus_stop}")

    return op


def write_list_to_file(lst, filename):
    with open(filename, "w") as f:
        for item in lst:
            f.write(str(item) + "\n")


def upload_document_and_ingest(files, progress=gr.Progress()):
    if len(files) == 0:
        return "Please add some files..."
    Upload_files(files, progress)
    return Ingest(False, progress)


def clear_chat_engine():
    chat_engine.reset()


if __name__ == "__main__":
    Ingest(True)
