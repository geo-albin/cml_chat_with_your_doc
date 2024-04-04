import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.readers.file import UnstructuredReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from llama_index.vector_stores.milvus import MilvusVectorStore
from huggingface_hub import hf_hub_download
import time
import torch
import utils.vector_db_utils as vector_db
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

load_dotenv()
# print(f"PINECONE_API_KEY = {os.environ['PINECONE_API_KEY']}\nPINECONE_ENVIRONMENT = {os.environ['PINECONE_ENVIRONMENT']}")

def Ingest(stop_vector_db=False):
    # Step 1: read and clean the data from HTML.
    # We are using UnStructured reader for the same.
    reader = SimpleDirectoryReader(input_dir="./doc_list", recursive=True, file_extractor={'.html': UnstructuredReader(), '.pdf': UnstructuredReader(), '.txt': UnstructuredReader()})
    documents = reader.load_data(num_workers=16, show_progress=True)

    # Step 2: Create a node parser abstraction
    node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=20)
    # nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)

    # Settings.llm = llm
    # Settings.embed_model = embed_model


    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     # bnb_4bit_compute_dtype=torch.float16,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_use_double_quant=True,
    # )

    # Settings.llm = HuggingFaceLLM(
    #     model_name="mistralai/Mistral-7B-Instruct-v0.1",
    #     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    #     context_window=3900,
    #     max_new_tokens=256,
    #     generate_kwargs={"temperature": 0, "top_k": 50, "top_p": 0.95},
    #     device_map="auto",
    #     model_kwargs={
    #         "torch_dtype": torch.float16, 
    #         "llm_int8_enable_fp32_cpu_offload": True,
    #         "bnb_4bit_quant_type": 'nf4',
    #         "bnb_4bit_use_double_quant":True,
    #         "bnb_4bit_compute_dtype":torch.bfloat16,
    #         "load_in_4bit": True}
    # )

    MODELS_PATH = "./models"

    model_path = hf_hub_download(
        repo_id= "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        resume_download=True,
        cache_dir=MODELS_PATH,)

    n_gpu_layers = -1
    if torch.cuda.is_available():
        n_gpu_layers = 1

    Settings.llm =LlamaCPP(
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
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.node_parser = node_parser

    # index_name = os.environ['PINECONE_INDEX']
    # pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    # pinecone_index = pc.Index(index_name)
    # vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    melvus_start = vector_db.start_milvus()
    print(f"melvus_start = {melvus_start}")

    vector_store = MilvusVectorStore(dim=384, overwrite=True)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    start_time = time.time()

    index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)

    op = "Completed data ingestion. took " + str(time.time() - start_time) + " seconds"

    print(f"{op}")
    
    if stop_vector_db:
        melvus_stop = vector_db.stop_milvus()
        print(f"melvus_stop = {melvus_stop}")

    return op

if __name__ == '__main__':
    Ingest()