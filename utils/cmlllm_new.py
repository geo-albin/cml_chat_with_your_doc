import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import UnstructuredReader, PDFReader
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from huggingface_hub import hf_hub_download, snapshot_download
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
import subprocess
import gradio as gr
import atexit
import utils.vectordb as vectordb
from llama_index.core.memory import ChatMemoryBuffer


QUESTIONS_FOLDER = "questions"


def exit_handler():
    print("cmlllmapp is exiting!")
    vectordb.stop_vector_db()


atexit.register(exit_handler)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])

supported_embed_models = ["thenlper/gte-large"]


def get_supported_embed_models():
    embedList = list(supported_embed_models)
    return embedList


supported_llm_models = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "google/gemma-7b-it": "gemma-7b-it.gguf",
}


def get_supported_models():
    llmList = list(supported_llm_models)
    return llmList


active_collection_available = {"cml_rag_collection": False}


def get_active_collections():
    return list(active_collection_available)


print("resetting the questions")
print(subprocess.run([f"rm -rf {QUESTIONS_FOLDER}"], shell=True))

milvus_start = vectordb.reset_vector_db()
print(f"milvus_start = {milvus_start}")


class CMLLLM:
    MODELS_PATH = "./models"
    EMBED_PATH = "./embed_models"

    questions_folder = QUESTIONS_FOLDER

    def __init__(
        self,
        model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        embed_model_name="thenlper/gte-large",
        temperature=0.0,
        max_new_tokens=256,
        context_window=3900,
        gpu_layers=20,
        dim=1024,
        collection_name="cml_rag_collection",
        memory_token_limit=3900,
        sentense_embedding_percentile_cutoff=0.8,
        similarity_top_k=2,
        progress=gr.Progress(),
    ):
        if len(model_name) == 0:
            model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        if len(embed_model_name) == 0:
            embed_model_name = "thenlper/gte-large"
        n_gpu_layers = 0
        if torch.cuda.is_available():
            print("It is a GPU node, setup GPU.")
            n_gpu_layers = gpu_layers

        model_path = self.get_model_path(model_name)
        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=128)

        progress((1, 25), desc="setting the global parameters")

        self.set_global_settings(
            model_path=model_path,
            embed_model_path=embed_model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            node_parser=self.node_parser,
        )

        self.collection_name = collection_name

        if not self.collection_name in active_collection_available:
            active_collection_available[self.collection_name] = False

        progress((2, 25), desc="setting the vector db")

        self.vector_store = MilvusVectorStore(
            dim=dim,
            overwrite=True,
            collection_name=self.collection_name,
        )

        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        progress((3, 25), desc="setting the chat engine")

        self.chat_engine = self.index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            postprocessor=[
                SentenceEmbeddingOptimizer(
                    percentile_cutoff=sentense_embedding_percentile_cutoff
                ),
                DuplicateRemoverNodePostprocessor(),
            ],
            memory=ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit),
            system_prompt=(
                "You are an expert Q&A system that is trusted around the world.\n"
                "Always answer the query using the provided context information and not prior knowledge."
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context' or 'The context information'"
                " or 'This information is not directly stated in the context provided' or anything along those lines.\n"
                "If the provided context dont have the information, answer 'I dont know'.\n"
                "Please cite file name and page number along with your answers."
            ),
            similarity_top_k=similarity_top_k,
        )
        print("Albin, started the chat engine")

    def infer(self, msg, history):
        query_text = msg
        print(f"query = {query_text}")

        if len(query_text) == 0:
            yield "Please ask some questions"
            return

        if (
            self.collection_name in active_collection_available
            and active_collection_available[self.collection_name] != True
        ):
            yield "No documents are processed yet. Please process some documents.."
            return

        # history[-1][1] = ""

        streaming_response = self.chat_engine.chat(query_text)
        generated_text = ""
        for token in streaming_response.response:
            print(f"Albin, chat response = {token}")
            # history[-1][1] += token
            generated_text = generated_text + token
            yield generated_text
        # return history
        # history[-1][1] = generated_text

    def ingest(self, files, questions, progress=gr.Progress()):
        if not (self.collection_name in active_collection_available):
            return "Some issues with the llm and colection setup. please try setting up the llm and the vector db again."

        file_extractor = {
            ".html": UnstructuredReader(),
            ".pdf": PDFReader(),
            ".txt": UnstructuredReader(),
        }

        if torch.cuda.is_available():
            file_extractor[".pdf"] = PDFNougatOCR()

        print(f"questions = {questions}")

        progress(0.3, desc="loading the documents")

        filename_fn = lambda filename: {"file_name": os.path.basename(filename)}

        try:
            start_time = time.time()
            op = ""
            i = 1
            for file in files:
                progress(0.4, desc=f"loading document {os.path.basename(file)}")
                reader = SimpleDirectoryReader(
                    input_files=[file],
                    file_extractor=file_extractor,
                    file_metadata=filename_fn,
                )
                document = reader.load_data(num_workers=1, show_progress=True)

                progress(0.4, desc=f"done loading document {os.path.basename(file)}")

                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )

                progress(
                    0.4, desc=f"start indexing the document {os.path.basename(file)}"
                )
                nodes = self.node_parser.get_nodes_from_documents(document)

                index = VectorStoreIndex(
                    nodes, storage_context=storage_context, show_progress=True
                )

                progress(
                    0.4, desc=f"done indexing the document {os.path.basename(file)}"
                )

                op += (
                    "Completed data ingestion. took "
                    + str(time.time() - start_time)
                    + " seconds."
                )

                print(f"{op}")
                progress(0.4, desc=op)

                start_time = time.time()
                print(
                    f"start dataset generation from the document {os.path.basename(file)}."
                )
                progress(
                    0.4,
                    desc=f"start dataset generation from the document {os.path.basename(file)}.",
                )

                data_generator = DatasetGenerator.from_documents(documents=document)

                dataset_op = (
                    f"Completed data set generation for file {os.path.basename(file)}. took "
                    + str(time.time() - start_time)
                    + " seconds."
                )
                op += "\n" + dataset_op
                print(f"{dataset_op}")
                progress(0.4, desc=dataset_op)
                print(
                    f"start generating questions from the document {os.path.basename(file)}"
                )
                progress(
                    0.4,
                    desc=f"start generating questions from the document {os.path.basename(file)}",
                )
                eval_questions = data_generator.generate_questions_from_nodes(
                    num=questions
                )

                for q in eval_questions:
                    op += "\nQuestion " + str(i) + " - " + str(q)
                    i += 1

                self.write_list_to_file(
                    eval_questions,
                    self.questions_folder + self.collection_name + "questions.txt",
                )
                print(
                    f"done generating questions from the document {os.path.basename(file)}"
                )
                progress(
                    0.4,
                    desc=f"done generating questions from the document {os.path.basename(file)}",
                )
                print(subprocess.run([f"rm -f {file}"], shell=True))

            progress(0.9, desc="done processing the documents...")
            print("done processing the documents...")
            active_collection_available[self.collection_name] = True

        except Exception as e:
            print(e)
            op = f"ingestion failed with exception {e}"
            progress(0.9, desc=op)
        return op

    def write_list_to_file(self, lst, filename):
        with open(filename, "a") as f:
            for item in lst:
                f.write(str(item) + "\n")

    def read_list_from_file(self, filename="questions.txt"):
        lst = []
        if os.path.exists(self.questions_folder + self.collection_name + filename):
            with open(
                self.questions_folder + self.collection_name + filename, "r"
            ) as f:
                for line in f:
                    lst.append(line.strip())  # Remove newline characters
        return lst

    def upload_document_and_ingest(self, files, questions, progress=gr.Progress()):
        if len(files) == 0:
            return "Please add some files..."
        return self.ingest(files, questions, progress)

    def set_global_settings(
        self,
        model_path,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        node_parser,
    ):
        Settings.llm = LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=context_window,
            # kwargs to pass to __call__()
            # generate_kwargs={"temperature": 0.0, "top_k": 5, "top_p": 0.95},
            generate_kwargs={"temperature": temperature},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_path,
            cache_folder=self.EMBED_PATH,
            # encode_kwargs={"normalize_embeddings": True},
        )

        Settings.callback_manager = callback_manager
        Settings.node_parser = node_parser

    def get_model_path(self, model_name):
        filename = supported_llm_models[model_name]
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            resume_download=True,
            cache_dir=self.MODELS_PATH,
            local_files_only=False,
        )
        return model_path

    def get_embed_model_path(self, embed_model):
        embed_model_path = snapshot_download(
            repo_id=embed_model,
            resume_download=True,
            cache_dir=self.EMBEDS_PATH,
            local_files_only=False,
        )
        return embed_model_path

    def clear_chat_engine(self):
        self.chat_engine.reset()
