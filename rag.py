import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

from duplicate_preprocessing import DuplicateRemoverNodePostprocessor
from transformers import BitsAndBytesConfig
from llama_index.vector_stores.milvus import MilvusVectorStore
import torch


load_dotenv()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])

# LLM = OpenAI(model="gpt-3.5-turbo", temperature=0)
# Embed_Model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)

# LLM = MistralAI(model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0)
# Embed_Model = MistralAIEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", embed_batch_size=100)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # bnb_4bit_compute_dtype=torch.float16,
#     # bnb_4bit_quant_type="nf4",
#     # bnb_4bit_use_double_quant=True,
# )

Settings.llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0, "top_k": 50, "top_p": 0.95},
    device_map="auto",
    model_kwargs={
        "torch_dtype": torch.float16, 
        "llm_int8_enable_fp32_cpu_offload": True,
        "bnb_4bit_quant_type": 'nf4',
        "bnb_4bit_use_double_quant":True,
        "bnb_4bit_compute_dtype":torch.bfloat16,
        "load_in_4bit": True}
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Settings.llm = LLM
# Settings.embed_model = Embed_Model 
Settings.callback_manager = callback_manager

# pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
# pinecone_index = pc.Index(os.environ['PINECONE_INDEX'])
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

vector_store = MilvusVectorStore(dim=1536, overwrite=True, collection_name="cml_rag_collection")

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

postprocessor = SentenceEmbeddingOptimizer(
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

chat_engine=index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True, postprocessor=[postprocessor, DuplicateRemoverNodePostprocessor()])

def Infer(query, history):
    print(f"Albin : query = {query}")
    
    query_text = ""
    if isinstance(query, dict) and query["text"]:
        query_text = query["text"]
    elif isinstance(query, str):
        query_text = query
    else:
        return ""
    
    streaming_response=chat_engine.stream_chat(query_text)
    generated_text=""
    for token in streaming_response.response_gen:
        generated_text=generated_text+token
        yield generated_text
    


# demo = gr.Interface(
#     fn=infer,
#     inputs=["text"],
#     outputs=["text"],
# )
# infer = gr.ChatInterface(
#     fn=infer, 
#     examples=["What is llama index?", "What is RAG?"], 
#         title="llama index chat Bot", 
#         chatbot=gr.Chatbot(height=700),
#         multimodal=False
#         )

# # ingest = gr.Interface(
# #     fn=Ingest,
# #     inputs=["button"],
# #     outputs=["text"],
# # )
# ingest = gr.Blocks()
# with ingest:
#     btn = gr.Button(value="Please press to start ingestion")
#     output = gr.Textbox(label="ingestion progress", max_lines=10, interactive=False)
#     btn.click(Ingest, inputs=None, outputs=[output])

# demo = gr.TabbedInterface([infer, ingest], ["Chat bot", "Data Ingestion"])
# demo.launch()

if __name__ == '__main__':
    pass