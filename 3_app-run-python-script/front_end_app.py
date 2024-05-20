import os
from chainlit.types import AskFileResponse
import chainlit as cl

# import gradio as gr

from utils.cmlllm import (
    CMLLLM,
    get_active_collections,
    get_supported_embed_models,
    get_supported_models,
    infer2,
)
from utils.check_dependency import check_gpu_enabled

MAX_QUESTIONS = 5


file_types = ["pdf", "html", "txt"]


llm_choice = get_supported_models()
collection_list_items = get_active_collections()
embed_models = get_supported_embed_models()


@cl.on_message
async def on_message(message: cl.Message):
    await message.reply("Hello")


if __name__ == "__main__":
    cl.run()
