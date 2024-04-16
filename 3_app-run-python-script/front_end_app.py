import os
import gradio as gr

from utils.cmlllm import (
    CMLLLM,
    get_active_collections,
    get_supported_embed_models,
    get_supported_models,
)

MAX_QUESTIONS = 5


file_types = ["pdf", "html", "txt"]


def get_value(label):
    return label.value


clear_btn = gr.ClearButton("Clear")
chat_bot = gr.Chatbot(
    height=500,
    show_label=False,
    show_copy_button=True,
    layout="bubble",
    bubble_full_width=True,
)
submit_btn = gr.Button("Submit")

llm_choice = get_supported_models()
collection_list_items = get_active_collections()
embed_models = get_supported_embed_models()


def update_active_collections():
    global collection_list_items
    collection_list_items = get_active_collections()
    return collection_list_items
    print(f"Albin : new collection {collection_list_items}")


llm = CMLLLM()


def read_list_from_file_button(filename="questions.txt"):
    lst = llm.read_list_from_file(filename=filename)
    lists = []
    for i in range(MAX_QUESTIONS):
        if (len(lst) > i) and (len(lst[i]) != 0):
            lists.append(
                gr.Label(
                    visible=True,
                    value=lst[i],
                    container=True,
                )
            )
        else:
            lists.append(
                gr.Label(
                    visible=False,
                    value="",
                    container=True,
                )
            )

    return lists[0], lists[1], lists[2], lists[3], lists[4]


def upload_document_and_ingest_new(files, questions, progress=gr.Progress()):
    if files is None or len(files) == 0:
        gr.Error("Please add some files...")
    return llm.ingest(files, questions, progress)


def update_chatbot(user_message, history):
    return "", history + [[user_message, None]]


def reconfigure_llm(
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    embed_model_name="thenlper/gte-large",
    temperature=0.0,
    max_new_tokens=256,
    context_window=3900,
    gpu_layers=20,
    dim=1024,
    memory_token_limit=3900,
    sentense_embedding_percentile_cutoff=0.8,
    similarity_top_k=2,
    progress=gr.Progress(),
):
    global llm
    llm = CMLLLM(
        model_name=model_name,
        embed_model_name=embed_model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        gpu_layers=gpu_layers,
        dim=dim,
        memory_token_limit=memory_token_limit,
        sentense_embedding_percentile_cutoff=sentense_embedding_percentile_cutoff,
        similarity_top_k=similarity_top_k,
        progress=progress,
    )
    return "Done configuring llm!!!"


def validate_llm(model_name, embed_model_name):
    ret = True
    if model_name is None or len(model_name) == 0:
        gr.Error("Select a valid model name")
        ret = False

    if embed_model_name is None or len(embed_model_name) == 0:
        gr.Error("Select a valid embed model name")
        ret = False

    return ret


def validate_collection_name(collectionname):
    ret = True
    if collectionname is None or len(collectionname) == 0:
        gr.Error("invalid collection name, please set a valid collection name string.")
        ret = False

    return ret


def demo():
    with gr.Blocks(title="CML chat application") as demo:
        gr.Markdown(
            """<center><h2>CML Chat application - v2</center></h2>
        <h3>Chat with your documents (pdf and html)</h3>"""
        )
        with gr.Tab("Step 1 - Review the LLM and Vector DB configuration[Optional]"):
            admin = gr.Blocks()
            with admin:
                with gr.Accordion("LLM Configuration", open=True):
                    llm_model = gr.Dropdown(
                        choices=llm_choice,
                        value=llm_choice[0],
                        label="LLM Model",
                    )
                    embed_model = gr.Dropdown(
                        choices=embed_models,
                        value=embed_models[0],
                        label="Embed Model",
                    )
                    with gr.Accordion("Configure model parameters", open=False):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0.0,
                            step=0.1,
                            label="Temperature configuration",
                            info="Temperature configuration",
                            interactive=True,
                        )
                        max_new_tokens = gr.Slider(
                            minimum=100,
                            maximum=512,
                            value=256,
                            step=1,
                            label="max_new_tokens",
                            info="max_new_tokens",
                            interactive=True,
                        )
                        context_window = gr.Slider(
                            minimum=1000,
                            maximum=5000,
                            value=3900,
                            step=1,
                            label="context_window",
                            info="context_window",
                            interactive=True,
                        )
                        gpu_layers = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=20,
                            step=1,
                            label="gpu_layers",
                            info="gpu_layers",
                            interactive=True,
                        )
                        memory_token_limit = gr.Slider(
                            minimum=1000,
                            maximum=5000,
                            value=3900,
                            step=1,
                            label="memory_token_limit",
                            info="memory_token_limit",
                            interactive=True,
                        )
                        sentense_embedding_percentile_cutoff = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.8,
                            step=0.1,
                            label="sentense_embedding_percentile_cutoff",
                            info="sentense_embedding_percentile_cutoff",
                            interactive=True,
                        )
                        similarity_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=2,
                            step=1,
                            label="similarity_top_k",
                            info="similarity_top_k",
                            interactive=True,
                        )

                with gr.Row():
                    with gr.Accordion("Configure vector DB parameters", open=False):
                        dim = gr.Slider(
                            minimum=100,
                            maximum=2000,
                            value=1024,
                            step=1,
                            label="dim",
                            info="dim",
                            interactive=True,
                        )
                with gr.Row():
                    llm_progress = gr.Textbox(
                        label="LLM processing status",
                        value="None",
                        interactive=False,
                        max_lines=10,
                    )
                with gr.Accordion("LLM reconfiguration", open=False):
                    with gr.Row():
                        configure_button = gr.Button("Click to configure LLM")
                        configure_button.click(
                            validate_llm,
                            inputs=[
                                llm_model,
                                embed_model,
                            ],
                            outputs=[],
                        ).success(
                            reconfigure_llm,
                            inputs=[
                                llm_model,
                                embed_model,
                                temperature,
                                max_new_tokens,
                                context_window,
                                gpu_layers,
                                dim,
                                memory_token_limit,
                                sentense_embedding_percentile_cutoff,
                                similarity_top_k,
                            ],
                            outputs=[llm_progress],
                        )

        with gr.Tab("Step 2 - Document pre-processing"):
            upload = gr.Blocks()
            with upload:
                with gr.Row():
                    documents = gr.Files(
                        height=100,
                        file_count="multiple",
                        file_types=file_types,
                        interactive=True,
                        label="Upload your pdf, html or text documents (single or multiple)",
                    )
                with gr.Row():
                    db_progress = gr.Textbox(
                        label="Document processing status",
                        value="None",
                        interactive=False,
                        max_lines=10,
                    )
                with gr.Row():
                    with gr.Accordion(
                        "Advanced options - automatic question generation", open=False
                    ):
                        with gr.Row():
                            questions_slider = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=1,
                                step=1,
                                label="Number of questions to be generated per document",
                                info="Number of questions",
                                interactive=True,
                            )
                with gr.Row():
                    collection_list = gr.Dropdown(
                        choices=collection_list_items,
                        label="Collection to use",
                        allow_custom_value=True,
                        value=collection_list_items[0],
                    )
                    collection_list.change(
                        llm.set_collection_name,
                        inputs=[collection_list],
                        outputs=[db_progress],
                    ).then(
                        update_active_collections,
                        inputs=[],
                        outputs=[collection_list],
                    )
                with gr.Row():
                    upload_button = gr.Button("Click to process the files")
                    upload_button.click(
                        validate_collection_name, inputs=collection_list, outputs=None
                    ).success(
                        llm.set_collection_name, inputs=collection_list, outputs=None
                    ).then(
                        upload_document_and_ingest_new,
                        inputs=[documents, questions_slider],
                        outputs=[db_progress],
                    )

        with gr.Tab("Step 3 - Conversation with chatbot"):
            gr.ChatInterface(
                fn=llm.infer,
                title="CML chat Bot - v2",
                chatbot=chat_bot,
                clear_btn=clear_btn,
                submit_btn=submit_btn,
            )
            clear_btn.click(llm.clear_chat_engine())

        with gr.Tab("Some questions about the topic"):
            questions_tab = gr.Blocks(css="assets/custom_label.css")
            with questions_tab:
                list0, list1, list2, list3, list4 = read_list_from_file_button()
                with gr.Row():
                    list0
                    list1
                    list2
                    list3
                    list4
                    with gr.Row():
                        question_reload_btn = gr.Button("Update the topic")
                        question_reload_btn.click(
                            read_list_from_file_button,
                            inputs=None,
                            outputs=[list0, list1, list2, list3, list4],
                        )

    demo.queue()

    if "CML" in os.environ and os.environ["CML"] == "yes":
        demo.launch(
            show_error=True,
            debug=True,
            server_name="127.0.0.1",
            server_port=int(os.getenv("CDSW_APP_PORT")),
        )
    else:
        demo.launch(debug=True)


if __name__ == "__main__":
    demo()
