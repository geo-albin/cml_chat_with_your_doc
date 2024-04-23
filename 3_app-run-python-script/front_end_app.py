import os
import gradio as gr

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


def update_active_collections(collection_name):
    global collection_list_items
    collection_list_items = get_active_collections()
    print(f"new collection {collection_list_items}")
    collection = ""
    if collection_name is not None and len(collection_name) != 0:
        collection = collection_name
    elif len(collection_list_items) != 0:
        collection = collection_list_items[0]

    return gr.Dropdown(choices=collection_list_items, value=collection)


def get_latest_default_collection():
    collection_list_items = get_active_collections()
    collection = ""
    if len(collection_list_items) != 0:
        collection = collection_list_items[0]

    return collection


llm = CMLLLM()
llm.set_collection_name(collection_name=collection_list_items[0])


def upload_document_and_ingest_new(
    files, questions, collection_name, progress=gr.Progress()
):
    if files is None or len(files) == 0:
        gr.Error("Please add some files...")
    return llm.ingest(files, questions, collection_name, progress)


def update_chatbot(user_message, history):
    return "", history + [[user_message, None]]


def reconfigure_llm(
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    embed_model_name="thenlper/gte-large",
    temperature=0.0,
    max_new_tokens=256,
    context_window=3900,
    gpu_layers=20,
):
    llm.set_global_settings_common(
        model_name=model_name,
        embed_model_path=embed_model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        n_gpu_layers=gpu_layers,
    )
    return "Done reconfiguring llm!!!"


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


def open_chat_accordion():
    return gr.Accordion("Chat with your documents", open=True)


def update_header(collection_name):
    string = f"Now using the collection {collection_name}"
    if collection_name is None or len(collection_name) == 0:
        string = f"Please set the collection name in the 'Advanced Options' and process the documents"

    return gr.TextArea(
        value=f"AI Chat with your document. \n{string}",
        show_label=False,
        interactive=False,
        max_lines=1,
        lines=1,
    )


def close_doc_process_accordion():
    return gr.Accordion("Process your documents", open=False)


def get_runtime_information():
    if check_gpu_enabled():
        return "AI chatbot is running using GPU"
    else:
        return "AI chatbot is running using CPU"


def demo():
    with gr.Blocks(title="AI Chat with your documents") as demo:
        collection_name = gr.State(value="default_collection")
        nr_of_questions = gr.State(value=1)

        gr.Markdown(
            """<center><h2>AI Chat with your documents</h2></center>
        <h3>Chat with your documents (pdf, text and html)</h3>"""
        )
        with gr.Tab("Chat with your document"):
            upload = gr.Blocks()
            with upload:
                doc_accordion = gr.Accordion("Process your documents", open=True)
                chat_accordion = gr.Accordion("Chat with your documents", open=False)
                with doc_accordion:
                    header = gr.TextArea(
                        value=f"Now using the collection {get_latest_default_collection()}",
                        show_label=False,
                        interactive=False,
                        max_lines=1,
                        lines=1,
                    )
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
                            # value="None",
                            interactive=False,
                            max_lines=3,
                            lines=3,
                        )
                    upload_button = gr.Button("Click to process the files")
                    upload_button.click(
                        upload_document_and_ingest_new,
                        inputs=[documents, nr_of_questions, collection_name],
                        outputs=[db_progress],
                    ).then(open_chat_accordion, inputs=[], outputs=[chat_accordion])
            with chat_accordion:
                gr.ChatInterface(
                    fn=infer2,
                    chatbot=chat_bot,
                    clear_btn=clear_btn,
                    submit_btn=submit_btn,
                    additional_inputs=[collection_name],
                )
                clear_btn.click(
                    llm.clear_chat_engine, inputs=[collection_name], outputs=None
                )

        with gr.Tab("Advanced Options"):
            admin = gr.Blocks()
            with admin:
                with gr.Row():
                    llm_progress = gr.Textbox(
                        label="LLM processing status",
                        show_label=False,
                        # value="",
                        interactive=False,
                        max_lines=10,
                        visible=True,
                    )
                with gr.Accordion("Runtime informations", open=True):
                    gr.TextArea(
                        show_label=False,
                        value=get_runtime_information,
                        interactive=False,
                        max_lines=1,
                        lines=1,
                    )
                with gr.Accordion("LLM Configuration", open=False):
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
                            ],
                            outputs=[],
                        )
                with gr.Row():
                    with gr.Accordion(
                        "Number of automatic questions generated per the uploaded docs",
                        open=False,
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
                                show_label=False,
                            )
                            questions_slider.change(
                                lambda questions: questions,
                                inputs=[questions_slider],
                                outputs=[nr_of_questions],
                            )
                with gr.Row():
                    with gr.Accordion("collection configuration", open=False):
                        with gr.Row():
                            collection_list = gr.Dropdown(
                                choices=collection_list_items,
                                label="Configure an existing collection or create a new one below",
                                allow_custom_value=True,
                                value=get_latest_default_collection,
                            )
                            collection_list.change(
                                llm.set_collection_name,
                                inputs=[collection_list],
                                outputs=[llm_progress],
                            ).then(
                                lambda collection_name: collection_name,
                                inputs=[collection_list],
                                outputs=[collection_name],
                            ).then(
                                update_active_collections,
                                inputs=[collection_name],
                                outputs=[collection_list],
                            ).then(
                                update_header,
                                inputs=[collection_name],
                                outputs=[header],
                            )
                        with gr.Row():
                            with gr.Accordion(
                                "collection operations",
                                open=False,
                                label="Select a collection, and click the button to delete it",
                            ):
                                collection_delete_btn = gr.Button(
                                    "Delete the collection and the associated document embeddings"
                                )
                                collection_delete_btn.click(
                                    llm.delete_collection_name,
                                    inputs=[
                                        collection_list,
                                    ],
                                    outputs=[llm_progress],
                                ).then(
                                    update_active_collections,
                                    inputs=[],
                                    outputs=[collection_list],
                                ).then(
                                    lambda collection_name: collection_name,
                                    inputs=[collection_list],
                                    outputs=[collection_name],
                                ).then(
                                    update_header,
                                    inputs=[collection_name],
                                    outputs=[header],
                                )

                                refresh_btn = gr.Button("Refresh the collection list")
                                refresh_btn.click(
                                    update_active_collections,
                                    inputs=[],
                                    outputs=[collection_list],
                                ).then(
                                    lambda collection_name: collection_name,
                                    inputs=[collection_list],
                                    outputs=[collection_name],
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
