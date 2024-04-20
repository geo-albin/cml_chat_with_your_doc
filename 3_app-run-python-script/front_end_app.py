import os
import gradio as gr

from utils.cmlllm import (
    CMLLLM,
    get_active_collections,
    get_supported_embed_models,
    get_supported_models,
    infer2,
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
    print(f"new collection {collection_list_items}")
    return gr.Dropdown(choices=collection_list_items)


llm = CMLLLM()
global_chat_engine = llm.set_collection_name(collection_name=collection_list_items[0])


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
        embed_model_name=embed_model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        gpu_layers=gpu_layers,
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


def close_doc_process_accordion():
    return gr.Accordion("Process your documents", open=False)


def demo():
    with gr.Blocks(title="AI Chat with your documents") as demo:
        chat_engine = gr.State(value=global_chat_engine)
        collection_name = gr.State(value="cml_rag_collection")
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
                    upload_button = gr.Button("Click to process the files")
                    upload_button.click(
                        upload_document_and_ingest_new,
                        inputs=[documents, nr_of_questions, collection_name],
                        outputs=[db_progress],
                    ).then(open_chat_accordion, inputs=[], outputs=chat_accordion).then(
                        close_doc_process_accordion, inputs=[], outputs=doc_accordion
                    )
            with chat_accordion:
                gr.ChatInterface(
                    fn=llm.infer2,
                    title=f"AI Chat with your document - Currently using the collection {collection_name}",
                    chatbot=chat_bot,
                    clear_btn=clear_btn,
                    submit_btn=submit_btn,
                    additional_inputs=[collection_name, chat_engine],
                )
                clear_btn.click(
                    llm.clear_chat_engine, inputs=[chat_engine], outputs=None
                )

        with gr.Tab("Admin configurations[Optional]"):
            admin = gr.Blocks()
            with admin:
                with gr.Row():
                    llm_progress = gr.Textbox(
                        label="LLM processing status",
                        value="None",
                        interactive=False,
                        max_lines=10,
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
                        "Advanced options - automatic question generation",
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
                                label="Configure an existing collection or create a new one",
                                allow_custom_value=True,
                                value=collection_list_items[0],
                            )
                            collection_list.change(
                                llm.set_collection_name,
                                inputs=[collection_list],
                                outputs=[chat_engine, llm_progress],
                            ).then(
                                lambda collection_name: collection_name,
                                inputs=[collection_list],
                                outputs=[collection_name],
                            ).then(
                                update_active_collections,
                                inputs=[],
                                outputs=[collection_list],
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
