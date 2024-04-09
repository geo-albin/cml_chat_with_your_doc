import os
import gradio as gr
import subprocess
from utils.cmlllm import (
    upload_document_and_ingest,
    clear_chat_engine,
    Infer,
    get_supported_models,
    get_active_collections,
)

MAX_QUESTIONS = 5


def read_list_from_file(filename="questions.txt"):
    lst = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                lst.append(line.strip())  # Remove newline characters
    return lst


def read_list_from_file_button(filename="questions.txt"):
    lst = read_list_from_file(filename=filename)
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


def delete_docs(progress=gr.Progress()):
    progress(0.1, desc="deleting server side documents...")
    print(subprocess.run(["rm -rf ./assets/doc_list"], shell=True))
    progress(0.5, desc="deleting answer files...")
    print(subprocess.run(["rm -f questions.txt"], shell=True))
    progress(0.9, desc="done deleting server side documents...")
    return "done deleting server side documents..."


questions = read_list_from_file()

file_types = ["pdf", "html", "txt"]


def get_value(label):
    return label.value


clear_btn = gr.ClearButton("Clear")

infer = gr.ChatInterface(
    fn=Infer,
    title="CML chat Bot - v2",
    examples=questions,
    chatbot=gr.Chatbot(
        height=700,
        show_label=False,
        show_copy_button=True,
        layout="bubble",
        bubble_full_width=True,
    ),
    clear_btn=clear_btn,
    submit_btn=gr.Button("Submit"),
)
with infer:
    clear_btn.click(clear_chat_engine)
    with gr.Row():
        with gr.Accordion("Advanced - Document references", open=False):
            with gr.Row():
                doc_source1 = gr.Textbox(
                    label="Reference 1", lines=2, container=True, scale=20
                )
                source1_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                doc_source2 = gr.Textbox(
                    label="Reference 2", lines=2, container=True, scale=20
                )
                source2_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                doc_source3 = gr.Textbox(
                    label="Reference 3", lines=2, container=True, scale=20
                )
                source3_page = gr.Number(label="Page", scale=1)

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
        upload_button = gr.Button("Click to process the files")
        upload_button.click(
            upload_document_and_ingest,
            inputs=[documents, questions_slider],
            outputs=[db_progress],
        )

questions = gr.Blocks(css="assets/custom_label.css")
with questions:
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

admin_submit = gr.Button("Submit")
admin = gr.Blocks()
with admin:
    with gr.Row():
        with gr.Accordion("Select LLM", open=True):
            with gr.Row(equal_height=True):
                llm_model = gr.Dropdown(
                    choices=get_supported_models(),
                    value="mistralai/Mistral-7B-Instruct-v0.2",
                    label="LLM Model",
                )

    with gr.Row():
        gr.Dropdown(
            choices=get_active_collections(),
            label="Collection to use",
            allow_custom_value=True,
            info="Please select a collection to use for saving the data and querying!",
        ),

    with gr.Row():
        admin_submit


demo = gr.TabbedInterface(
    interface_list=[admin, upload, infer, questions],
    tab_names=[
        "Step 1 - Setup the LLM and Vector DB",
        "Step 2 - Document pre-processing",
        "Step 3 - Conversation with chatbot",
        "Some questions about the topic",
    ],
    title="CML Chat application - v2",
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
