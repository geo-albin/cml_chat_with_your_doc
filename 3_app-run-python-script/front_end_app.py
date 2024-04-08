import os
import gradio as gr
import subprocess
from utils.cmlllm import upload_document_and_ingest, clear_chat_engine, Infer

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


infer = gr.ChatInterface(
    fn=Infer,
    title="CML chat Bot - v2",
    examples=questions,
    chatbot=gr.Chatbot(height=700),
    # multimodal=False,
    submit_btn=gr.Button("Submit"),
)

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
                    value=5,
                    step=1,
                    label="Number of questions to be generated about the topic",
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

admin = gr.Blocks()
with admin:
    with gr.Row():
        admin_progress = gr.Textbox(label="Admin event status", value="None")
    with gr.Row():
        clean_up_docs = gr.Button("Click to do file cleanup")
        clean_up_docs.click(delete_docs, inputs=None, outputs=admin_progress)


demo = gr.TabbedInterface(
    interface_list=[upload, infer, questions, admin],
    tab_names=[
        "Step 1 - Document pre-processing",
        "Step 2 - Conversation with chatbot",
        "Some questions about the topic",
        "Admin tab",
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
