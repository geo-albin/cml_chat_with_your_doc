import os
import gradio as gr
import subprocess
from utils.cmlllm import upload_document_and_ingest, clear_chat_engine, Infer

MAX_QUESTIONS = 5


def read_list_from_file(filename="questions.txt") -> list:
    lst = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                lst.append(line.strip())  # Remove newline characters
    return lst


def read_list_from_file_button(filename="questions.txt"):
    lst = read_list_from_file(filename=filename)
    buttons = []
    for i in range(MAX_QUESTIONS):
        if len(lst[i]) != 0:
            buttons.append(
                gr.Button(
                    visible=True,
                    value=lst[i],
                )
            )
        else:
            buttons.append(gr.Button(visible=False, value=""))

    return buttons[0], buttons[1], buttons[2], buttons[3], buttons[4]


def read_list_from_file_string(filename="questions.txt") -> str:
    lst = read_list_from_file(filename=filename)
    numbered_questions = [f"{i+1}. {lst}" for i, lst in enumerate(lst)]
    return "\n\n".join(numbered_questions)


def delete_docs(progress=gr.Progress()):
    progress(0.1, desc="deleting server side documents...")
    print(subprocess.run(["rm -rf ./assets/doc_list"], shell=True))
    progress(0.9, desc="done deleting server side documents...")
    return "done deleting server side documents..."


questions = read_list_from_file()
if len(questions) == 0:
    questions = ["What is CML?", "What is Cloudera?"]

file_types = ["pdf", "html", "txt"]

submit_btn = gr.Button("Submit")

question_reload_btn = gr.ClearButton("Update suggestions")
examples = gr.Examples(fn=read_list_from_file)

# questions_text = gr.Textbox(
#     value=read_list_from_file_string,
#     interactive=False,
#     autoscroll=True,
# )

# button0, button1, button2, button3, button4 = read_list_from_file_button()


def get_value(button):
    return button.value


# chat = gr.Blocks()
# with chat:
#     with gr.Row():

#         # question_reload_btn.click(read_list_from_file, inputs=["questions.txt"], outputs=[questions])
#     with gr.Row():
#         gr.Markdown(
#             """
#         # Hey!!
#         Click the button to get some good questions for the uploaded document!!.
#         """
#         )
#         bt = gr.Button("Get questions")
#         out = gr.Textbox()
#         bt.change(read_list_from_file, inputs=["questions.txt"], outputs=[out])
infer = gr.ChatInterface(
    fn=Infer,
    examples=examples,
    title="CML chat Bot - v2",
    chatbot=gr.Chatbot(height=700),
    multimodal=False,
    submit_btn=submit_btn,
    additional_inputs=[
        question_reload_btn,
    ],
    additional_inputs_accordion=gr.Accordion(
        label="Some possible questions", open=True
    ),
)

with infer:
    # question_reload_btn.click(
    #     read_list_from_file_button,
    #     inputs=None,
    #     outputs=[button0, button1, button2, button3, button4],
    # )
    # button0.click(get_value, None, infer.textbox)
    # button1.click(get_value, None, infer.textbox)
    # button2.click(get_value, None, infer.textbox)
    # button3.click(get_value, None, infer.textbox)
    # button4.click(get_value, None, infer.textbox)
    question_reload_btn.click(read_list_from_file, inputs=None, outputs=infer.examples)

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
        db_progress = gr.Textbox(label="Document processing status", value="None")
    with gr.Row():
        upload_button = gr.Button("Click to Upload a File")
        upload_button.click(
            upload_document_and_ingest, inputs=[documents], outputs=[db_progress]
        )

admin = gr.Blocks()
with admin:
    with gr.Row():
        admin_progress = gr.Textbox(label="Admin event status", value="None")
    with gr.Row():
        clean_up_docs = gr.Button("Click to do file cleanup")
        clean_up_docs.click(delete_docs, inputs=None, outputs=admin_progress)

demo = gr.TabbedInterface(
    interface_list=[upload, infer, admin],
    tab_names=[
        "Step 1 - Document pre-processing",
        "Step 2 - Conversation with chatbot",
        "Admin tab",
    ],
    title="CML Chat application - v2",
)


if "CML" in os.environ and os.environ["CML"] == "yes":
    demo.launch(
        show_error=True,
        debug=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
else:
    demo.launch(debug=True)
