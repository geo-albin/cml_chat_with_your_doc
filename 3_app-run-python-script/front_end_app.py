import os
import gradio as gr
import subprocess
from utils.cmlllm import upload_document_and_ingest, clear_chat_engine, Infer


def read_list_from_file(filename) -> list:
    lst = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                lst.append(line.strip())  # Remove newline characters
    return lst


def delete_docs(progress=gr.Progress()):
    progress(0.1, desc="deleting server side documents...")
    print(subprocess.run(["rm -rf ./assets/doc_list"], shell=True))
    progress(0.9, desc="done deleting server side documents...")
    return "done deleting server side documents..."


questions = read_list_from_file("questions.txt")
if len(questions) == 0:
    questions = ["What is CML?", "What is Cloudera?"]

file_types = ["pdf", "html", "txt"]

submit_btn = gr.Button("Submit")

question_reload_btn = gr.Button("Refresh examples")

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
    # examples=questions,
    title="CML chat Bot - v2",
    chatbot=gr.Chatbot(height=700),
    multimodal=False,
    submit_btn=submit_btn,
    additional_inputs=[question_reload_btn],
    additional_inputs_accordion=gr.Accordion(label="Additional Inputs", open=False),
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
