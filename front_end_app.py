import gradio as gr

from cmlllm import upload_document_and_ingest, clear_chat_engine
from cmlllm import Infer

import vectordb as vectordb
import os

def read_list_from_file(filename):
    lst = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                lst.append(line.strip())  # Remove newline characters
    return lst

questions = read_list_from_file("questions.txt")
if len(questions) == 0:
    questions = ["What is CML?", "What is Cloudera?"]

file_types=["pdf", "html", "txt"]


questions_state = gr.State(questions)
submit_btn = gr.Button("Submit")


infer = gr.ChatInterface(
        fn=Infer, 
        examples=questions_state.value, 
        title="CML chat Bot - v2", 
        chatbot=gr.Chatbot(height=700),
        multimodal=False,
        submit_btn=submit_btn,
        )

def set_upload_in_progess():
    submit_btn.interactive = False
    submit_btn.value = "File upload in progress"
    print("Albin : set_upload_in_progess")
    infer.render()
    
def clear_upload_in_progess():
    submit_btn.interactive=True
    submit_btn.value = "Submit" 
    print("Albin : clear_upload_in_progess")
    infer.render()

def upload_document_and_ingest_local(upload_button):
    set_upload_in_progess()
    upload_document_and_ingest(upload_button)

upload = gr.Blocks()
with upload:
    with gr.Row():
        documents = gr.Files(height=100, file_count="multiple", file_types=file_types, interactive=True, label="Upload your pdf, html or text documents (single or multiple)")
    with gr.Row():
        db_progress = gr.Textbox(label="Document processing status", value="None")
    with gr.Row():
        upload_button = gr.UploadButton("Click to Upload a File", file_types=file_types, file_count="multiple")     
        upload_button.upload(upload_document_and_ingest_local, inputs=[upload_button], outputs=[db_progress, questions_state]).then(clear_upload_in_progess, None, None)
    



demo = gr.TabbedInterface(
                interface_list=[upload, infer], 
                tab_names=["Step 1 - Document pre-processing", "Step 2 - Conversation with chatbot"],
                title="CML Chat application - v2")


if "CML" in os.environ and os.environ["CML"] == "yes": 
    demo.launch(show_error=True,
                debug=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
else:
    demo.launch(debug=True)

