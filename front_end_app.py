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


# infer = gr.ChatInterface(
#     fn=Infer, 
#     examples=lst, 
#         title="CML chat Bot", 
#         chatbot=gr.Chatbot(height=700),
#         multimodal=False
#         )

# ingest = gr.Blocks()
# with ingest:
#     btn = gr.Button(value="Please press to start ingestion")
#     output = gr.Textbox(label="ingestion progress", interactive=True)
#     btn.click(Ingest, inputs=None, outputs=[output])

# upload = gr.Blocks()
# with upload:
#     file_output = gr.File()
#     upload_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf", ".html", ".txt"], file_count="multiple")
#     upload_button.upload(Upload_files, upload_button, file_output)

# vectorDB = gr.Blocks()
# with vectorDB:
#     status = gr.Button(value="Check vectorDB status")
#     addCollection = gr.Button(value="Please press to add collection to the vector DB")
#     resetCollection = gr.Button(value="Please press to reset the vector DB")
#     output = gr.Textbox(label="", max_lines=10, interactive=False)
#     status.click(vectordb.vector_db_status, inputs=None, outputs=[output])
#     addCollection.click(vectordb.create_vector_db, inputs=None, outputs=[output])
#     resetCollection.click(vectordb.reset_vector_db, inputs=None, outputs=[output])

# demo = gr.TabbedInterface(interface_list=[infer, upload, ingest, vectorDB], 
#                 tab_names=["Chat bot", "Upload files", "Data Ingestion", "vector DB operations"],
#                 title="CML Chat application - v2")

# if "CML" in os.environ and os.environ["CML"] == "yes": 
#     demo.launch(show_error=True,
#                 debug=True,
#                 server_name='127.0.0.1',
#                 server_port=int(os.getenv('CDSW_APP_PORT')))
# else:
#     demo.launch(debug=True)


def demo():
    with gr.Blocks(theme="base") as demo:
        questions_state = gr.State()
        questions_state = questions
        gr.Markdown(
        """<center><h2>CML - Chatbot v2</center></h2>""")
        with gr.Tab("Step 1 - Document pre-processing"):
            with gr.Row():
                documents = gr.Files(height=100, file_count="multiple", file_types=file_types, interactive=True, label="Upload your pdf, html or text documents (single or multiple)")
            with gr.Row():
                db_progress = gr.Textbox(label="Document processing status", value="None")
            with gr.Row():
                # doc_btn = gr.Button("Process the documents")
                # doc_btn.click(upload_document_and_ingest, \
                #     inputs=[documents], \
                #     outputs=[questions_state, db_progress])
                upload_button = gr.UploadButton("Click to Upload a File", file_types=file_types, file_count="multiple")
            
        # with gr.Tab("Step 2 - QA chain initialization"):
        #     with gr.Row():
        #         llm_btn = gr.Radio(list_llm_simple, \
        #             label="LLM models", value = list_llm_simple[0], type="index", info="Choose your LLM model")
        #     with gr.Accordion("Advanced options - LLM model", open=False):
        #         with gr.Row():
        #             slider_temperature = gr.Slider(minimum = 0.0, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Model temperature", interactive=True)
        #         with gr.Row():
        #             slider_maxtokens = gr.Slider(minimum = 224, maximum = 4096, value=1024, step=32, label="Max Tokens", info="Model max tokens", interactive=True)
        #         with gr.Row():
        #             slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k samples", info="Model top-k samples", interactive=True)
        #     with gr.Row():
        #         llm_progress = gr.Textbox(value="None",label="QA chain initialization")
        #     with gr.Row():
        #         qachain_btn = gr.Button("Initialize question-answering chain...")

        with gr.Tab("Step 2 - Conversation with chatbot"):
            # clear_btn=gr.ClearButton("🗑️  Clear")
            # bot = gr.Chatbot(render=False, height=300)
            # clear_btn.add(bot)
            infer = gr.ChatInterface(
                fn=Infer, 
                examples=questions_state, 
                title="CML chat Bot")
        
        upload_button.upload(upload_document_and_ingest, 
                            inputs=[documents], 
                            outputs=[questions_state, db_progress])
        # Chatbot events
        # clear_btn.click(clear_chat_engine, \
        #     inputs=None, \
        #     outputs=None, \
        #     queue=False)
    demo.queue()

    if "CML" in os.environ and os.environ["CML"] == "yes": 
        demo.launch(show_error=True,
                    debug=True,
                    server_name='127.0.0.1',
                    server_port=int(os.getenv('CDSW_APP_PORT')))
    else:
        demo.launch(debug=True)


if __name__ == "__main__":
    demo()
