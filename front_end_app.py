import gradio as gr
from ingestion import Ingest
from rag import Infer
from upload import Upload_files
import vectordb as vectordb
import os

infer = gr.ChatInterface(
    fn=Infer, 
    examples=["What is CML?", "What is Cloudera?"], 
        title="CML chat Bot", 
        chatbot=gr.Chatbot(height=700),
        multimodal=False
        )

ingest = gr.Blocks()
with ingest:
    btn = gr.Button(value="Please press to start ingestion")
    output = gr.Textbox(label="ingestion progress", max_lines=10, interactive=False)
    btn.click(Ingest, inputs=None, outputs=[output])

upload = gr.Blocks()
with upload:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf", ".html", ".txt"], file_count="multiple")
    upload_button.upload(Upload_files, upload_button, file_output)

vectorDB = gr.Blocks()
with vectorDB:
    status = gr.Button(value="Check vectorDB status")
    addCollection = gr.Button(value="Please press to add collection to the vector DB")
    resetCollection = gr.Textbox(label="", max_lines=10, interactive=False)
    status.click(vectordb.vector_db_status, inputs=None, outputs=[output])
    addCollection.click(vectordb.create_vectro_db, inputs=None, outputs=[output])
    resetCollection.click(vectordb.reset_vector_db, inputs=None, outputs=[output])

demo = gr.TabbedInterface(interface_list=[infer, upload, ingest, vectorDB], 
                tab_names=["Chat bot", "Upload files", "Data Ingestion", "vector DB operations"]
                title="CML Chat application - v2")

if "CML" in os.environ and os.environ["CML"] == "yes": 
    demo.launch(show_error=True,
                debug=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
else:
    demo.launch(debug=True)
