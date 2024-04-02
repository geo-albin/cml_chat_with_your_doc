import gradio as gr
from ingestion import Ingest
from rag import Infer
from upload import Upload_files
import vectordb as vectordb

infer = gr.ChatInterface(
    fn=Infer, 
    examples=["What is llama index?", "What is RAG?"], 
        title="llama index chat Bot", 
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
    startVectorDB = gr.Button(value="Please press to start vector DB")
    stopVectorDB = gr.Button(value="Please press to stop vector DB")
    addCollection = gr.Button(value="Please press to add collection to the vector DB")
    output = gr.Textbox(label="", max_lines=10, interactive=False)
    startVectorDB.click(vectordb.start_vector_db, inputs=None, outputs=[output])
    addCollection.click(vectordb.create_vectro_db, inputs=None, outputs=[output])
    stopVectorDB.click(vectordb.stop_vector_db, inputs=None, outputs=[output])

demo = gr.TabbedInterface([infer, ingest, upload, vectorDB], ["Chat bot", "Data Ingestion", "Upload files", "vector DB operations"])
demo.launch(debug=True)