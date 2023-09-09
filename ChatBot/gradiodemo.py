'''
Adpated from this example: https://huggingface.co/spaces/fffiloni/langchain-chat-with-pdf/blob/main/app.py
'''

import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, repo_id):
    
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":250})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
    result = qa({"query": query})

    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
    <a style="display:inline-block; margin-left: 1em" href="https://huggingface.co/spaces/fffiloni/langchain-chat-with-pdf?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space%20to%20skip%20the%20queue-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column():
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            repo_id = gr.Dropdown(label="LLM", choices=["google/flan-ul2", "OpenAssistant/oasst-sft-1-pythia-12b", "bigscience/bloomz"], value="google/flan-ul2")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message")
    #load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()