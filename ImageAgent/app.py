import gradio as gr
from langchain.agents import initialize_agent
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import os
from dotenv import load_dotenv
from ImageAgent import ImageCaptionTool, ObjectDetectionTool

# set openAI access
load_dotenv()
os.environ['CURL_CA_BUNDLE'] = '' # use this if you are having SSL issue
model_name = os.getenv('OPENAI_MODEL')
openai_api_base = os.getenv('OPENAI_API_BASE')
openai_api_key = os.getenv('API_KEY')
openai_api_version = os.getenv('OPENAI_API_VERSION')

#initialize the agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# use azurechatopenAI
llm = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.7,
    deployment_name=model_name,
    openai_api_base=openai_api_base,
    openai_api_version=openai_api_version)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)


def run_task(query, image_file):
    result = agent.run(f'{query}, this is the image: {image_file}')
    return result

# main UI


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Chat with your Image
    Upload an image to get started!
    """
    )
    image_file = gr.Image(label='image_path', type='filepath')
    with gr.Row():
        inp = gr.Textbox(label="Question", placeholder="Type your questions here")
        out = gr.Textbox(label="Response", placeholder="")
    submit_btn = gr.Button("Submit")
    submit_btn.click(run_task, inputs=[inp, image_file], outputs=out)
    examples = gr.Examples(examples=["Generate a caption for this image?", "What objects are present in the image?"], inputs=[inp])

demo.launch()

# # UI
# gr.Interface(fn=run_task, 
#              inputs=[gr.Textbox(label="Question", placeholder="Type your questions here"), gr.Image(type='filepath')],
#              outputs=gr.Textbox(label="Response", placeholder="")).launch()

# # main UI part
# with gr.Blocks(css=css) as demo:
#     with gr.Column(elem_id="col-container"):
#         gr.HTML(title)      
#         with gr.Column():
#             image_file = gr.Image(type="pil", label='image'),
#             tasks = gr.Dropdown(label="tasks", choices=["Generate Caption", "Object Detection"], value="Generate Caption")
#             with gr.Row():
#                 question = gr.Textbox(label="Question", placeholder="Type your questions here", interactive=False)
#                 submit_btn = gr.Button("Generate Answer")
#         response = gr.Textbox(label="Response", placeholder="")

#         # chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
#         # question = gr.Textbox(label="Question", placeholder="Type your questions here")
#         # submit_btn = gr.Button("Send message")

#     # callbacks 
#     # tasks.change(image_change, inputs=[image_file], outputs=[langchain_status], queue=False)
#     submit_btn.click(run_task, inputs=[question, image_file], outputs=response)
#     examples = gr.Examples(examples=["Generate a caption for this image?", "Describe the image"],inputs=[question])
#     # question.submit(add_text, [chatbot, question], [chatbot, question])
#     # submit_btn.click(run_task, inputs=[question, image_file], outputs=[chatbot])

# demo.launch()