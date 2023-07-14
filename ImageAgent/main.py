'''
Code adapted from https://nayakpplaban.medium.com/ask-questions-to-your-images-using-langchain-and-python-1aeb30f38751
'''
# import openai
# from getpass import getpass

import os
from dotenv import load_dotenv
# from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
# from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from ImageAgent import ImageCaptionTool, ObjectDetectionTool

# set openAI access
load_dotenv()
os.environ['CURL_CA_BUNDLE'] = '' # use this if you are having SSL issue
# openai.api_type = os.getenv('API_TYPE')
# openai.api_base = os.getenv('OPENAI_API_BASE')
# openai.api_version = os.getenv('OPENAI_API_VERSION') # "2023-05-15"
# openai.api_key = os.getenv('API_KEY')
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

# use openai chat model
# llm = ChatOpenAI(
#     openai_api_key= openai_api_key,
#     temperature=0,
#     model_name=model_name
# )

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

# Test: Q1
image_path = "./test/Parsons_PR.jpg"
user_question = "generate a caption for this image?"
response = agent.run(f'{user_question}, this is the image path: {image_path}')
print(response)

# Test: Q2
# user_question = "Please tell me what are the items present in the image."
# response = agent.run(f'{user_question}, this is the image path: {image_path}')
# print(response)

'''
>Question 1 Entering new AgentExecutor chain...
{
    "action": "Image captioner",
    "action_input": "/content/Parsons_PR.jpg"
}
Observation: cars are driving down the street in traffic at a green light
Thought:{
    "action": "Final Answer",
    "action_input": "The image shows cars driving down the street in traffic at a green light."
}

> Finished chain.
response: The image shows cars driving down the street in traffic at a green light.
'''