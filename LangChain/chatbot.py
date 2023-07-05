import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from langchain import OpenAI, Wikipedia
from langchain.agents.react.base import DocstoreExplorer

# load openAI access keys
# os.environ['CURL_CA_BUNDLE'] = '' # use this if you are having SSL issue
load_dotenv()
model_name = os.getenv('OPENAI_MODEL')
openai_api_base = os.getenv('OPENAI_API_BASE')
openai_api_key = os.getenv('API_KEY')
openai_api_version = os.getenv('OPENAI_API_VERSION')

# set prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# use azurechatopenAI
llm = AzureChatOpenAI(
    openai_api_key= openai_api_key,
    temperature=0.7,
    deployment_name=model_name,
    openai_api_base = openai_api_base,
    openai_api_version=openai_api_version)

memory = ConversationBufferMemory(return_messages=True)
# conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
# print(conversation.predict(input="Hi there!"))

# set up tools
docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

agent = initialize_agent(
    agent=AgentType.REACT_DOCSTORE,
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=memory,
    early_stopping_method='generate'
)

response = agent.run("Hi there! What is large language models?")
print(response)