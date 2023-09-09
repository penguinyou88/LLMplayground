import requests
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['CURL_CA_BUNDLE'] = '' # use this if you are having SSL issue
API_TOKEN = os.getenv('huggingface_token')

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": "Bearer "+API_TOKEN}

print(headers)

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)