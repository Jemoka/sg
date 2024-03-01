from openai import AzureOpenAI
import os

from dotenv import load_dotenv
load_dotenv()
    
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT")
)

def msgify(text):
    return [{"role": "user", "content": text}]

def prompt_for_completion(text):
    print("COMPLETION")
    # print("completion: running", text)
    params = {
        "model": "gpt-4-turbo",
        "messages": msgify(text),
        "max_tokens": 25,
        "temperature": 0.7,
        "logprobs": False,
        "n": 1,
    }

    completion = client.chat.completions.create(**params)
    return completion

def prompt_for_next(text):
    print("NTP")
    # print("ntp: running", text)
    params = {
        "model": "gpt-4-turbo",
        "messages": msgify(text),
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "n": 1,
    }

    completion = client.chat.completions.create(**params)
    return completion



