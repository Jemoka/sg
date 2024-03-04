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
    
    # tmp = client.completions.create(
    #     model="gpt-4-large",
    #     prompt="Say this is a test",
    #     max_tokens=7,
    #     temperature=0
    # )
    # breakpoint()
# print("completion: running", text)
    params = {
        "model": "gpt-35-turbo",
        "prompt": text,
        "max_tokens": 25,
        "temperature": 0.7,
        "n": 1,
    }

    completion = client.completions.create(**params)
    return completion

def prompt_for_next(text):
    print("NTP")
    # print("ntp: running", text)
    params = {
        "model": "gpt-35-turbo",
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
        "n": 1,
    }

    completion = client.completions.create(**params)
    return completion



