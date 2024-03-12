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
    # print("COMPLETION")
    
    params = {
        "model": "gpt-35-turbo",
        "prompt": text,
        "max_tokens": 500,
        "temperature": 0.7,
        "n": 1,
    }

    completion = client.completions.create(**params)
    return completion.choices[0].text
    
    # params = {
    #     "model": "gpt-4",
    #     "messages": msgify(text),
    #     "max_tokens": 500,
    #     "temperature": 0.7,
    #     "n": 1,
    # }

    # completion = client.chat.completions.create(**params)
    # return completion.choices[0].message.content.strip()


def prompt_for_next(text):
    # print("NTP")

    # params = {
    #     "model": "gpt-35-turbo",
    #     "prompt": text,
    #     "max_tokens": 1,
    #     "temperature": 0,
    #     "logprobs": 5,
    #     "n": 1,
    # }

    # completion = client.completions.create(**params)
    # return completion.choices[0].logprobs.top_logprobs[0]

    params = {
        "model": "gpt-4",
        "messages": msgify(text),
        "max_tokens": 1,
        "temperature": 0,
        "n": 1,
    }

    completion = client.chat.completions.create(**params)
    return {
        completion.choices[0].message.content.strip(): -1.0
    }

# tmp = prompt_for_next("test")
# tmp


# params = {
#     "model": "gpt-4",
#     "messages": msgify("test"),
#     "max_tokens": 1,
#     "temperature": 0,
#     "logprobs": True,
#     "n": 1,
# }

# completion = client.chat.completions.create(**params)
