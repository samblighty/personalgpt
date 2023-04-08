from dotenv import load_dotenv
import os
import openai
import pinecone
from colorama import init, Fore, Style

init()  # Init Colorama
load_dotenv()  # Init ENV vars

GPT_API_KEY = os.getenv("GPT_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DIMENSION = 1024
METRIC = "cosine"
POD_TYPE = "p2"
GPT_MODEL = "gpt-3.5-turbo"
GPT_TEMPERATURE = 0.8
EMBEDDING_MODEL = "text-embedding-ada-002"

if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        PINECONE_INDEX_NAME, dimension=DIMENSION, metric=METRIC, pod_type=POD_TYPE
    )


def printSys(msg):
    print(f"{Fore.GREEN}SYSTEM:{Style.RESET_ALL} {msg}")


def printChat(msg):
    print(f"{Fore.CYAN}GPT:{Style.RESET_ALL} {msg}")


class Gpt:
    def __init__(self):
        openai.api_key = GPT_API_KEY

    def chat(self, message):
        finalMessage = message + self.getAdditionalInfo()
        printSys(f"Asking GPT about: {finalMessage}")
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": finalMessage,
                }
            ],
            temperature=GPT_TEMPERATURE,
        )
        gptResponse = response.choices[0].message.content.strip()
        printChat(gptResponse)
        return gptResponse

    def getEmbedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)["data"][0][
            "embedding"
        ]

    def getAdditionalInfo(self):
        return " "


db = pinecone.Index(PINECONE_INDEX_NAME)
gpt = Gpt()

while True:
    printSys("Please enter a prompt")
    prompt = input().strip()
    gpt.chat(prompt)
