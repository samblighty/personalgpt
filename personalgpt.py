from dotenv import load_dotenv
import os
import openai
import pinecone
from colorama import init, Fore, Style
import time

init()  # Init Colorama
load_dotenv()  # Init ENV vars

GPT_API_KEY = os.getenv("GPT_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

PINECONE_ENV = "us-east1-gcp"
DIMENSION = 1536
METRIC = "cosine"
POD_TYPE = "p2"
GPT_MODEL = "gpt-3.5-turbo"
GPT_TEMPERATURE = 0.8
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_RESULTS = 5


class Gpt:
    def __init__(self):
        openai.api_key = GPT_API_KEY

    def chat(self, message):
        additionalInput = self.getAdditionalInfo(message)
        message += additionalInput
        printSys(f"Additional information passed: {additionalInput}")
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": message,
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

    def getAdditionalInfo(self, message):
        additionalInfo = vectorSearch(message)
        additionalString = (
            f"\nThis list of information may be relevant in answering the prompt: \n"
        )
        for info in additionalInfo:
            additionalString += f"{info}\n"
        return additionalString


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
db = pinecone.Index(PINECONE_INDEX_NAME)
gpt = Gpt()


if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        PINECONE_INDEX_NAME, dimension=DIMENSION, metric=METRIC, pod_type=POD_TYPE
    )


def printSys(msg):
    print(f"{Fore.GREEN}SYSTEM:{Style.RESET_ALL} {msg}")


def printChat(msg):
    print(f"{Fore.CYAN}GPT:{Style.RESET_ALL} {msg}")


def vectorUpload(data):
    vector = gpt.getEmbedding(data)
    db.upsert([(f"gpt_{time.time()}", vector, {"prompt": data})])


def vectorSearch(data):
    vector = gpt.getEmbedding(data)
    results = db.query(vector, top_k=TOP_RESULTS, include_metadata=True)
    topResults = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["prompt"])) for item in topResults]


while True:
    printSys("Please enter a prompt")
    prompt = input().strip()

    gpt.chat(prompt)

    vectorUpload(prompt)
