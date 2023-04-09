from dotenv import load_dotenv
import os
import openai
import pinecone
from colorama import init, Fore, Style
import time
from TTS.api import TTS
from pydub import AudioSegment
from playsound import playsound


init()  # Init Colorama
load_dotenv()  # Init ENV vars


# ENV VARIABLES AND GLOBAL VARS


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
TOP_RESULTS = 5  # Max number of results to pass to GPT
RESPONSE_SAVE_PATH = "gptResponse"  # To save TTS output
ENABLE_GPT_AUDIO_RESPONSE = True  # Enhance prompts with vector information
ENABLE_VECTOR_ENHANCEMENT = True  # Use TTS to voice response back to you


if ENABLE_GPT_AUDIO_RESPONSE:
    model = TTS.list_models()[0]
    tts = TTS(model)  # Init TTS


class Gpt:
    def __init__(self, addInfo=False, promptToSpeech=False):
        openai.api_key = GPT_API_KEY
        # A boolean that toggles using vector information or not
        self.addInfo = addInfo
        self.promptToSpeech = promptToSpeech

    def chat(self, message):
        message = self.addAdditionalInfoIfApplicable(message)
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
        self.audioResponse(gptResponse)
        return gptResponse

    def addAdditionalInfoIfApplicable(self, message):
        if self.addInfo:
            additionalInput = self.getAdditionalInfo(message)
            message += additionalInput
            printSys(f"Additional information passed: {additionalInput}")
        return message

    def getEmbedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)["data"][0][
            "embedding"
        ]

    def isRepeatPrompt(self, text):
        return True if text in self.additionalInfo else False

    def getAdditionalInfo(self, message):
        # Get relevant prompts to the vector, and remove any duplicates if they exist.
        self.additionalInfo = list(dict.fromkeys(vectorSearch(message)))
        additionalString = f"\nThis numbered list is ONLY additional information that may or may not be relevant to the prompt above. Do NOT treat them as questions:\n"
        for x in range(len(self.additionalInfo)):
            additionalString += f"{x+1}. {self.additionalInfo[x]}\n"
        return additionalString

    def audioResponse(self, gptResponse):
        if self.promptToSpeech:
            saveTTS(gptResponse, RESPONSE_SAVE_PATH)
            convertWavToMp3(RESPONSE_SAVE_PATH)
            playsound(RESPONSE_SAVE_PATH + ".mp3")


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


def saveTTS(data, path):
    tts.tts_to_file(
        text=data,
        speaker=tts.speakers[0],
        language=tts.languages[0],
        file_path=path + ".wav",
    )


def convertWavToMp3(path):
    sound = AudioSegment.from_wav(path + ".wav")
    sound.export(path + ".mp3", format="mp3")


# MAIN CODE


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
db = pinecone.Index(PINECONE_INDEX_NAME)
gpt = Gpt(ENABLE_VECTOR_ENHANCEMENT, ENABLE_GPT_AUDIO_RESPONSE)


if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    printSys("Creating index...")
    pinecone.create_index(
        PINECONE_INDEX_NAME, dimension=DIMENSION, metric=METRIC, pod_type=POD_TYPE
    )


while True:
    printSys("Please enter a prompt")
    prompt = input().strip().lower()

    gpt.chat(prompt)

    if not gpt.isRepeatPrompt(prompt):
        printSys("Uploading vector...")
        vectorUpload(prompt)
