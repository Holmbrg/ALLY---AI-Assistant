import os
from openai import OpenAI
import pyttsx3
import time
import sounddevice as sd
import vosk
import json
import queue
from pathlib import Path
from playsound import playsound
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pytesseract
from PIL import Image
from google.cloud import vision
import io
import logging
import warnings
import sys
import threading
import random
import pygame
from typing import Optional, Tuple, List

# ALLY - Assistive Learning and Logic Yield.

# v3 - Project: Make conversations more fluent, add stop command to stop the AI from speaking and ready to handle new input. <- Done.
# Allow the AI to make its own memory documents about different things such as shopping lists, that it can delete when no longer needed.
# Record the sentences that are the same each time, making sure the TTS function isn't abused, thereby saving money on the API. <- Done.

# The voice assistant is pretty self explanatory, especially when you see the cmd window. Talk to it and it responds, 
# it should guide you, roughly, through how parts of it works. In regards to, for example, what you have to respond to certain things.

# Troubleshooting:
# If the cmd window does not open upon click the shortcut, make sure you are using the correct shortcut,
# and otherwise, locate the .ahk files, double left click them, and try again.

tts_engine = pyttsx3.init()
openai_api_key = os.getenv('OPENAI_API_KEY')  # Add environment variable in cmd (use as admin) with "setx OPENAI_API_KEY 'Add api key here'"
client = OpenAI(api_key=openai_api_key)
conversation_log = "conversation_log_ALLY.txt"
command_log = "command_log_ALLY.txt"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
pytesseract.pytesseract.tesseract_cmd = r'C:\Programmer\Tesseract-OCR\tesseract.exe'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'C:\\ALLY - AI Assistant\\fabled-gist-429918-k7-3f2dfb1ef283.json'  # Add Google API key here, see README.txt

if not os.path.exists(conversation_log):
    with open(conversation_log, 'w') as log:
        log.write("")

if not os.path.exists(command_log):
    with open(command_log, 'w') as log:
        log.write("")

model_path = "C:\\vosk-model-small-en-us-0.15"
model = vosk.Model(model_path)
q = queue.Queue()

system_notis = "C:\\ALLY - AI Assistant\\system-notification-199277.mp3"

def transcribe_audio() -> str:
    """
    Listens to the microphone and transcribes audio input into text using the Vosk speech recognition model.
    """
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, 16000)
        playsound(system_notis)
        print("Listening...")
        silence_counter = 0

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'text' in result and result['text']:
                    print("Stopped listening.")
                    return result['text']
            else:
                silence_counter += 1
                if silence_counter > 30:  # Adjust the threshold as necessary
                    print("Silence detected, waiting for input...")
                    return ""
                # PROBLEM: This still plays the system_notis sound after silence detected, I want it to play the system notice sound only once

def callback(indata: bytes, frames: int, time, status) -> None:
    """
    Callback function for the audio stream that puts audio data into a queue.
    """
    q.put(bytes(indata))

def TTS_openai(text: str) -> Path:
    """
    Converts text to speech using OpenAI's TTS model and saves the audio to a file.
    """
    speech_file_path = Path(__file__).parent / "Speech_ALLY_AI.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=f"{text}"
    )

    with open(speech_file_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    return speech_file_path

def log_conversation(user_input: str, ai_response: str) -> None:
    """
    Logs the user input and AI response to the conversation log file.
    """
    with open(conversation_log, 'a') as log:
        log.write(f"User: {user_input}\nAI: {ai_response}\n")

def log_command(command: str) -> None:
    """
    Logs a command to the command log file.
    """
    with open(command_log, 'a') as log:
        log.write(f"{command}\n")

def refine_command(command: str) -> Optional[str]:
    """
    Refines and validates a command to ensure it makes sense and is concise.
    Returns the refined command or None if the command doesn't make sense.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Refine the following command to make sense and be concise: {command}. If the command doesn't make sense please reply 'does not make sense', nothing else. Don't change it, the command should be informational. Your output should function as a memory for an AI of what it can and can't do.",
            }
        ],
        model="gpt-4o",
    )
    refined_command = response.choices[0].message.content.strip()
    if "does not make sense" in refined_command.lower():
        return None
    return refined_command

def get_ai_response(prompt: str, relevant_history: str, command_history: str) -> str:
    """
    Generates an AI response based on the prompt, relevant conversation history, and command history.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant in a continuous speech conversation. Respond appropriately to follow-up comments and questions. You are generating the text that will be turned into speech. You can view screenshots."
            },
            {
                "role": "user",
                "content": f"Commands you are to follow:\n{command_history}\n\n Conversation:\n{relevant_history}\n User: {prompt}"
            }
        ],
        model="gpt-4o",
    )
    ai_response = response.choices[0].message.content.strip()
    return ai_response

def ask_to_print_response(response: str) -> None:
    """
    Asks the user if they want the AI's response saved to a text document and handles the response accordingly.
    """
    print_prompt = "This response might be complex. Do you want the solution in a text document? Say 'yes' or 'no'."
    speech_file_path = TTS_openai(print_prompt)
    playsound(speech_file_path)
    os.remove(speech_file_path)
    
    user_input = transcribe_audio()
    if "yes" in user_input.lower():
        file_path = Path(__file__).parent / "AI_Solution.txt"
        with open(file_path, 'w') as file:
            file.write(response)
        confirmation = f"Solution saved to {file_path}."
        # Open the file with the default text editor
        os.startfile(file_path)
    else:
        confirmation = "Solution not saved."
    
    speech_file_path = TTS_openai(confirmation)
    playsound(speech_file_path)
    os.remove(speech_file_path)
    
def is_complex_question_or_statement(input_text: str, conversation_history: str) -> bool:
    """
    Determines if the input is likely to lead to a complex response better suited for a text document.
    Returns True if complex, False otherwise.
    """
    check_prompt = f"Based on the following conversation history and the next input, determine if the input is likely to lead to a response from chatgpt better suited for a text document than said in a conversation. Here is the conversation history: {conversation_history}. Here is the input: {input_text}. If the input can be answered in a normal speech conversation, it isn't going to generate a complex response. Respond 'yes' if it requires a detailed or technical response that sounds weird when said in a speech conversation, otherwise respond 'no'."
    decision_response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": check_prompt,
            }
        ],
        model="gpt-4o",
    )
    decision = decision_response.choices[0].message.content.strip().lower()
    return "yes" in decision

def get_relevant_conversation_history(conversation_history: str, user_input: str) -> str:
    """
    Extracts relevant parts of the conversation history needed to respond to the next input.
    """
    check_prompt = f"Based on the following conversation history, determine the relevant parts that are needed to respond to the next input. Here is the conversation history: {conversation_history}. Here is the input: {user_input}. Provide only the relevant parts, in other words, please don't provide a lengthy text, only if you have to."
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": check_prompt,
            }
        ],
        model="gpt-4o",
    )
    relevant_history = response.choices[0].message.content.strip()
    return relevant_history

def access_command_log() -> str:
    """
    Reads and returns the command log content.
    """
    if os.path.exists(command_log):
        with open(command_log, 'r') as log:
            return log.read()
    return ""

def ask_screenshot(user_input: str) -> bool:
    """
    Checks if the user is asking to look at a screenshot.
    Returns True if yes, False otherwise.
    """
    check_prompt = f"Based on the following user input, is the user asking you to look at a screenshot? {user_input}. Answer 'yes' if true otherwise answer 'no', nothing else."
    decision_response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": check_prompt,
            }
        ],
        model="gpt-4o",
    )
    decision = decision_response.choices[0].message.content.strip().lower()
    return decision == 'yes'

class ScreenshotHandler(FileSystemEventHandler):
    """
    Handles the creation events of screenshot files.
    """
    def __init__(self):
        super().__init__()
        self.screenshot_filename: Optional[str] = None

    def on_created(self, event) -> None:
        """
        Called when a file is created in the monitored directory.
        """
        if event.src_path.endswith(".png"):
            self.screenshot_filename = event.src_path
            observer.stop()

def detect_screenshot_and_get_filename() -> Optional[str]:
    """
    Waits for a screenshot to be created and returns its filename.
    """
    SCREENSHOT_FOLDER = "C:\\Users\\aholm\\OneDrive\\Pictures\\Screenshots"
    event_handler = ScreenshotHandler()
    global observer
    observer = Observer()
    observer.schedule(event_handler, path=SCREENSHOT_FOLDER, recursive=False)
    observer.start()

    ai_response = "Awaiting screenshot."
    speech_file_path = TTS_openai(ai_response)
    playsound(speech_file_path)
    os.remove(speech_file_path)

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    return event_handler.screenshot_filename

def interpret_image(image_path: str) -> Tuple[str, List[str], List[float]]:
    """
    Extracts text from an image using OCR and analyzes it using Google Cloud Vision.
    Returns the extracted text, labels, and confidence scores.
    """
    ai_response = "Processing the screenshot."
    speech_file_path = TTS_openai(ai_response)
    playsound(speech_file_path)
    os.remove(speech_file_path)

    # Extract text using OCR
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    # Analyze image using Google Cloud Vision
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    vision_image = vision.Image(content=content)
    response = client.label_detection(image=vision_image)
    labels = response.label_annotations
    descriptions = [label.description for label in labels]
    confidence_scores = [label.score for label in labels]

    result = (extracted_text, descriptions, confidence_scores)
    return result

def Img_response(cleaned_text: str, confidences: List[str], relevant_history: str, command_history: str) -> str:
    """
    Generates an AI response specifically for images, based on extracted text and confidences.
    """
    prompt = (
        "Here is some text extracted from an image. Respond as if you are reacting to the text. "
        "Identify and solve any tasks or problems mentioned. Respond in the same language as the text. "
        "Do not ask for additional input, the user cannot respond."
    )
    combined_input = cleaned_text + "\n" + "\n".join(confidences) + "\n" + prompt
    history_input = f'Heres your conversation history, keep this in mind: {relevant_history} and here are some commands you are to follow: {command_history}'
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": combined_input + history_input,
            }
        ],
        model="gpt-4o",
    )
    response = response.choices[0].message.content.strip()
    return response

def main() -> None:
    """
    Main program loop that activates the assistant and handles user interactions.
    """
    print("Assistant activated.")
    ai_response = "Assistant activated. Say 'quit' to exit or 'clear' to clear the conversation log."
    speech_file_path = TTS_openai(ai_response)
    playsound(speech_file_path)
    os.remove(speech_file_path)

    while True:
        user_input = transcribe_audio().strip().lower()
        if user_input:
            if ask_screenshot(user_input) == True:
                screenshot_filename = detect_screenshot_and_get_filename()
                if screenshot_filename:
                    text, descriptions, scores = interpret_image(screenshot_filename)
                    cleaned_text = text.strip().replace("\n", " ").replace("  ", " ")
                    confidences = [f"{desc} (confidence: {score})" for desc, score in zip(descriptions, scores)]

                    conversation_history = ""
                    if os.path.exists(conversation_log):
                        with open(conversation_log, "r") as log:
                            conversation_history = log.read()

                    command_history = access_command_log()
                    relevant_history = get_relevant_conversation_history(conversation_history, user_input)

                    if is_complex_question_or_statement(cleaned_text, relevant_history):
                        ai_response = "Do you want the response in a text document? Say 'yes' or 'no'."
                        speech_file_path = TTS_openai(ai_response)
                        playsound(speech_file_path)
                        os.remove(speech_file_path)

                        user_input_response = transcribe_audio().strip().lower()
                        if user_input_response == "yes":
                            ai_response = Img_response(cleaned_text, confidences, relevant_history, command_history)
                            log_conversation(user_input, ai_response)
                            file_path = Path(__file__).parent / "AI_Solution.txt"
                            with open(file_path, 'w') as file:
                                file.write(ai_response)
                            confirmation = f"Solution saved."
                            os.startfile(file_path)
                            speech_file_path = TTS_openai(confirmation)
                            playsound(speech_file_path)
                            os.remove(speech_file_path)
                        else:
                            ai_response = Img_response(cleaned_text, confidences, relevant_history, command_history)
                            log_conversation(user_input, ai_response)
                            speech_file_path = TTS_openai(ai_response)
                            playsound(speech_file_path)
                            os.remove(speech_file_path)
                    else:
                        ai_response = Img_response(cleaned_text, confidences, relevant_history, command_history)
                        log_conversation(user_input, ai_response)
                        speech_file_path = TTS_openai(ai_response)
                        playsound(speech_file_path)
                        os.remove(speech_file_path)

            elif user_input == "quit":
                ai_response = "Clearing conversation logs and powering down."
                speech_file_path = TTS_openai(ai_response)
                playsound(speech_file_path)
                os.remove(speech_file_path)
                print("Quitting the assistant.")
                os.remove(conversation_log)
                break
            elif user_input == "clear":
                ai_response = "Clearing the conversation log."
                speech_file_path = TTS_openai(ai_response)
                playsound(speech_file_path)
                os.remove(speech_file_path)
                with open(conversation_log, 'w') as log:
                    log.write("")
                print("Conversation log cleared.")
            elif user_input == "command":
                ai_response = "Please state your command."
                speech_file_path = TTS_openai(ai_response)
                playsound(speech_file_path)
                os.remove(speech_file_path)

                command = transcribe_audio().strip().lower()
                refined_command = refine_command(command)
                if refined_command:
                    log_command(refined_command)
                    ai_response = f"Command added."
                    speech_file_path = TTS_openai(ai_response)
                    playsound(speech_file_path)
                    os.remove(speech_file_path)
                else:
                    ai_response = "The command does not make sense and was not added."
                    speech_file_path = TTS_openai(ai_response)
                    playsound(speech_file_path)
                    os.remove(speech_file_path)
            else:
                conversation_history = ""
                if os.path.exists(conversation_log):
                    with open(conversation_log, "r") as log:
                        conversation_history = log.read()

                command_history = access_command_log()
                relevant_history = get_relevant_conversation_history(conversation_history, user_input)

                if is_complex_question_or_statement(user_input, relevant_history):
                    ai_response = "Do you want the response in a text document? Say 'yes' or 'no'."
                    speech_file_path = TTS_openai(ai_response)
                    playsound(speech_file_path)
                    os.remove(speech_file_path)

                    user_input_response = transcribe_audio().strip().lower()
                    if user_input_response == "yes":
                        ai_response = get_ai_response(user_input, relevant_history, command_history)
                        log_conversation(user_input, ai_response)
                        file_path = Path(__file__).parent / "AI_Solution.txt"
                        with open(file_path, 'w') as file:
                            file.write(ai_response)
                        confirmation = f"Solution saved."
                        os.startfile(file_path)
                        speech_file_path = TTS_openai(confirmation)
                        playsound(speech_file_path)
                        os.remove(speech_file_path)
                    else:
                        ai_response = get_ai_response(user_input, relevant_history, command_history)
                        log_conversation(user_input, ai_response)
                        speech_file_path = TTS_openai(ai_response)
                        playsound(speech_file_path)
                        os.remove(speech_file_path)
                else:
                    ai_response = get_ai_response(user_input, relevant_history, command_history)
                    log_conversation(user_input, ai_response)
                    speech_file_path = TTS_openai(ai_response)
                    playsound(speech_file_path)
                    os.remove(speech_file_path)

            time.sleep(2)  # Prevents double triggering

        time.sleep(0.1)  # Adjust as necessary to prevent high CPU usage

if __name__ == "__main__":
    main()
