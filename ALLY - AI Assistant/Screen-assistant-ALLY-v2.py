import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pytesseract
from PIL import Image
from google.cloud import vision
import io
import logging
import warnings
from openai import OpenAI
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading

# ALLY - Assistive Learning and Logic Yield

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
pytesseract.pytesseract.tesseract_cmd = r'C:\Programmer\Tesseract-OCR\tesseract.exe'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'c:\\ALLY - AI Assistant\\fabled-gist-429918-k7-3f2dfb1ef283.json'

class ScreenshotHandler(FileSystemEventHandler):
    def __init__(self):
        self.screenshot_filename = None

    def on_created(self, event):
        if event.src_path.endswith(".png"):
            self.screenshot_filename = event.src_path
            observer.stop()

def loading(message):
    root = tk.Tk()
    root.withdraw()

    loading_box = tk.Toplevel(root)
    loading_box.title("Loading")
    loading_box.geometry("+50+50")

    label = tk.Label(loading_box, text=f"{message}", font=("Helvetica", 12))
    label.pack(pady=10, padx=10)

    return loading_box, root

def detect_screenshot_and_get_filename():
    SCREENSHOT_FOLDER = "C:\\Users\\aholm\\OneDrive\\Pictures\\Screenshots"

    event_handler = ScreenshotHandler()
    observer.schedule(event_handler, path=SCREENSHOT_FOLDER, recursive=False)
    observer.start()

    program_loading = 'Loading...'
    dura_program = 4
    loading_box, root = loading(program_loading)

    def stop_loading_box():
        time.sleep(dura_program)
        loading_box.destroy()
        root.quit()

    threading.Thread(target=stop_loading_box).start()

    root.mainloop()

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    return event_handler.screenshot_filename

def interpret_image(image_path):
    loading_box, root = loading("Interpreting image, please wait...")

    def process_image():
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
        root.after(0, lambda: loading_box.destroy())
        root.quit()
        return result

    result = [None]
    def target():
        result[0] = process_image()

    thread = threading.Thread(target=target)
    thread.start()
    root.mainloop()
    thread.join()

    return result[0]

def AI_response(cleaned_text, confidences):
    loading_box, root = loading("Generating AI response, please wait...")

    def generate_response():
        prompt = (
            "Here is some text extracted from an image. Respond as if you are reacting to the text. "
            "Identify and solve any tasks or problems mentioned. Respond in the same language as the text. "
            "Do not ask for additional input, the user cannot respond."
        )
        combined_input = cleaned_text + "\n" + "\n".join(confidences) + "\n" + prompt
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": combined_input,
                }
            ],
            model="gpt-4o",
        )
        response_text = response.choices[0].message.content.strip()

        result[0] = response_text
        root.after(0, lambda: loading_box.destroy())
        root.quit()

    result = [None]
    thread = threading.Thread(target=generate_response)
    thread.start()
    root.mainloop()
    thread.join()

    return result[0]

def save_and_open_response(response_text):
    with open("AI_help.txt", "w") as f:
        f.write(response_text)
    os.system("notepad.exe AI_help.txt")

if __name__ == "__main__":
    observer = Observer()
    screenshot_file = detect_screenshot_and_get_filename()
    if screenshot_file:
        text, descriptions, scores = interpret_image(screenshot_file)
        cleaned_text = text.strip().replace("\n", " ").replace("  ", " ")
        confidences = [f"{desc} (confidence: {score})" for desc, score in zip(descriptions, scores)]
        AI_help = AI_response(cleaned_text, confidences)
        save_and_open_response(AI_help)
        sys.exit()
