import speech_recognition as sr 
import pyttsx3 
import os 
from dotenv import load_dotenv


load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')

import openai 
openai.api_key = OPENAI_KEY

def SpeakText(record):
    engine= pyttsx3.init()
    engine.say(record)
    engine.runAndWait()


r = sr.Recognizer()

def record_text():
    while(1):
        try:
            # Use the available microphone as input 
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.2)

                print("Im Listening")

                audio =- r.listen(source)

                mytext = r.recognize_google(audio)

                return mytext
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        
        except:
            print("Unknown Error Occurred")
        
def API_Request(messages, model="gpt-3.5-turbo"):
    