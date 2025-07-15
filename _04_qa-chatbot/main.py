import google.generativeai as genai
from dotenv import load_dotenv #? For loading environment variables
import os #! Getting values of environment variables
import chainlit as cl

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

@cl.on_chat_start
async def handle_chat_start():
    await cl.Message(content="Hello there, I am a Gemini AI chatbot. How can I help you?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    prompt = message.content
    response =  model.generate_content(prompt)
    response_text = response.text if hasattr(response, 'text') else ""
    await cl.Message(content=response_text).send()

