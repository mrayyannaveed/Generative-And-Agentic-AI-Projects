import chainlit as cl
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

@cl.on_message
async def main(message: cl.Message):
    prompt = message.content

    if prompt:
        try:
            response = requests.post(
                url=BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )

            data = response.json()

            if "choices" in data:
                reply = data["choices"][0]["message"]["content"]
                await cl.Message(content=reply).send()
            else:
                cl.Message(content=f"Unexpected API response format:\n\n{json.dumps(data, indent=2)}").send()
            
        except Exception as e:
            cl.Message(content=f"❌ API call failed: {e}").send()