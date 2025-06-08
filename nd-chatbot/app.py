import chainlit as cl
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-")


@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        # model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": message.content,
            }
        ],
    )
    await cl.Message(
        content=response.choices[0].message.content,
    ).send()
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Hello there, I am a bot",
    ).send()

@cl.set_starters
async def set_starters():
    return[
        cl.Starter(
            label="Refine my email",
            message="Can you correct and refine my email",
            icon="📧"
        ),
        cl.Starter(
            label="Summarize my email",
            message="Can you summarize my email",
            icon="📧"
        ),
        cl.Starter(
            label="Suggest a headline",
            message="Can you suggest a headline for my email",
            icon="📧"
        ),
        cl.Starter(
            label="Debug a code",
            message="Can you debug a code of my webapp",
            icon="📧"
        )
    ]