from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import chainlit as cl

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
GEMINI_MODEL = "Gemini-2.0-flash"

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)

model = OpenAIChatCompletionsModel(openai_client=client, model=GEMINI_MODEL)

cooking_idea_agent = Agent(
    name="Cooking Idea Agent",
    description="You are a cooking idea agent. You will be given a list of ingredients and you will return a cooking idea.",
    model=model,
    handoffs=["Chef Agent"]
)

chef_agent = Agent(
    name="Chef Agent",
    description="You are a chef agent. You will be given a cooking idea and you will return a recipe.",
    model=model,
)

cooking_agent = Agent(
    name="Cooking Agent",
    description="You are a cooking agent. If user want a cooking idea, you will hand offs it to Cooking Idea Agent. And if user wants to create a recipe, you will hand off to Chef Agent.",
    model=model,
    handoffs=["Cooking Idea Agent", "Chef Agent"]
)


@cl.on_message
async def handle_message(message: cl.Message):
    response = await Runner.run(
        cooking_agent,
        input=message.content
    )
    await cl.Message(content=response.final_output).send()