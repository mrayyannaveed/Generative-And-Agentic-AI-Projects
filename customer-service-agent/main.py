from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, WebSearchTool, function_tool, set_default_openai_client, set_default_openai_api
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import chainlit as cl
import requests

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
GEMINI_MODEL = "gemini-2.0-flash"

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
set_default_openai_client(client)
set_default_openai_api('chat_completions')
model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.0-flash")

# @function_tool
# async def show_products():
#     """Show products that user wants"""
#     products = requests.get(f"https://hackathon-apis.vercel.app/api/products")
#     products = products.json()
#     return products

shoe_products = Agent(
    name="Shoe Products",
    instructions="""You are a shoe products agent. You are to show products that user wants. You can search on web for the required produc then provide details.""",
    model=model,
)

customer_support_agent = Agent(
    name="Customer Support Agent",
    instructions="""""You are a customer support agent for a e-commerce website.
    If user wants to see products, you can use the shoe_products tool.
    You are to help customers with their questions.
    Decline questions other than car-related inquiries.""",
    model=model,
    tools=[shoe_products.as_tool(tool_name="show_shoe_products", tool_description="Show products that user wants")]
) 

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="Hello! I am your customer support agent. How can I help you?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get('history', [])
    history.append({'role': 'user', 'content': message.content})
    cl.user_session.set('history', history)

    msg = cl.Message(content="🔃")
    await msg.send()

    result = await Runner.run(
        customer_support_agent,
        input=history
    )
    history.append({'role': 'assistant', 'content': result.final_output})
    cl.user_session.set('history', history)
    await cl.Message(content=result.final_output).send()
