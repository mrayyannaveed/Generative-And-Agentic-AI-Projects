
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.5-flash")

inventory_list = []

@function_tool
def add_data(data: str) -> str:
    inventory_list.append(data)
    return f"✅ Added '{data}' to inventory."

@function_tool
def get_data() -> str:
    if inventory_list:
        return f"📦 Current inventory: {inventory_list}"
    return "📦 Inventory is empty."

@function_tool
def update_data(old_item: str, new_item: str) -> str:
    if old_item in inventory_list:
        index = inventory_list.index(old_item)
        inventory_list[index] = new_item
        return f"🔄 Updated '{old_item}' to '{new_item}'."
    return f"⚠️ '{old_item}' not found in inventory."

@function_tool
def delete_data(item: str) -> str:
    if item in inventory_list:
        inventory_list.remove(item)
        return f"🗑️ Deleted '{item}' from inventory."
    return f"⚠️ '{item}' not found in inventory."


inventory_agent = Agent(
    name="Inventory Agent",
    instructions="""
    You are an inventory manager.

    - **Get Data Tool** → Retrieve data from the inventory. Trigger if the prompt contains "get", "show", or a product name.
    - **Add Data Tool** → Add a new product. Trigger if the prompt contains "add", "insert data", or "insert product".
    - **Update Data Tool** → Update an existing product. Trigger if the prompt contains "update data" or "update product".
    - **Delete Data Tool** → Remove a product. Trigger if the prompt contains "delete" or "remove product".

    After using any tool, always enhance the tool's output before responding to the user.
    """,
    model=model,
    tools=[add_data, get_data, update_data, delete_data]
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="I am your inventory assistant. How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get('history', [])
    history.append({'role': 'user', 'content': message.content})
    cl.user_session.set('history', history)

    result = Runner.run_streamed(starting_agent=inventory_agent, input=message.content)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await cl.Message(content=event.data.delta).send()
