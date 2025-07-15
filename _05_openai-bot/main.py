from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"    
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


backend_agent = Agent(
    name="Backend Expert",
    instructions="""You are backend development expert, You help with backend topics like APIs, satabases, authentication, server frameworks (e.g., Express.js, Django etc)
    Do Not answer frontend or UI questions.
    """
)

frontend_agent = Agent(
    name="Frontend Expert",
    instructions="""You are frontend development expert, You help with frontend topics like UI/UX using HTML, CSS, and Tailwind CSS etc.
    Do Not answer backend or server side topics.
    """
)

web_dev_agent = Agent(
    name="Web Dev Expert",
    instructions="""
    You are a generalist web developer who decides whether the question is about frontend or backend.

    If the user asks about UI, HTML, CSS, React, NextJs, etc., handoff to the frontend expert.
    If the user asks about APIs, databases, authentication, servers, backend frameworks, etc., handoff to the backend expert.
    If it's unrelated to both, politely decline.

    Do Not answer any other topics.
    """,
    handoffs=[frontend_agent, backend_agent]
)

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello From Frontend Developer. How can i help you.").send()

@cl.on_message
async def handleMessage(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content" : message.content})

    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(
        web_dev_agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    # await cl.Message(content=result.final_output).send()