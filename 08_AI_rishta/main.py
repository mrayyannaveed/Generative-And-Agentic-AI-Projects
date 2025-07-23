from agents import Runner, Agent, OpenAIChatCompletionsModel,RunConfig
from openai import AsyncOpenAI
import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()
# set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("BASE_URL")

client = AsyncOpenAI(api_key=gemini_api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash" 
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

male_gender_rishta = Agent(
    name="Male Gender Rishta",
    instructions="""
    You are a rishta agent for males. When a male user requests, find a suitable female rishta.
    details
    The rishta details must include:
    - Name
    - Age
    - Location
    - Family background
    - Cast
    - Education
    - Profession
    - Marital status
    Format the output as:
    Assalam u Alaikum! Dear user, we found a rishta for you:
    Name: ...
    Age: ...
    Location: ...
    Family Background: ...
    Cast: ...
    Education: ...
    Profession: ...
    Marital Status: ...
    """,
    model=model
)

female_gender_rishta = Agent(
    name="Female Gender Rishta",
    instructions="""
    You are a rishta agent for females. When a female user requests, find a suitable male rishta.
    details
    The rishta details must include:
    - Name
    - Age
    - Location
    - Family background
    - Cast
    - Education
    - Profession
    - Marital status
    Format the output as:
    Assalam u Alaikum! Dear user, we found a rishta for you:
    Name: ...
    Age: ...
    Location: ...
    Family Background: ...
    Cast: ...
    Education: ...
    Profession: ...
    Marital Status: ...
    """,
    model=model
)

rishta_agent = Agent(
    name="AI Rishta Aunty",
    instructions="You are an AI assistant helping users find suitable rishtas. If a gender is male then hand off to male_gender_rishta, and if a gender is female then hand off to female_gender_rishta.",
    model=model,
    handoffs=[male_gender_rishta, female_gender_rishta]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="Welcome to the AI Rishta service! Please provide your details to get started.").send()

@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get('history', [])
    history.append({"role": "user", "content": message.content})
    cl.user_session.set('history', history)
    result = await Runner.run(
        rishta_agent,
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set('history', history)
    await cl.Message(content=result.final_output).send()
