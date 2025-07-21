import asyncio
from agents import Agent, RunConfig, Runner, set_tracing_disabled, OpenAIChatCompletionsModel 
from openai import AsyncOpenAI  
import os
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(disabled=True)
# Replace with your actual API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(openai_client=client, model=GEMINI_MODEL)


male_agent = Agent(
    name="Male Rishta Assistant",
    instructions="""
        Find a suitable female rishta for the user.
        Provide only rishta details with same age or difference of 2 years not more than.
        The rishta includes details: name, age, profession, location, cast, etc.
    """
)

female_agent = Agent(  
    name="Female Rishta Assistant",
    instructions="""
        Find a suitable male rishta for the user.
        Provide only rishta details with same age or difference of 2 years not more than.
        The rishta includes details: name, age, salary, profession, location, cast, etc.
    """
)

rishta_agent = Agent(
    name="Rishta Assistant",
    instructions="""
        This is an AI rishta assistant.
        If the rishta is for a male, then hand off to the male agent.
        If the rishta is for a female, then hand off to the female agent.
    """,
    handoffs= [female_agent, male_agent]
    )


# Main async function
async def main():
    result = await Runner.run(
        rishta_agent,
        'I\'m looking for a suitable partner. This is my details: name: "Rayyan", age: 25, profession: "Student", location: "Karachi", cast: "Urdu Speaking".',
        run_config=RunConfig(model=model)
    )
    print(result.final_output)
    print(result.last_agent.name)

# Entry point
def start():
    asyncio.run(main())