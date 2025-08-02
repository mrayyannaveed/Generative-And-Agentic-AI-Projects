from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import chainlit as cl
import requests
from agents import input_guardrail, output_guardrail, RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput,InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from pydantic import BaseModel

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
GEMINI_MODEL = "gemini-2.0-flash"

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL
)
model = OpenAIChatCompletionsModel(openai_client=client, model=GEMINI_MODEL)

@function_tool
def get_weather_info(city: str) -> str:
    """
    Get the current weather information for a given city.
    """
    try:    
        result = requests.get(f"https://api.weatherapi.com/v1/current.json?key=51fe7d6f832c480b96725004252507&q={city}")

        data = result.json()

        return f"""
        The weather in {city}, {data['location']['country']} is currently
        {data['current']['condition']['text']} with a temperature {data['current']['temp_c']}°C ({data['current']['temp_f']}°F).
        The wind is coming from the {data['current']['wind_dir']} at {data['current']['wind_kph']} kph.
        The humidity is {data['current']['humidity']}%.
        """
    except Exception as e:
        return f"""Error fetching weather data for {city}: {e}"""
    
class OutputGuard(BaseModel):
    is_weather_related: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="weather input guardrail",
    instructions="""
    You are a input guardrail agent. If user query is related to weather of any city, return true. Otherwise, politely inform the user that you can only assist with weather-related queries. 
    """,
    model=model,
    output_type=OutputGuard
)

@input_guardrail
async def input_guardrail_func(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
)-> GuardrailFunctionOutput:
    
    result = await Runner.run(
        input_guardrail_agent,
        input=input
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_weather_related
    )

class MessageOutput(BaseModel):
    content: str

class WeatherOutput(BaseModel):
    is_weather: bool
    reasoning: str

output_guardrail_agent = Agent(
    name="weather output guardrail",
    instructions="""
    You are a output guardrail agent. Check if output contains weather related response.
    """,
    model=model,
    output_type=WeatherOutput
)

@output_guardrail
async def output_guardrail_func(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
)->GuardrailFunctionOutput:
    result_output = await Runner.run(
        output_guardrail_agent, 
        output
    )

    return GuardrailFunctionOutput(
        output_info=result_output.final_output,
        tripwire_triggered=not result_output.final_output.is_weather
    )

agent = Agent(
    name="weather assistant",
    instructions="""
    You are a helpful assistant. If user query is related to weather of any city, call the 'get_weather_info' function with city name.
    """,
    model=model,
    input_guardrails=[input_guardrail_func],
    output_guardrails=[output_guardrail_func],
    tools=[get_weather_info]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="Hello! I'm a weather assistant. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        history = cl.user_session.get('history')
        history.append({'role': 'user', 'content': message.content})
        cl.user_session.set('history', history)
        response = await Runner.run(agent, input=history)
        history.append({'role': 'assistant', 'content': response.final_output})
        cl.user_session.set('history', history)
        await cl.Message(content=response.final_output).send()
    except InputGuardrailTripwireTriggered:
        await cl.Message(content="I'm sorry, but I can only assist with weather-related queries.").send()
    except OutputGuardrailTripwireTriggered:
        await cl.Message(content="I'm sorry, but I can only provide weather information.").send()

