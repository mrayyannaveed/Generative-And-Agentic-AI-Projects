from agents import Runner, Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
import chainlit as cl
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import input_guardrail, RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput, InputGuardrailTripwireTriggered

load_dotenv()
set_tracing_disabled(disabled=True) 

gemini_api_key = os.getenv("GEMINI_API_KEY")


client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=os.getenv("base_url")
)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash"
)

class OutputPython(BaseModel):
    is_python_related: bool
    reasoning: str

input_guardrails_agent = Agent(
    name="input Guardrail Checker",
    instructions="""Check if the user's question related to python programming, if it is, return true, if it is not, return false.""",
    model=model,
    output_type=OutputPython
)

@input_guardrail
async def input_guardrails_func(
    ctx :RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
)-> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrails_agent,
        input
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_related
    )

main_agent = Agent(
    name="Python Expert Agent",
    instructions="""You are a Python expert. Answer only the user's questions about Python programming.""",
    model=model,
    input_guardrails=[input_guardrails_func]
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Hello! I'm your Python expert agent. How can I assist you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        result = await Runner.run(
        main_agent,
        input=message.content
        )

        await cl.Message(content=result.final_output).send()

    except InputGuardrailTripwireTriggered:
        await cl.Message(content="I can only answer questions related to Python programming.").send()