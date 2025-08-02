from agents import (
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_default_openai_api,
    set_default_openai_client,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered
)
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import chainlit as cl
import os

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=BASE_URL
)

set_default_openai_client(external_client)
set_default_openai_api('chat_completions')

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

class WebDevInputExpert(BaseModel):
    is_web_dev_related: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="Input Guardrail",
    instructions="""
    You are an input guardrail agent. Your task is to analyze the user's input and determine whether it is related to web development.

    Mark the input as web development-related if it involves:
    - Frontend technologies (e.g., HTML, CSS, JavaScript, React, Next.js, Tailwind CSS, UI/UX)
    - Backend technologies (e.g., APIs, databases, authentication, Node.js, Express, Django, etc.)

    Mark it as unrelated if it concerns general programming, personal questions, gaming, mobile apps, or DevOps.

    Respond with structured reasoning explaining your decision.
    """,
    model=model,
    output_type=WebDevInputExpert
)

@input_guardrail
async def web_dev_input_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input=input
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_web_dev_related
    )

class WebDevOutput(BaseModel):
    output: str

class WebDevOutputExpert(BaseModel):
    is_web_dev: bool
    reasoning: str

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="""
    You are an output guardrail agent. Your task is to verify whether the assistant's output is relevant to web development.

    Mark it as invalid if it is vague, off-topic, or unrelated to frontend/backend development.
    Provide structured reasoning for your decision.
    """,
    model=model,
    output_type=WebDevOutputExpert
)

@output_guardrail
async def web_dev_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: str | WebDevOutput
) -> GuardrailFunctionOutput:
    output_result = await Runner.run(
        output_guardrail_agent,
        output
    )
    return GuardrailFunctionOutput(
        output_info=output_result.final_output,
        tripwire_triggered=not output_result.final_output.is_web_dev
    )

frontend_agent = Agent(
    name="Frontend Expert",
    instructions="""
    You help with frontend.
    Fullfill user requirements in the boundary of frontend.
    Always provide complete and helpful code snippets when relevant.
    Do NOT answer backend or server-side questions.
    """,
    model=model
)

backend_agent = Agent(
    name="Backend Expert",
    instructions="""
    You help with backend.
    Fullfill user requirements in the boundary of backend.
    Always provide complete and helpful code snippets when relevant.
    Do NOT answer frontend or UI questions.
    """,
    model=model
)

web_dev_agent = Agent(
    name="Web Dev Expert",
    instructions="""
    You are a generalist web developer who routes queries to the correct expert.
    If query is related to frontend development use frontend agent tool.
    If query is related to backend development use backend agent tool.
    """,
    model=model,
    input_guardrails=[web_dev_input_guardrail],
    output_guardrails=[web_dev_output_guardrail],
    tools=[
        frontend_agent.as_tool(
            tool_name='frontend_agent',
            tool_description='Handles frontend (HTML, CSS, React, or UI-related) questions.'
        ),
        backend_agent.as_tool(
            tool_name='backend_agent',
            tool_description='Handles backend (APIs, databases, authentication, or server-side) questions.'
        )
    ]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am a web development expert. How can I help you?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        history = cl.user_session.get("history", [])
        history.append({'role': 'user', 'content': message.content})
        cl.user_session.set("history", history)

        result = await Runner.run(
            web_dev_agent,
            input=message.content
        )

        history.append({'role': 'assistant', 'content': result.final_output})
        cl.user_session.set("history", history)

        await cl.Message(content=result.final_output).send()

    except InputGuardrailTripwireTriggered as e:
        await cl.Message(content=f"🚫 Input not related to web development: {e}").send()

    except OutputGuardrailTripwireTriggered as e:
        await cl.Message(content=f"⚠️ Output rejected by output guardrail: {e}").send()

    except Exception as e:
        await cl.Message(content=f"❌ An unexpected error occurred: {e}").send()
