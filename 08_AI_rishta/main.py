from agents import Runner, Agent, OpenAIChatCompletionsModel, set_tracing_disabled, set_default_openai_api, set_default_openai_client
from agents import input_guardrail, output_guardrail, RunContextWrapper, TResponseInputItem, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, GuardrailFunctionOutput
from openai import AsyncOpenAI
import chainlit as cl
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("BASE_URL")

client = AsyncOpenAI(api_key=gemini_api_key, base_url=base_url)
set_default_openai_client(client)
set_default_openai_api('chat_completions')

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash" 
)

class Rishta_Input(BaseModel):
    is_rishta_related: bool
    reason: str

input_guardrail_agent = Agent(
    name='Input Guardrail',
    instructions="""
    You are input guardrail, you can check whether the input is related to rishta.
    If user input is not related to rishta decline politely.
    """,
    model=model,
    output_type=Rishta_Input
)

@input_guardrail
async def input_rishta_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
)->GuardrailFunctionOutput:
    input_response = await Runner.run(
        input_guardrail_agent,
        input=input
    )
    return GuardrailFunctionOutput(
        output_info=input_response.final_output,
        tripwire_triggered=not input_response.final_output.is_rishta_related
    )

class Response_Output(BaseModel):
    name: str
    age: int
    gender: str
    profession: str
    cast : str
    location: str
    status: str

class Rishta_Output(BaseModel):
    is_rishta: bool
    reasoning: str

output_guardrail_agent = Agent(
    name='Output Guardrail',
    instructions="""
    You are output guardrail, you can check whether the output is related to rishta.
    """,
    model=model,
    output_type=Rishta_Input
)

@output_guardrail
async def output_rishta_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: str | Response_Output
)->GuardrailFunctionOutput:
    output_response = await Runner.run(
        output_guardrail_agent,
        output=output
    )
    return GuardrailFunctionOutput(
        output_info=output_response.final_output,
        tripwire_triggered=not output_response.final_output.is_rishta
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
    input_guardrails=[input_rishta_guardrail],
    output_guardrails=[output_rishta_guardrail],
    handoffs=[male_gender_rishta, female_gender_rishta]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="Welcome to the AI Rishta service! Please provide your details to get started.").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        history = cl.user_session.get('history', [])
        history.append({"role": "user", "content": message.content})
        cl.user_session.set('history', history)
        result = await Runner.run(
            rishta_agent,
            input=history
        )
        history.append({"role": "assistant", "content": result.final_output})
        cl.user_session.set('history', history)
        await cl.Message(content=result.final_output).send()
    
    except InputGuardrailTripwireTriggered:
        await cl.Message(content='Only rishta related queries get response.').send()
    except OutputGuardrailTripwireTriggered:
        await cl.Message(content=f"Output is not related to rishta.").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()
