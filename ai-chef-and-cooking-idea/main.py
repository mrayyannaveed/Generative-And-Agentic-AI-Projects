from agents import Agent, InputGuardrailTripwireTriggered, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import chainlit as cl
from pydantic import BaseModel
from agents import input_guardrail, RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput, output_guardrail, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
BASE_URL = os.getenv("BASE_URL")
GEMINI_MODEL = "gemini-2.0-flash"

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)

model = OpenAIChatCompletionsModel(openai_client=client, model=GEMINI_MODEL)

class CookingRequest(BaseModel):
    is_cooking_related : bool
    response : str

input_guardrail_agent = Agent(
    name="input Guardrail Checker",
    instructions="You are a guardrail checker. You will check if the input is related to cooking, or asking for cooking ideas or asking what user should make or not.",
    model=model,
    output_type=CookingRequest
)

@input_guardrail
async def cooking_input_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
)-> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input=input
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_cooking_related
    )

class MessageOutput(BaseModel):
    response: str

class cookingOutput(BaseModel):
    is_cooking : bool
    reasoning : str

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="You are an output guardrail. You will check if the output is related to cooking, chef, making recipe, generating any idea what to cook or not.",
    model=model,
    output_type=cookingOutput
)

@output_guardrail
async def cooking_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: str | MessageOutput
)-> GuardrailFunctionOutput:
    output_result = await Runner.run(
        output_guardrail_agent,
        output
    )

    return GuardrailFunctionOutput(
        output_info=output_result.final_output,
        tripwire_triggered=not output_result.final_output.is_cooking
    )



cook_idea_agent = Agent(
    name="Cook Idea Agent",
    instructions="""You are a cooking idea agent. 
    You will be given a list of ingredients or if someone asks for a cooking idea or ask what make today etc., 
    you will return a cooking idea then you will hand off to the Chef Agent to provide its recipe.
      If user query in english, urdu, roman urdu, or roman english etc.. , you will reply in the same language like if query in english, urdu, roman urdu, or roman english etc.,
     you will reply in that language.You will Provide Pakistani, Chinese, american and all Halal foods.
     If someone say another or kuch aur it means suggest other than previous one""",
    model=model
)

chef_agent = Agent(
    name="Chef Agent",
    instructions="""You are a chef agent. You will be given a cooking idea or dish idea or dish name from Cooking Agent and you will return a recipe.
    If user query in english, urdu, roman urdu, or roman english etc.. , you will reply in the same language like if query in english, urdu, roman urdu, or roman english etc.
    Ingredients,
    how to make it.
    etc..""",
    model=model,
)

main_cooking_agent = Agent(
    name="Cooking Agent",
    instructions=("""You are a helpful cooking agent. If user want a cooking idea or any dish idea like suggest anything what i will make today etc., you will hand offs it to Cook Idea Agent or if user wants to create a recipe, you will hand off to Chef Agent. Then They will output the user query."""),
    model=model,
    handoffs=["Cook Idea Agent", "Chef Agent"],
    input_guardrails=[cooking_input_guardrail],
    output_guardrails=[cooking_output_guardrail],
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('history', [])
    await cl.Message(content="Welcome to the Cooking Assistant! How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        history = cl.user_session.get('history', [])
        history.append({'role': 'user', 'content': message.content})
        cl.user_session.set('history', history)

        response = await Runner.run(
            main_cooking_agent,
            history
        )

        main_reply = response.final_output.lower()

        if "cook idea agent" in main_reply:
            idea_response = await Runner.run(cook_idea_agent, history)
            final_output = idea_response.final_output

        elif "chef agent" in main_reply:
            chef_response = await Runner.run(chef_agent, history)
            final_output = chef_response.final_output

        else:
            final_output = response.final_output

        # Send and record the final message actually shown to user
        await cl.Message(content=final_output).send()
        history.append({"role": "assistant", "content": final_output})
        cl.user_session.set("history", history)

    except InputGuardrailTripwireTriggered:
        await cl.Message(content=f"Please try cooking related queries.").send()
    except OutputGuardrailTripwireTriggered:
        await cl.Message(content=f"Output is not related to cooking.").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()