import pywhatkit as kit
import streamlit as st
from dotenv import load_dotenv
from agents import Runner, Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("base_url")


client = AsyncOpenAI(api_key=gemini_api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash"
    )

male_gender_rishta = Agent(
    name="Male Gender Rishta",
    instructions="""
    You are a rishta agent for males. When a male user requests, find a suitable female rishta.
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

rishta_agents = Agent(
    name="Rishta Agents",
    instructions="A rishta agent, if the gender is male then it will hand off the male agent, otherwise it will hand off the female agent.",
    handoffs={
        "male": male_gender_rishta,
        "female": female_gender_rishta
    },
    model=model
)


st.set_page_config(page_title="Whatsapp", page_icon="🧑‍🤝‍🧑", layout="centered")

st.image('https://2.imimg.com/data2/GT/KW/MY-4179296/rishta-logo-1000x1000.jpg', use_container_width=True)
st.title("AI Rishta App")
st.write("This app uses pywhatkit to se" \
"nd messages on Whatsapp.")
name = st.text_input(label="Name", placeholder="Enter your name")
age = st.number_input(label="Age", min_value=0, max_value=50)
gender = st.selectbox(label="Gender", options=["Male", "Female"])
phone = st.text_input(label="Phone Number", placeholder="Enter your phone number")
bio = st.text_area(label="Bio (optional)", placeholder="Tell us about yourself")


if st.button('Find Rishta'):
    result = Runner.run(
        rishta_agents,
        input={
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "bio": bio
        }
    )
    # Get the agent's generated rishta details as a string
    rishta_details = result if isinstance(result, str) else str(result)

    formatted_phone = phone if phone.startswith('+') else '+92' + phone.lstrip('0+')

    try:
        kit.sendwhatmsg_instantly(formatted_phone, rishta_details)
        st.success("Rishta details sent to WhatsApp!")
    except Exception as e:
        st.error(f"Failed to send WhatsApp message: {e}")