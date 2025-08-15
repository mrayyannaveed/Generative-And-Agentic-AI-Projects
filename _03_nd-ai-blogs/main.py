# import google.generativeai as genai
from dotenv import load_dotenv #? For loading environment variables
import os #! Getting values of environment variables
import chainlit as cl
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, WebSearchTool
from openai import AsyncOpenAI

load_dotenv()
set_tracing_disabled(disabled=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel(model_name="gemini-2.0-flash")

client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.5-flash")

blog_writer = Agent(
    name="Blog Writer Agent",
    instructions="""You are a professional and creative blog post writer with expertise in SEO optimization. You will be given a topic, and you will write a comprehensive blog post on that topic.

Your writing should be:
- Engaging and informative with a natural flow
- Well-structured with clear headings and subheadings
- Include relevant statistics and examples where appropriate
- Optimized for SEO with strategic keyword placement
- 800-1200 words in length for optimal readability
- Include a compelling meta description
- Have a catchy title that incorporates primary keywords""",
    handoff_description=[
        "Write blog posts on a given topic",
        "Write blog posts with SEO optimization in mind",
        "Write blog posts with a natural flow",
        "Write blog posts with clear headings and subheadings",
        "Write blog posts with relevant statistics and examples",
        "Write blog posts with strategic keyword placement",
        "Write blog posts with 800-1200 words in length for optimal readability",
        "Write blog posts with a compelling meta description",
        "Write blog posts with a catchy title that incorporates primary keywords",
    ],
    model=model
)

base_agent = Agent(
    name="Base Agent",
    instructions="""You are a helpful agent. If prompt related to blog, article then use Blog Writer tool. Otherwise use web search tool to answer.""",
    tools=[blog_writer.as_tool(tool_name="blog_writer", tool_description="You are a comprehendive blog writer.")],
    model=model
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    # await cl.Message(content="Hello there, I am a Gemini AI chatbot. How can I help you?").send()
    await cl.Message(content="Hello there! I'm a blog post writer. Give me a topic and I'll create an engaging blog post for you. Please note that I can only help with blog post creation.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({'role': 'user', 'content': message.content})
    cl.user_session.set("history", history)

    result = await Runner.run(starting_agent=base_agent, input=history)

    history.append({'role': 'assistant', 'content': result.final_output})
    cl.user_session.set("history", history)
    
    await cl.Message(content=result.final_output).send()
    # if "blog" in history.lower() or "post" in history.lower() or "write" in history.lower() or "article" in history.lower():
    #     full_prompt = f"{BLOG_PROMPT}\nTopic: {history}"
    #     response = model.generate_content(full_prompt)
    #     response_text = response.text if hasattr(response, 'text') else ""
    #     await cl.Message(content=response_text).send()
    # elif "hello" in history.lower() or "hi" in history.lower() or "assalam u alaikum" in history.lower():
    #     await cl.Message(content="Hello! How can I assist you today?").send()
    # else:
    #     await cl.Message(content="I apologize, but I can only help with creating blog posts. Please provide a topic for a blog post.").send()
