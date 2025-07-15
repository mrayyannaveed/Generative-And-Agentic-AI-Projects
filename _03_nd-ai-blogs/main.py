import google.generativeai as genai
from dotenv import load_dotenv #? For loading environment variables
import os #! Getting values of environment variables
import chainlit as cl

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

BLOG_PROMPT = """You are a professional and creative blog post writer with expertise in SEO optimization. You will be given a topic, and you will write a comprehensive blog post on that topic.

Your writing should be:
- Engaging and informative with a natural flow
- Well-structured with clear headings and subheadings
- Include relevant statistics and examples where appropriate
- Optimized for SEO with strategic keyword placement
- 800-1200 words in length for optimal readability
- Include a compelling meta description
- Have a catchy title that incorporates primary keywords

Format the blog post in markdown with:
- A clear title (H1)
- Organized sections with H2 and H3 headings
- Bullet points and numbered lists where relevant
- Bold and italic text for emphasis
- At least one call-to-action
- A conclusion section

Please include:
- Primary and secondary SEO keywords naturally distributed
- Meta description (160 characters)
- Internal and external linking opportunities
- Image placement suggestions
"""

@cl.on_chat_start
async def handle_chat_start():
    await cl.Message(content="Hello there, I am a Gemini AI chatbot. How can I help you?").send()
    await cl.Message(content="Hello there! I'm a blog post writer. Give me a topic and I'll create an engaging blog post for you. Please note that I can only help with blog post creation.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    prompt = message.content
    if "blog" in prompt.lower() or "post" in prompt.lower() or "write" in prompt.lower() or "article" in prompt.lower():
        full_prompt = f"{BLOG_PROMPT}\nTopic: {prompt}"
        response = model.generate_content(full_prompt)
        response_text = response.text if hasattr(response, 'text') else ""
        await cl.Message(content=response_text).send()
    elif "hello" in prompt.lower() or "hi" in prompt.lower() or "assalam u alaikum" in prompt.lower():
        await cl.Message(content="Hello! How can I assist you today?").send()
    else:
        await cl.Message(content="I apologize, but I can only help with creating blog posts. Please provide a topic for a blog post.").send()
