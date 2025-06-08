import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# Validate environment variable
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# Correct way to set page config
st.set_page_config = {
    "page_title": "Inde-Gpt",
    "page_icon": "🤖",
    "layout": "wide"
}

st.title("🤖 Inde-Gpt")
st.caption("🚀 A Streamlit chatbot powered by OpenRouter")

# Prompt input
prompt = st.chat_input("Pls ask me a question")

# Only make API call if prompt exists
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    try:
        response = requests.post(
            url=BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )

        data = response.json()
        # print("API response:", data)

        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            with st.chat_message("assistant"):
                st.write(reply)
        else:
            st.error(f"Unexpected API response format:\n\n{json.dumps(data, indent=2)}")

    except Exception as e:
        st.error(f"❌ API call failed: {e}")

# Optional debug function
def main():
    print("Hello from Inde-Gpt!")

if __name__ == "__main__":
    main()
