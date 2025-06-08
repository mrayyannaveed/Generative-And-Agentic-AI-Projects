import streamlit as st
import googletrans

st.set_page_config(
    page_title="Inde Translater",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Inde Translater")

language = st.selectbox("Select Language", googletrans.LANGUAGES.values())

if language:
    text = st.text_area("Enter Text", height=100)
    if  st.button("Translate"):
        with st.status("Translating...", expanded=True) as status: 
            translator = googletrans.Translator()
            translated_text = translator.translate(text, dest=language)
            status.update(label="Translation complete ✅", state="complete")
            st.info(translated_text.text)
else:
    st.info("Please select a language.")
