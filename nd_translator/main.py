import googletrans
from googletrans import Translator
import streamlit as st

st.set_page_config = {
    'page_title': 'Language Translator',
    'page_icon': '📝',
    'layout': 'wide'
}

st.title('Language Translator')

language = st.selectbox('Select Language', googletrans.LANGUAGES.values())

if language:
    translator = Translator()
    text = st.text_area('Enter Text')
    if text:
        translated_text = translator.translate(text, dest=language)
        st.info(translated_text.text)
else:
    st.info('Please select a language')