# import streamlit as st
# from components.audio_selector import select_audio_file
# from features.predictor import predict_genre_streamlit

# st.title("Music Genre Classifier with XAI")

# file_path = select_audio_file()
# model_choice = st.selectbox("Choose prediction model:", ["SVM", "CNN", "VGGish"])

# if file_path:
#     st.audio(file_path)

#     if st.button("Predict"):
#         st.write("Prediction...")
#         predict_genre_streamlit(file_path, model_choice)

#     # if st.button("Waterfall plot"):
#     #     st.write("📊 Generowanie wykresu waterfall...")
#     #     fig = generate_waterfall_plot()
#     #     st.pyplot(fig)

# main app file (e.g., app.py or streamlit_app.py)

from components.config import configure_page
from components.audio_selector import select_audio_file
from features.predictor import predict_genre_streamlit
import streamlit as st

configure_page()

st.title("Music Genre Classifier with XAI")

model_choice = st.selectbox("Choose prediction model:", ["SVM", "CNN", "VGGish"])

file_path = select_audio_file()

if file_path:
    st.audio(file_path)

    if st.button("Predict"):
        st.write("Prediction...")
        predict_genre_streamlit(file_path, model_choice)
