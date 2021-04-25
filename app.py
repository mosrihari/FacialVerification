from util.match_face import is_match
from util.embedding import get_embeddings
from PIL import Image
import streamlit as st
st.title("Facial Similarity verification")

if __name__ == "__main__":
    uploaded_file1 = st.file_uploader("Choose the first image", type="jpg")
    uploaded_file2 = st.file_uploader("Choose the second image", type="jpg")
    if((uploaded_file1 is not None) and (uploaded_file2 is not None)):
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption='Uploaded Image.1', use_column_width=True)
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption='Uploaded Image.2', use_column_width=True)
        filenames = [uploaded_file1, uploaded_file2]
        # get embeddings file filenames
        embeddings = get_embeddings(filenames)
        score, threshold = is_match(embeddings[0], embeddings[1])
        if(score < threshold):
            st.write("Face is a match")
        else:
            st.write("Face is not a match")
    else:
        st.write("")