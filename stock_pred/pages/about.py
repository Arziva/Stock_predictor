import streamlit as st
from PIL import Image
import time


st.title(":blue[ABOUT US]")

with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(2)
    st.success("Done!")



st.subheader("Arun kumar M III year CSE")
st.subheader("Harsha prada S III year CSE")
st.subheader("Lakshay kumar III year CSE")
image = Image.open('/Users/arunkumar/Desktop/Projects/pages/images/team.jpeg')
st.image(image)
