import pickle as pickle
import pandas as pd
import streamlit as st
from summarization import summarization
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import sent_tokenize
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx



input_text = st.text_area("Nhập đoạn văn bản cần tóm tắt",height=600)
btntomtat=st.button("Tóm tắt")

if btntomtat:
    if input_text:
        output_value = summarization(input_text)
        output_text = st.text_area("Nội dung tóm tắt ở đây:",value=output_value,height=400)
    else:
        st.warning("Vui lòng nhập một đoạn văn bản.")
