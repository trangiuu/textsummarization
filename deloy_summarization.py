import pickle as pickle
import pandas as pd
import streamlit as st
from summarization import summarization
from phan_loai_chu_de import train_vocab
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import sent_tokenize
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx



input_text = st.text_area("Nhập đoạn văn bản cần tóm tắt",height=600)
btntomtat=st.button("Tóm tắt")
btnchude=st.button("Chủ đề")
output_value = summarization(input_text)
clf=pickle.load(open('Plchude.pkl', 'rb'))
vectorizer = CountVectorizer(vocabulary=train_vocab)

if btntomtat:
    if input_text:
        output_text = st.text_area("Nội dung tóm tắt ở đây:",value=output_value,height=400)
        new_text_vectorized = vectorizer.transform([output_text])
        label = clf.predict(new_text_vectorized)
            
    else:
        st.warning("Vui lòng nhập một đoạn văn bản.")

if btnchude:
        st.text_area("Chủ đề đoạn tóm tắt",value=label,height=100)

