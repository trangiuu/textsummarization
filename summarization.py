import pandas as pd
import numpy as np
import underthesea
from stopwords import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import sent_tokenize
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

stopword=stopwords
#tiền xử lý dữ liệu
#tách đoạn thành các câu riêng lẻ

def text_sentence(text):
    sentence=sent_tokenize(text)
    return sentence
#chuyển chữ hoa thành chữ thường
def text_lower(text):
    textlower=text.lower()
    return text
#loại bỏ kí tự đặc biệt

def text_remove_charac(text):
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
#loại bỏ stop word
from underthesea import word_tokenize
def text_remove_stopword(text,stopword):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopword]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#vector hóa câu trong văn bản
#dùng Tfid

#tạo từ điển dữ liệu
def tfid_vocab(text):
    vectorizer=TfidfVectorizer()
    matrix=vectorizer.fit_transform(text)
    vocab=vectorizer.get_feature_names_out()
    return matrix,vocab

#tạo ra ma trận tương tự
#dùng coisine

def prepare_matrix(text,matrix,vocab):
    sim_mat = np.zeros([len(text), len(text)])
    for i in range(len(text)):
        for j in range(len(text)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(matrix[i].reshape(1,len(vocab)), matrix[j].reshape(1,(len(vocab))))[0,0]
    graph = nx.from_numpy_array(sim_mat)
    return graph

#tính độ tương đồng, sắp xếp
def rank_sort(graph,text):
    scores = nx.pagerank(graph)
    rank = sorted(((scores[i],s) for i,s in enumerate(text)), reverse=True)
    return rank

def sum(rank_sort,quan):
    sum=""
    for i in range(quan):
        sum=sum+str(rank_sort[i][1])
    return sum

def summarization(content):
    text=[]
    text_pre=[]
    text=text_sentence(content)
    for i in range(len(text)):
        sen=text_lower(text[i])
        sen=text_remove_charac(text[i])
        sen=text_remove_stopword(text[i],stopword)
        text_pre.append(sen)
    
    matran,tudien=tfid_vocab(text_pre)
    dothi=prepare_matrix(text_pre,matran,tudien)
    xephang=rank_sort(dothi,text_pre)

    soluongcau=round(len(text_pre)*0.3)
    sum_output=sum(xephang,soluongcau)
    #print(sum_output)
    return sum_output



    

