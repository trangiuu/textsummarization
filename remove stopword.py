import pandas as pd
import numpy as np
import underthesea
from underthesea import word_tokenize

#import cái stopwords, cho ra một list
data=open('stopwords2.txt',encoding='utf-8')
stopwords=[]
for word in data.readline():
    stopwords.append(word)

#hàm xóa stopwords, input (một câu, tập stopwords), output là câu sau khi xóa
def text_remove_stopword(text,stopword):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopword]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#lưu dataset thành file mới, đã xóa stopwords

# with open('newcontent.txt', 'w',encoding='utf-8') as fp: 
#     for line in open('news_categories.txt',encoding='utf-8'):
#         line = text_remove_stopword(line,stopwords) 
#         fp.write(line + '\n') 