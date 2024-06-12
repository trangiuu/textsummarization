import pandas as pd
import re
from underthesea import word_tokenize
#hàm tạo từ điền stopword
#tạo bằng tệp dữ liệu crawl
data=pd.read_csv('dataset0001.csv')
data= data.dropna(subset=['Noidung'])
content=list(data['Noidung'])

from collections import Counter
def cre_stop_word(data,per):
    all_tokens = []
    for text in data:
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens=word_tokenize(text)
        all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)
    
    total_words = sum(word_freq.values())
    stopword_threshold = total_words * per
    stopwords = [word for word, freq in word_freq.items() if freq > stopword_threshold]
    
    return stopwords

stopwords=cre_stop_word(content,0.01)
#print(stopwords)