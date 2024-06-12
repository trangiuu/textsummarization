import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

#mở file dữ liệu đã loại stopwords
data=open('newcontent.txt',encoding='utf-8')
lines=[]
for line in data:
    line=data.readline()
    part=line.split(' ',1)
    label=part[0]
    content=part[1]
    lines.append([label, content])

#chuyển thành dataframe có 2 cột Label, Content
df = pd.DataFrame(lines, columns=['Label', 'Content'])
content=df['Content'].tolist()
lb=df['Label'].tolist()

#chia tập train/test tỷ lệ 70/30
X_train, X_test, y_train, y_test = train_test_split(content, lb, test_size=0.3, random_state=42)

#vector hóa dữ liệu
vectorizer = CountVectorizer()
for i in range(len(X_train)):
    X_train[i]=X_train[i].lower()
X_train=vectorizer.fit_transform(X_train)

train_vocab=vectorizer.vocabulary_
vectorizer = CountVectorizer(vocabulary=train_vocab)

for i in range(len(X_test)):
    X_test[i]=X_test[i].lower()
X_test= vectorizer.fit_transform(X_test)

# huấn luyện mô hình
clf = MultinomialNB()
clf.fit(X_train, y_train)

#đóng gói dữ liệu
pickle.dump(clf,open("Plchude.pkl", 'wb'))