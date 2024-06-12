#lấy stopwwords cho phân loại chủ đề
import pandas as pd
import numpy as np

#tạo từ điển chứa các label, các từ thuộc label và số lần xuất hiện của từ
#input: file txt chứa label và các bài báo thuộc label
#output: labelvocab, gồm các dict con có dạng tên label{từ: số lần xuất hiện} + vocab, có dạng từ{tên label}
vocab = {}
label_vocab = {}

for line in open('news_categories.txt','r', encoding='utf-8'):
    words = line.split() 
    label = words[0] 
    if label not in label_vocab:
        label_vocab[label] = {} 
    for word in words[1:]:
        label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
        if word not in vocab:
            vocab[word] = set()
        vocab[word].add(label) 

#lấy các label

label_=[]
for line in open('news_categories.txt','r', encoding='utf-8'):
    part=line.split(' ',1)
    lb=part[0]
    if(lb not in label_):
        label_.append(lb)

#tạo một từ điển lưu các từ xuất hiện ở TẤT CẢ các label+ số lần xuất hiện của từ
#input: vocab, có dạng từ{tên label}
#output: dictionaryc count, có dạng: từ {số lần xuất hiện ít nhất trong các label}.  ví dụ: {và 10000}
count = {} 
for word in vocab: 
    if len(vocab[word]) == len(label_): 
        count[word] = min([label_vocab[x][word] for x in label_vocab])

#sắp xếp các phần tử trong count{} theo thứ tự giảm dần số lần xuất hiện
sorted_count = sorted(count, key=count.get, reverse=True) 

#lấy 100 từ đầu tiên trong sort_count làm stopwords
#xuất ra file txt các stopwords

stopword=set()
with open('stopwords2.txt', 'w',encoding='utf-8') as fp: 
    for word in sorted_count[:100]:
        stopword.add(word)
        fp.write(word + '\n')