# https://heytech.tistory.com/337

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

print("!pip install -U -q pandas numpy sklearn")

documents = [
   "배가 부르다",
   "배의 가격이 비싸다",
   "진짜 사과가 진짜 좋다",
   "아침엔 사과가 좋다"
]

vectorizer = CountVectorizer()

# Document Term Matrix
dtm = vectorizer.fit_transform(documents)

# Term Freqeuncy
tf = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names_out())
print(tf)

# Document Frequency 
df = tf.astype(bool).sum(axis = 0)
print(df)

# 문서 개수
D = len(tf)

# Inverse Document Frequency
idf = np.log((D+1) / (df+1)) + 1
print(idf)

# TF-IDF
tfidf = tf * idf                      
tfidf = tfidf / np.linalg.norm(tfidf, axis = 1, keepdims = True)
print(tfidf)