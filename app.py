
import streamlit as st
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header=("Hey ask me anything I will tell you similar things")
import pandas as pd
data = pd.read_excel('/content/Dummy.xlsx')
print(data)
df= pd.DataFrame(data,columns =['Words', 'category'])
print(df)
from sentence_transformers import SentenceTransformer
text= df['Words']
encoder= SentenceTransformer("paraphrase-mpnet-base-v2")
vectors= encoder.encode(text)
print(vectors)
def get_text():
  input_text =st.text_input("You: ",key=input)
  return input_text

user_input=get_text()

import faiss
vector_dimension = vectors.shape[1]
index= faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

import numpy as np
search_text = user_input
search_vector = encoder.encode(search_text)
_vector= np.array([search_vector])
faiss.normalize_L2(_vector)

k=index.ntotal
distances, ann =index.search(_vector, k=k)

results = pd.DataFrame({'distances' : distances[0],'ann': ann[0]})
print(results)
merge = pd.merge(results,df,left_on='ann',right_index=True)
submit=st.button('Find similar things')
if submit:
  st.text(merge['Words'].head(2))
print(merge)