#TFIDF class
import pandas as pd
import re
import math
from collections import Counter
import numpy as np
class TFIDF:
  def __init__(self, tokenized_documents,vocab):
        #recibe los documentos ya tokenizados
        self.documents = tokenized_documents
        #conteo de los documentos totales
        self.N = len(tokenized_documents)
        #inicialización del vocabulario
        self.vocab = vocab
        #tamaño del vocabulario
        self.vocab_size = len(self.vocab)
        #ejecutar los cálculos
        self.df = self.compute_df()
        self.idf = self.compute_idf()
        self.tfidf_vectors = self.compute_tfidf_vectors()

  #calcula la frecuencia de los términos en todos los documentos
  def compute_df(self):
      df = {term: 0 for term in self.vocab}
      for doc in self.documents:
          unique_terms = set(doc)
          for term in unique_terms:
              df[term] += 1
      return df

  #calcula la frecuencia inversa de documentos
  def compute_idf(self):
      return {
          term: math.log(self.N / (1 + df_t))  # se suma 1 para evitar división por cero
          for term, df_t in self.df.items()
          }
  #calcula la frecuencia de termino
  def compute_tf(self, tokens):
      counts = Counter(tokens)
      total = len(tokens)
      return {term: counts.get(term, 0) / total for term in self.vocab}

  #calcula un solo vector tf-idf
  def compute_tfidf_vector(self, tokens):
      tf = self.compute_tf(tokens)
      return [tf[term] * self.idf[term] for term in self.vocab]

  #calcula todos los vectores tf-idf para todos los documentos
  def compute_tfidf_vectors(self):
      return [self.compute_tfidf_vector(doc) for doc in self.documents]

  #regresa los vectores TF-IDF
  def get_vectors(self):
      return self.tfidf_vectors

  # construye la matriz termino documento a partir de los vectores TF-IDF calculados
  def get_tfidf_matrix(self):
    tfidf_matrix = self.compute_tfidf_vectors()
    return pd.DataFrame(tfidf_matrix, columns=self.vocab)

