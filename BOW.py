from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#Clase que implementa la Bolsa de Palabras
class BoW:
  #Recibe un vocabulario de palabras (set)
  def __init__(self, vocabulario):
    self.vocabulario = vocabulario
    self.vocabulario_size = len(vocabulario)

  #genera los vectores bow contando cuantas veces aparece cada palabra del vocabulario en el texto actual
  def vectorizar(self, text_tokens):
    bow_vectors = []
    for tokens in text_tokens:
        word_counts = Counter(tokens)
        vector = [word_counts.get(word, 0) for word in vocabulario]
        bow_vectors.append(vector)
    return bow_vectors