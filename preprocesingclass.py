import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import re

#Esta clase define los pasos del preprocesamiento
class PreprocesamientoTexto:
  def __init__(self, aplicar_lowercase=True, remover_stopwords=True, aplicar_stemming=True, aplicar_lematizacion=True):
    #El constructor inicializa las opciones seleccionadas por el usuario
    self.aplicar_lowercase = aplicar_lowercase
    self.remover_stopwords = remover_stopwords
    self.aplicar_stemming = aplicar_stemming
    self.aplicar_lematizacion = aplicar_lematizacion

    #Inicializa las herramientas de NLTK
    self.stop_words = set(stopwords.words('english'))
    self.stemmer = PorterStemmer()
    self.lemmatizer = WordNetLemmatizer()

  def procesamiento(self, texto):
    #Tokenizar el texto en palabras
    tokens = word_tokenize(texto)

    #Paso 1: Eliminar signos de puntuación
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]

    #Paso 2: Lowercasing
    if self.aplicar_lowercase:
      tokens = [token.lower() for token in tokens]

    #Paso 3: Remover Stopwords
    if self.remover_stopwords:
      tokens = [token for token in tokens if token not in self.stop_words]

    #Paso 4: Stemming
    if self.aplicar_stemming:
      tokens = [self.stemmer.stem(token) for token in tokens]

    #Paso 5: Lematización
    if self.aplicar_lematizacion:
      tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

    return tokens
