import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

texto_a_ser_comparado = "suponha que seja o texto que desejo comparar"
texto_fonte = "suponha que essa seja o texto principal"


# UNIGRAMA
# numero de n_gramas
n = 1

# instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# cria um dicionario de n-gramas
vocb2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# printa o dicionario de palavras: index
print(vocab2int)


# BIGRAMA
# numero de n_gramas
n = 2

# instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# cria um dicionario de n-gramas
vocb2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# printa o dicionario de palavras: index
print(vocab2int)


# TRIGRAMA
# numero de n_gramas
n = 3

# instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# cria um dicionario de n-gramas
vocb2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# printa o dicionario de palavras: index
print(vocab2int)


# ARRAY DE N-GRAMAS
# numero de n_gramas
n = 1

# instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# cria uma matriz de contagem de n-grama para os dois textos
vocb2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

n_grams_array = n_grams.toarray()

print('Vetor de n-gramas:\n\n', n_grams_array)
print()
print('Dicionario de n-gramas (unigrama):\n\n', vocb2int)

# Valores de CONTAINMENT
n_grams

n_grams.toarray()

intersection_list = np.amin(n_grams.toarray(), axis = 0)
intersection_list

intersection_count = np.sum(intersection_list)
intersection_count

index_A = 0
A_count = np.sum(n_grams.toarray()[index_A])
A_count

# grau de similaridade
intersection_count/A_count


def containment(n_gram_array):
    intersection_list = np.amin(n_gram_array, axis = 0)
    intersection_count = np.sum(intersection_list)

    A_idx = 0
    A_count = np.sum(n_gram_array[A_idx])

    containment_val = intersection_count / A_count

    return containment_val

# para o n_gram calculado anteriormente e n=1
containment_val = containment(n_grams.toarray())
print('Containment: ', containment_val)

# para n = 2
counts_2grams = CountVectorizer(analyzer='word', ngram_range=(2,2))
bigram_counts = counts_2grams.fit_transform([texto_a_ser_comparado, texto_fonte])
containment_val = containment(bigram_counts.toarray())

