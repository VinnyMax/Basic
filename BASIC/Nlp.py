from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 1. Preparar os dados
X = df['coluna_descricao']
y = df['coluna_classe']

# 2. Criar um Pipeline (Une a vetorização com o modelo)
modelo = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# 3. Treinar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo.fit(X_train, y_train)

# 4. Testar
precisao = modelo.score(X_test, y_test)
print(f"Acurácia: {precisao:.2f}")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# 1. Configurar a validação cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Iterar sobre os folds
fold = 1
for train_index, test_index in skf.split(X, y):
    # Separar dados de treino e teste do fold atual
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Treinar o modelo no fold
    modelo.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = modelo.predict(X_test)
    
    # Exibir o relatório
    print(f"\n{'='*20} FOLD {fold} {'='*20}")
    print(classification_report(y_test, y_pred))
    
    fold += 1






import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer # Stemmer específico para Português
import re

nltk.download('stopwords')
nltk.download('rslp')
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()





def limpar_texto(texto):
    # 1. Minúsculas e remoção de caracteres especiais/números
    texto = re.sub(r'[^a-zA-Záéíóúâêîôûãõç\s]', '', str(texto).lower())
    
    # 2. Tokenização simples
    palavras = texto.split()
    
    # 3. Remover Stop Words e aplicar Stemming (redução ao radical)
    palavras_limpas = [
        stemmer.stem(p) for p in palavras 
        if p not in stop_words and len(p) > 2
    ]
    
    return " ".join(palavras_limpas)

# Aplicar ao DataFrame
df['coluna_limpa'] = df['coluna_descricao'].apply(limpar_texto)


import spacy
import re

# Carregar o modelo de português (desabilitando o que não precisamos para ganhar velocidade)
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

def preprocessar_lemmatize(texto):
    # 1. Limpeza básica (remover números e pontuação)
    texto = re.sub(r'[^a-zA-Záéíóúâêîôûãõç\s]', '', str(texto).lower())
    
    # 2. Processar o texto com spaCy
    doc = nlp(texto)
    
    # 3. Lemmatização e remoção de Stop Words
    # .lemma_ extrai o lema (ex: 'compras' -> 'compra')
    tokens_limpos = [
        token.lemma_ for token in doc 
        if not token.is_stop and len(token.text) > 2
    ]
    
    return " ".join(tokens_limpos)

# Exemplo de aplicação no DataFrame
df['texto_lemmatizado'] = df['coluna_gastos'].apply(preprocessar_lemmatize)



import re
import spacy

# Carregar o modelo de português
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

def limpar_gastos_expert(texto):
    if not isinstance(texto, str): return ""
    
    # 1. Converter para minúsculas
    texto = texto.lower()
    
    # 2. Remover Datas (formatos comuns: 01/01, 2023-10-10, 15out)
    texto = re.sub(r'\d{2}/\d{2}(/\d{2,4})?', '', texto)
    texto = re.sub(r'\d{4}-\d{2}-\d{2}', '', texto)
    
    # 3. Remover Horários (formatos: 14:30, 14h30)
    texto = re.sub(r'\d{2}:\d{2}(:\d{2})?', '', texto)
    
    # 4. Remover IDs de transação e números isolados (ex: DOC 12345, #987)
    # Remove qualquer sequência de números com 3 ou mais dígitos
    texto = re.sub(r'\b\d{3,}\b', '', texto)
    
    # 5. Remover caracteres especiais e pontuação, mantendo acentos
    texto = re.sub(r'[^a-záéíóúâêîôûãõç\s]', ' ', texto)
    
    # 6. Lemmatização e Remoção de Stop Words (via spaCy)
    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and len(token.text) > 2
    ]
    
    # Retorna o texto limpo e unido por espaços
    return " ".join(tokens)

# Exemplo de uso:
# "PGTO ELETRONICO 15/10 DOC 987654 - SUPERMERCADOS BH" -> "pagamento eletronico supermercado"
df['coluna_limpa'] = df['coluna_gastos'].apply(limpar_gastos_expert)




# Remover linhas que ficaram vazias após a limpeza
df = df[df['coluna_limpa'].str.strip() != ""]



import pandas as pd
import re
import spacy
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Configuração do Processamento de Linguagem (NLP)
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

def limpar_gastos_expert(texto):
    if not isinstance(texto, str): return ""
    
    # Minúsculas e remoção de datas/horários/IDs longos
    texto = texto.lower()
    texto = re.sub(r'\d{2}/\d{2}(/\d{2,4})?', ' ', texto) # Datas
    texto = re.sub(r'\d{2}:\d{2}(:\d{2})?', ' ', texto)  # Horas
    texto = re.sub(r'\b\d{3,}\b', ' ', texto)            # Números IDs (3+ dígitos)
    texto = re.sub(r'[^a-záéíóúâêîôûãõç\s]', ' ', texto) # Símbolos
    
    # Lemmatização e Stop Words
    doc = nlp(texto)
    tokens = [t.lemma_ for t in doc if not t.is_stop and len(t.text) > 2]
    
    return " ".join(tokens)

# 2. Preparação dos Dados
# Supondo que seu df tenha colunas 'descricao' e 'categoria'
print("Iniciando limpeza de texto (isso pode levar um tempo)...")
df['texto_limpo'] = df['descricao'].apply(limpar_gastos_expert)

# Remover eventuais linhas que ficaram vazias após a limpeza
df = df[df['texto_limpo'].str.strip() != ""].copy()

X = df['texto_limpo']
y = df['categoria']

# 3. Definição do Pipeline e Validação Cruzada
# Usamos LinearSVC pois é excelente para textos curtos e muitas classes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # Pega palavras únicas e pares (ex: "posto", "shell")
    ('clf', LinearSVC(class_weight='balanced', random_state=42)) # 'balanced' ajuda nas 25 classes
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nIniciando Validação Cruzada...")
fold = 1
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Treino
    pipeline.fit(X_train, y_train)
    
    # Predição
    y_pred = pipeline.predict(X_test)
    
    # Relatório detalhado por Fold
    print(f"\n" + "="*30)
    print(f"RESULTADOS DO FOLD {fold}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    fold += 1

# 4. Treinamento Final (com todos os dados) para uso em produção
modelo_final = pipeline.fit(X, y)





