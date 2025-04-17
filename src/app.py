import sqlite3
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import re
# Funciona pero solo con 10000, habría que aplicarlo a todo el conjunto de datos
# Paso 1: Preprocesamiento de datos
# Carga de datos desde SQLite
current_dir = os.getcwd()
db_path = os.path.join(current_dir, r"text\en-es\ccmatrix.db")

# Función para cargar los datos desde la base de datos SQLite
def load_data_from_sqlite(db_path, min_score=1.1, limit=None):
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT english, spanish FROM translations
    WHERE score >= {min_score}
    """
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.dropna()

df = load_data_from_sqlite(db_path, min_score=1.1, limit=10000)  # Usamos un límite por ahora

# Preprocesamiento básico de texto
def preprocess_text(text):
    # Eliminar caracteres no alfabéticos y convertir a minúsculas
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text

# Preprocesamos los datos
df['english'] = df['english'].apply(preprocess_text)
df['spanish'] = df['spanish'].apply(preprocess_text)

# Dividir en datos de entrenamiento y prueba
train_data, val_data = train_test_split(df, test_size=0.1)

# Crear un vocabulario basado en las palabras únicas de ambos idiomas
def create_vocab(texts):
    words = [word for sentence in texts for word in sentence.split()]
    vocab = Counter(words)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}  # Asignamos un ID a cada palabra
    vocab['<PAD>'] = 0  # Añadimos el padding
    return vocab

english_vocab = create_vocab(train_data['english'])
spanish_vocab = create_vocab(train_data['spanish'])

# Función para convertir frases a secuencias de índices
def sentence_to_indices(sentence, vocab, max_len=50):
    indices = [vocab.get(word, vocab['<PAD>']) for word in sentence.split()]
    return indices[:max_len] + [vocab['<PAD>']] * (max_len - len(indices))  # Aseguramos que todas las secuencias tienen el mismo tamaño

# Convertir las frases en inglés y español a índices
X_train = [sentence_to_indices(sentence, english_vocab) for sentence in train_data['english']]
y_train = [sentence_to_indices(sentence, spanish_vocab) for sentence in train_data['spanish']]

X_val = [sentence_to_indices(sentence, english_vocab) for sentence in val_data['english']]
y_val = [sentence_to_indices(sentence, spanish_vocab) for sentence in val_data['spanish']]


# Paso 2: Definir el modelo Seq2Seq
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, n_layers=2, dropout=0.1):
        super(Seq2SeqModel, self).__init__()
        
        # Embedding para las secuencias de entrada
        self.encoder_embedding = nn.Embedding(input_dim, embedding_dim)
        self.decoder_embedding = nn.Embedding(output_dim, embedding_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Capa de salida
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Capa de atención (opcional, pero ayuda mucho en traducción)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, trg):
        # Embedding de las secuencias de entrada y salida
        src_embedded = self.encoder_embedding(src)
        trg_embedded = self.decoder_embedding(trg)
        
        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(src_embedded)
        
        # Decoder
        decoder_output, _ = self.decoder_lstm(trg_embedded, (hidden, cell))
        
        # Output layer
        output = self.fc_out(decoder_output)
        
        return output

# Paso 3: Entrenamiento del modelo
import torch.optim as optim

# Hiperparámetros
embedding_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.1
lr = 0.001
batch_size = 64
n_epochs = 10

# Inicializamos el modelo
model = Seq2SeqModel(input_dim=len(english_vocab), output_dim=len(spanish_vocab),
                     embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                     n_layers=n_layers, dropout=dropout)

# Optimización y función de pérdida
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=spanish_vocab['<PAD>'])

# Convertimos los datos de entrenamiento y validación a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Función de entrenamiento
def train(model, X_train, y_train, optimizer, criterion, n_epochs=10):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Pasamos los datos por el modelo
            output = model(batch_x, batch_y[:, :-1])  # Aseguramos que no incluya el token <PAD> en la salida
            
            # Calculamos la pérdida
            loss = criterion(output.view(-1, len(spanish_vocab)), batch_y[:, 1:].reshape(-1))  # Ignoramos <PAD>
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(X_train)}")

# Entrenamos el modelo
train(model, X_train_tensor, y_train_tensor, optimizer, criterion, n_epochs=n_epochs)

def translate(model, sentence, vocab, max_len=50):
    sentence_indices = sentence_to_indices(sentence, vocab, max_len)
    input_tensor = torch.tensor(sentence_indices, dtype=torch.long).unsqueeze(0)  # Aseguramos que sea un batch
    model.eval()
    
    # Paso por el modelo
    with torch.no_grad():
        output = model(input_tensor, input_tensor)  # En este caso, usamos la misma entrada
    
    # Convertir los índices de vuelta a palabras
    translated_sentence = [list(spanish_vocab.keys())[i] for i in output.argmax(dim=-1).squeeze().cpu().numpy()]
    
    return ' '.join(translated_sentence)

# Traducir una frase
print(translate(model, "I am learning NLP", english_vocab))
