import sqlite3
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import re
import pickle
from nltk.translate.bleu_score import corpus_bleu

# Liberar memoria de la GPU
torch.cuda.empty_cache()
# Paso 1: Preprocesamiento de datos
# Carga de datos desde SQLite
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "../en-es/ccmatrix.db")
print(db_path)
conn = sqlite3.connect(db_path)     # hacemos que la conexión sea solo una y no esté abriendo y cerrando
try:
    db_path = os.path.join(current_dir, "../en-es/ccmatrix.db")
    # Función para cargar los datos desde la base de datos SQLite
    def load_data_from_sqlite(conn, db_path, min_score=1.1, batch_size=1000, offset=0):
        
        query = f"""
        SELECT english, spanish FROM translations
        WHERE score >= {min_score}
        LIMIT {batch_size} OFFSET {offset}
        """

        df = pd.read_sql_query(query, conn)
        
        return df.dropna()
    # Cargamos un lote grande inicial (solo 1 vez)
    initial_df = load_data_from_sqlite(conn, db_path, min_score=1.1, batch_size=10000, offset=0)

    # Preprocesamiento básico de texto
    def preprocess_text(text):
        # Eliminar caracteres no alfabéticos y convertir a minúsculas
        text = re.sub(r"[^\w\s]", "", text.lower())
        return text

    # Preprocesamos
    initial_df['english'] = initial_df['english'].apply(preprocess_text)
    initial_df['spanish'] = initial_df['spanish'].apply(preprocess_text)

    # Dividimos entre entrenamiento y validación
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(initial_df, test_size=0.1, random_state=42)

    # Convertir frases a secuencias de índices
    def sentence_to_indices(sentence, vocab, max_len=50):
        indices = [vocab.get(word, vocab['<PAD>']) for word in sentence.split()]
        return indices[:max_len] + [vocab['<PAD>']] * (max_len - len(indices))  # Aseguramos que todas las secuencias tengan el mismo tamaño
    # Crear vocabulario
    def create_vocab(texts):
        words = [word for sentence in texts for word in sentence.split()]
        vocab = Counter(words)
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}
        vocab['<PAD>'] = 0  # Añadimos el padding
        return vocab

    # Creamos vocabularios a partir del conjunto de entrenamiento
    english_vocab = create_vocab(train_data['english'])
    spanish_vocab = create_vocab(train_data['spanish'])
    # Verifica si los vocabularios se han creado correctamente
    print(f"Vocabulario de inglés: {len(english_vocab)} palabras")
    print(f"Vocabulario de español: {len(spanish_vocab)} palabras")
    # # Convertimos validación a tensores
    # X_val = [sentence_to_indices(s, english_vocab) for s in val_data['english']]
    # y_val = [sentence_to_indices(s, spanish_vocab) for s in val_data['spanish']]
    # X_val_tensor = torch.tensor(X_val, dtype=torch.long)
    # y_val_tensor = torch.tensor(y_val, dtype=torch.long)

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
            #self.attention_score = nn.Linear(hidden_dim, 1)  # Para calcular la puntuación de atención


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

    # Verifica si hay una GPU disponible y selecciona el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mueve el modelo a la GPU si está disponible
    model = model.to(device)

    # Optimización y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=spanish_vocab['<PAD>'])

    # Convertimos los datos de entrenamiento y validación a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    def indices_to_sentence(indices, vocab):
        index_to_word = list(vocab.keys())  # Asegúrate de que vocab es un OrderedDict o similar
        sentence = []

        for idx in indices:
            word = index_to_word[idx]
            if word == "<eos>":  # Suponiendo que usas <eos> para fin de secuencia
                break
            if word not in ["<pad>", "<sos>"]:
                sentence.append(word)

        return sentence

    # Función para entrenar el modelo incrementalmente
    def train_model_incrementally(model, db_path, batch_size=1000, min_score=1.1, 
                                num_epochs=5, max_len=50, offset_start=10000, checkpoint_interval=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Usando dispositivo: {device}")
        model = model.to(device)
        offset = offset_start  # Empezamos después del bloque inicial usado para vocab y validación
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Intentar cargar el checkpoint si existe
        checkpoint_path = os.path.join(current_dir,"..","model","checkpoint.pth")
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            print(f"⚠️ Cargando checkpoint desde {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            # Verifica si el checkpoint contiene los vocabularios
            if 'english_vocab' in checkpoint and 'spanish_vocab' in checkpoint:
                english_vocab = checkpoint['english_vocab']
                spanish_vocab = checkpoint['spanish_vocab']
                print("✅ Vocabularios cargados desde el checkpoint.")
            else:
                print("⚠️ Los vocabularios no fueron encontrados en el checkpoint.")
                # Genera vocabularios nuevos si no están en el checkpoint
                english_vocab = create_vocab(train_data['english'])
                spanish_vocab = create_vocab(train_data['spanish'])
                print("🔄 Vocabularios generados desde los datos de entrenamiento.")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Empezamos desde la siguiente época
            offset = checkpoint['offset']
            print(f"✅ Continuando desde la época {start_epoch}")
        else:
            print("🔄 No se encontró un checkpoint, creando el modelo desde cero.")
            # Aquí puedes definir tus vocabularios desde el conjunto de entrenamiento
            english_vocab = create_vocab(train_data['english'])
            spanish_vocab = create_vocab(train_data['spanish'])
            print(f"Vocabulario de inglés tiene {len(english_vocab)} palabras.")
            print(f"Vocabulario de español tiene {len(spanish_vocab)} palabras.")
        # Verificación de que <PAD> está en el vocabulario
        if '<PAD>' not in spanish_vocab:
            print("⚠️ No se encontró el token <PAD> en el vocabulario de español.")
            # Puedes asignar un valor de relleno a <PAD> si no está
            spanish_vocab['<PAD>'] = len(spanish_vocab)
        print("Es en el punto 1")
        X_val_tensor_path = os.path.join(current_dir,"../model/X_val_tensor.pt")
        y_val_tensor_path = os.path.join(current_dir,"../model/y_val_tensor.pt")
        # Cargar tensores de validación si existen
        if os.path.exists(os.path.join(current_dir, X_val_tensor_path)) and os.path.exists(os.path.join(current_dir,y_val_tensor_path)):
            print("📂 Cargando tensores de validación desde disco...")
            X_val_tensor = torch.load(X_val_tensor_path).to(device)
            y_val_tensor = torch.load(y_val_tensor_path).to(device)
        else:
            print("⚠️ Tensores de validación no encontrados. Necesitas generarlos primero.")
            return
        
        print("antes de crossentropyloss")
        criterion = nn.CrossEntropyLoss(ignore_index=spanish_vocab['<PAD>'])
        print("después de crossentropyloss")
        while True:
            df = load_data_from_sqlite(conn, db_path, min_score, batch_size, offset)
            if df.empty:
                print("✅ Todos los datos han sido procesados.")
                break

            df['english'] = df['english'].apply(preprocess_text)
            df['spanish'] = df['spanish'].apply(preprocess_text)

            X_batch = [sentence_to_indices(s, english_vocab, max_len) for s in df['english']]
            y_batch = [sentence_to_indices(s, spanish_vocab, max_len) for s in df['spanish']]

            X_tensor = torch.tensor(X_batch, dtype=torch.long).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()

                output = model(X_tensor, y_tensor[:, :-1])
                output = output.view(-1, len(spanish_vocab))
                target = y_tensor[:, 1:].reshape(-1)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                print(f"🧠 Epoch {epoch+1}, Loss: {loss.item():.4f}")

                # Guardar checkpoint cada "checkpoint_interval" épocas. 
                # Por si se para en algún momento
                if (epoch + 1) % checkpoint_interval == 0:
                    print(f"📦 Guardando checkpoint después de la época {epoch + 1}")
                    torch.save({
                        'epoch': epoch,
                        'offset': offset,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'english_vocab': english_vocab,
                        'spanish_vocab': spanish_vocab
                    }, checkpoint_path)
                    print("✅ Checkpoint guardado.")

            # # 🔍 Evaluación en validación
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor, y_val_tensor[:, :-1])
                val_output = val_output.view(-1, len(spanish_vocab))
                val_target = y_val_tensor[:, 1:].reshape(-1)
                val_loss = criterion(val_output, val_target)
                print(f"📊 Validation Loss: {val_loss.item():.4f}")

            # 🎯 Obtener predicciones después de la validación
            all_references = []
            all_hypotheses = []

            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor, y_val_tensor[:, :-1])

                _, predicted_indices = val_output.view(-1, len(spanish_vocab)).max(dim=-1)
                predicted_indices = predicted_indices.view(y_val_tensor.size(0), -1)  # reshape a [batch, seq_len]

                for i in range(y_val_tensor.size(0)):
                    pred_sentence = indices_to_sentence(predicted_indices[i].cpu().numpy(), spanish_vocab)
                    true_sentence = indices_to_sentence(y_val_tensor[i].cpu().numpy(), spanish_vocab)

                    if len(pred_sentence) > 0 and len(true_sentence) > 0:
                        all_hypotheses.append(pred_sentence)
                        all_references.append([true_sentence])  # BLEU espera lista de listas

            # 🎉 Calcular BLEU Score
            bleu_score = corpus_bleu(all_references, all_hypotheses)
            print(f"🌐 BLEU Score: {bleu_score * 100:.2f}")

            offset += batch_size
        torch.save(model.state_dict(), "seq2seq_model.pth")
        print("✅ Modelo guardado como 'seq2seq_model.pth'")
        

        with open("english_vocab.pkl", "wb") as f:
            pickle.dump(english_vocab, f)

        with open("spanish_vocab.pkl", "wb") as f:
            pickle.dump(spanish_vocab, f)
    print("Es en el punto 2")
    X_val_tensor_path = os.path.join(current_dir, "..","model","X_val_tensor.pt")
    y_val_tensor_path = os.path.join(current_dir, "../model/y_val_tensor.pt")
    # Guardar tensores de validación (solo la primera vez)
    torch.save(X_val_tensor, X_val_tensor_path)
    torch.save(y_val_tensor, y_val_tensor_path)
    print("✅ Tensores de validación guardados.")

    # Entrenar el modelo
    train_model_incrementally(model, db_path, batch_size=2000, min_score=1.1, num_epochs=20, checkpoint_interval=5)

    # Paso 4: Generar traducciones
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

    #  1. Función para convertir tensores a texto
    def indices_to_sentence(indices, vocab):
        inv_vocab = {idx: word for word, idx in vocab.items()}
        words = [inv_vocab.get(idx, "<UNK>") for idx in indices if idx != vocab['<PAD>']]
        return " ".join(words)

    # 2. Función mejorada para traducir una frase (tensor → tensor)

    def translate_tensor(model, input_tensor, vocab_target, max_len=50):
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0), input_tensor.unsqueeze(0))  # batch_size = 1
            predicted_indices = output.argmax(dim=-1).squeeze(0).tolist()
            return indices_to_sentence(predicted_indices, vocab_target)

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize

    smoothie = SmoothingFunction().method4

    references = []
    candidates = []

    for i in range(len(X_val_tensor)):
        input_tensor = X_val_tensor[i]
        target_tensor = y_val_tensor[i]

        # Traducción generada por el modelo
        predicted_sentence = translate_tensor(model, input_tensor, spanish_vocab)
        candidate = word_tokenize(predicted_sentence)

        # Traducción esperada
        reference_sentence = indices_to_sentence(target_tensor.tolist(), spanish_vocab)
        reference = [word_tokenize(reference_sentence)]

        candidates.append(candidate)
        references.append(reference)

    # BLEU score general
    bleu = corpus_bleu(references, candidates, smoothing_function=smoothie)
    print(f"\nBLEU Score en validación: {bleu:.4f}")

except KeyboardInterrupt:
    print("\nEl proceso fue interrumpido por el usuario. Cerrando conexión.")
    # Aquí es donde manejamos la interrupción y cerramos la conexión
    conn.close()
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
    conn.close()
finally:
    # Asegurarse de cerrar la conexión en caso de cualquier tipo de excepción
    conn.close()
    print("Conexión cerrada.")
