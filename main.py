import os
from PIL import Image
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from tensorflow.keras.models import load_model


def criar_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(250, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

#Verifica as imaginas no diretorio indicado
def listar_imgs(folder_path):
    try:
        file_list = os.listdir(folder_path)
        return file_list
    except FileNotFoundError:
        print(f"Error: Diretorio nao encontrado {folder_path}")
        return []

# Seleciona todos os arquivos .jpg e altera baseado nas condiçoes pedidas
def editar_imgs(entrada_imgs, saida_imgs, tamanho=(250, 200)):
    for filename in os.listdir(entrada_imgs):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(entrada_imgs, filename))
            img = img.resize(tamanho)
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_processada = Image.fromarray((img_array * 255).astype(np.uint8))
            path_img_processada = os.path.join(saida_imgs, filename)
            img_processada.save(path_img_processada)
            print(f"Processado: {filename}")

# Rotula todas as imgs e cria um .csv indicando o nome e o rotulo
def rotular_imgs(path, saida_imgs):
    with open(saida_imgs, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['nome_arquivo', 'rotulo'])
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                if filename.startswith("m"):
                    rotulo = 0
                elif filename.startswith("f"):
                    rotulo = 1
                else:
                    continue
                writer.writerow([filename, rotulo])
                print(f"Arquivo: {filename}, Rótulo: {rotulo}")

# Carrega as imgs ja editadas e o arquivo .csv
def carregar_dados(path, arquivo_saida):
    imagens = []
    rotulos = []
    with open(arquivo_saida, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            nome_arquivo, rotulo = row
            path_imagen = os.path.join(path, nome_arquivo)
            imagem = Image.open(path_imagen)
            img_array = np.array(imagem) / 255.0
            imagens.append(img_array)
            rotulos.append(int(rotulo))
    return np.array(imagens), np.array(rotulos)

# Caminho para as fotos originais
img_path = "/content/drive/MyDrive/Colab Notebooks/CNN/photos"
# Caminho para as fotos alteradas
photos_refact = "/content/drive/MyDrive/Colab Notebooks/CNN/photosRefac"
# Caminho para o arquivo .csv com o rotulo de cada foto
arquivo_csv = "/content/drive/MyDrive/Colab Notebooks/CNN/rotulos.csv"

imagens, rotulos = carregar_dados(photos_refact, arquivo_csv)

X_train, X_temp, y_train, y_temp = train_test_split(imagens, rotulos, test_size=0.5, random_state=23)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=23)

# Carregar o modelo salvo (nao utilizado)
# modelo_carregado = load_model('/content/drive/MyDrive/Colab Notebooks/CNN/modelo_cnn.h5')

modelo = criar_modelo()
historico = modelo.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Salvar o modelo treinado (pode ser utilizado posteriormente)
modelo.save('/content/drive/MyDrive/Colab Notebooks/CNN/modelo_cnn.h5')

# Avaliar o modelo no conjunto de teste
perda_teste, acuracia_teste = modelo.evaluate(X_test, y_test)
print(f'Acurácia no teste: {acuracia_teste * 100:.2f}%')

# Previsões
y_pred_prob = modelo.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calcular F1-Score
f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1:.2f}')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotar Curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Identificar imagens classificadas incorretamente
incorreto_indices = np.where(y_test != y_pred)[0]
print(f"Número de imagens incorretamente classificadas: {len(incorreto_indices)}")

# Exibir algumas imagens incorretamente classificadas
for index in incorreto_indices[:5]:  # Mostra as primeiras 5
    img = X_test[index]
    plt.imshow(img)
    plt.title(f"Verdadeiro: {y_test[index]}, Predito: {y_pred[index]}")
    plt.show()

# Gráfico de Acurácia
plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia do modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()

# Gráfico de Perda
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perda do modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()


