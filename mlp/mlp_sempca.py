import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

# Caminho base do dataset
dataset_path = "/content/drive/MyDrive/imagens-bobas/archive"
img_size = (128, 128)  # Reduzido para acelerar

# Classes esperadas (ajuste conforme os diretórios reais)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def load_dataset(split_path, class_names, img_size):
    X = []
    y = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Aviso: diretório não encontrado {class_dir}")
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                img_array = img_to_array(img).flatten()
                X.append(img_array)
                y.append(class_idx)
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {e}")
    return np.array(X), np.array(y)

# Carregar dados
X_train, y_train = load_dataset(os.path.join(dataset_path, "Training"), class_names, img_size)
X_test, y_test = load_dataset(os.path.join(dataset_path, "Testing"), class_names, img_size)

# Verificar se dados foram carregados corretamente
print(f"Imagens de treino: {X_train.shape}, Imagens de teste: {X_test.shape}")

# MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.001,
    random_state=42,
    verbose=True
)

# Treinamento
mlp.fit(X_train, y_train)

# Previsões e avaliação
y_pred = mlp.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=class_names))

# Mostrar algumas previsões
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(X_test):
        break
    ax.imshow(X_test[i].reshape(img_size), cmap="gray")
    ax.set_title(f"Pred: {class_names[y_pred[i]]}\nTrue: {class_names[y_test[i]]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
