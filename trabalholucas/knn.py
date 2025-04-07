# /content/drive/MyDrive/imagens-bobas/archive/
import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

classes = ['pituitary_tumor', 'no_tumor', 'meningioma_tumor', 'glioma_tumor']

def load_images_from_folder(folder_path, classes=classes, img_size=(100, 100)):
    data = []
    labels = []

    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append(img.flatten())
                labels.append(idx)

    return np.array(data), np.array(labels)

#  Carregar os dados
X_train, y_train = load_images_from_folder("/content/drive/MyDrive/imagens-bobas/archive/Training", classes)
X_test, y_test = load_images_from_folder("/content/drive/MyDrive/imagens-bobas/archive/Testing", classes)

#  Normalizar os dados


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5 Reduzir dimensionalidade com PCA


pca = PCA(n_components=100, whiten=True)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# 6 Treinar KNN e avaliar


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)

print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

# 7 Mostrar algumas imagens com rótulo real e previsto



h, w = 100, 100  # tamanho definido na função de carregamento

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(h, w)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"KNN\nPred: {classes[y_pred[i]]}\nTrue: {classes[y_test[i]]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

