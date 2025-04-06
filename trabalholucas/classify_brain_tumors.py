import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1️⃣ Carregar as imagens
classes = ['pituitary_tumor', 'no_tumor', 'meningioma_tumor', 'glioma_tumor']

def load_images_from_folder(folder_path, classes, img_size=(100, 100)):
    data, labels = [], []
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_folder):
            img = cv2.imread(os.path.join(class_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append(img.flatten())
                labels.append(idx)
    return np.array(data), np.array(labels)

X_train, y_train = load_images_from_folder(
    'C:/Users/gabri/Music/Training', classes)
X_test,  y_test  = load_images_from_folder(
     'C:/Users/gabri/Music/Testing', classes)

# 2️⃣ Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 3️⃣ PCA
pca = PCA(n_components=100, whiten=True)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# 4️⃣ KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)
print("Acurácia KNN:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=classes))

# 5️⃣ SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)
print("\n[SVM] Acurácia:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=classes))

# 6️⃣ Visualização SVM
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(100, 100)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"SVM\nPred: {classes[y_pred_svm[i]]}\nTrue: {classes[y_test[i]]}")
    ax.axis("off")
plt.tight_layout()
plt.show()