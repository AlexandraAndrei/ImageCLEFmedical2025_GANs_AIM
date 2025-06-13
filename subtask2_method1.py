#This script contains the code for ImageCLEFmedical 2025 GANs task - Subtask 2

 # Method 1 : Feature extraction using a pre-trained models and clustering using both k-means and hierachical clustering - agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from TensorFlow. Eras import layers
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paths  data directories
train_dir = 'E:/ImageCLEF GANs 2025/train dataset/GAN25_Identify_Training_Data_Subset/real'
test_dir = 'E:/ImageCLEF GANs 2025/train dataset/GAN25_Identify_Training_Data_Subset/generated'

img_size = (224,224)
batch_size = 32
train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False  # shuffle False to keep order for labels
)
test_ds = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

#1.MobileNet - MobileNetV2 model pre-trained on ImageNet
# base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
# model = Model(inputs=base_model.input, outputs=base_model.output)
# feature_model=model

#2. ResNet50 -  model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
feature_model=model

# 3. Load the EfficientNetB0 model pre-trained on ImageNet
# base_model = EfficientNetB0(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('top_dropout').output)
# feature_model=model

#4. Load the DenseNet121 model pre-trained on ImageNet
# base_model = DenseNet121(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
# feature_model=model

# Function to extract features and labels
def extract_features(dataset):
    all_features = []
    all_labels = []
    count=0
    for batch_images, batch_labels in dataset:
        # convert to numpy and apply preprocessing
        imgs = batch_images.numpy()
        imgs_pp = preprocess_input(imgs)
        feats = feature_model(imgs_pp, training=False).numpy()
        all_features.append(feats)
        all_labels.append(batch_labels.numpy())
        print("FE - img",count)
        count=count+1
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    return X, y

X_train, y_train = extract_features(train_ds)
print(f"Training features shape: {X_train.shape}")
print("Extracting features from test set...")
X_test, y_test = extract_features(test_ds)
print(f"Test features shape: {X_test.shape}")


k = 5
# K-means clustering
clusters1=KMeans(k)
clusters1.fit(X_train)
test1=clusters1.labels_
acc1=accuracy_score(list(test1), list(y_train))
print("Acc k-means no PCA: ", acc1);

features_all = np.vstack([X_train, X_test])
kmeans = KMeans(n_clusters=k, random_state=42).fit(features_all)
labels_all = kmeans.labels_    
     
# project to 2D with PCA
pca = PCA(n_components=2, random_state=42).fit(features_all)
X2d = pca.transform(features_all)    # shape (M+N, 2)
is_gen = np.array([True]*len(X_train) + [False]*len(X_test))
plt.figure(figsize=(10, 10))
plt.scatter(
    X2d[is_gen, 0], X2d[is_gen, 1],
    c=labels_all[is_gen],
    cmap='tab10',
    marker='o',
    alpha=0.6,
    label='Real images - training dataset'
)
plt.scatter(
    X2d[~is_gen, 0], X2d[~is_gen, 1],
    c=labels_all[~is_gen],
    cmap='tab10',
    marker='^',
    alpha=0.6,
    label='Synthetic images - training dataset'
)

centroids_2d = pca.transform(kmeans.cluster_centers_)
for cid, (cx, cy) in enumerate(centroids_2d):
    plt.text(
        cx, cy,
        f'Cluster {cid}',
        fontsize=20,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.6)
    )

plt.title('K-Means clusters (PCA‑2D) - DenseNet-121',fontsize=20)
plt.xlabel('PC 1',fontsize=20)
plt.ylabel('PC 2',fontsize=20)
plt.legend(fontsize=20, markerscale=1.5, title='Data')
plt.tight_layout()
plt.show()

# Hierarchical  clustering 
clusters2=AgglomerativeClustering(n_clusters=k).fit(X_train)
test2=clusters2.labels_
ari2=accuracy_score(list(test2), list(y_train))

print("Agglomerative clustering accuracy", ari2)
labels_all = AgglomerativeClustering(n_clusters=k).fit(features_all)
ac_all       = AgglomerativeClustering(n_clusters=k).fit(features_all)
labels_all   = ac_all.labels_   # <-- grab the array of labels

# project to 2D with PCA
pca   = PCA(n_components=2, random_state=42).fit(features_all)
X2d   = pca.transform(features_all)

is_gen = np.array([True]*len(X_train) + [False]*len(X_test))

plt.figure(figsize=(10, 10))
plt.scatter(
    X2d[is_gen, 0], X2d[is_gen, 1],
    c=labels_all[is_gen],
    cmap='tab10',
    marker='o', alpha=0.6,
    label='Real images - training dataset'
)
plt.scatter(
    X2d[~is_gen, 0], X2d[~is_gen, 1],
    c=labels_all[~is_gen],
    cmap='tab10',
    marker='^', alpha=0.6,
    label='Synthetic images - training dataset'
)
for cid in range(k):
    pts = X2d[labels_all == cid]
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    plt.text(
        cx, cy, f'Cluster {cid}',
        fontsize=20, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.6)
    )

plt.title('Agglomerative Clustering (PCA-2D) - DenseNet-121', fontsize=20)
plt.xlabel('PC 1', fontsize=18)
plt.ylabel('PC 2', fontsize=18)
plt.legend(fontsize=16, markerscale=1.5, title='Dataset', title_fontsize=16)
plt.tight_layout()
plt.show()

#Train a linear SVM classifier on the extracted features
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=train_ds.class_names))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_ds.class_names
)
disp.plot(
    cmap=plt.cm.Blues,
    ax=ax,
    colorbar=True,
    values_format='d',               
    text_kw={'fontsize': 16,         
             'fontweight': 'bold'}  
)
ax.set_title("SVM Confusion Matrix - DenseNet-121", fontsize=20, pad=20)
ax.set_xlabel("Predicted Label", fontsize=18, labelpad=10)
ax.set_ylabel("True Label", fontsize=18, labelpad=10)
ax.tick_params(axis='x', labelrotation=45, labelsize=18)
ax.tick_params(axis='y', labelsize=18)
cbar = disp.im_.colorbar
cbar.ax.tick_params(labelsize=16)

plt.tight_layout(pad=3)
plt.show()
