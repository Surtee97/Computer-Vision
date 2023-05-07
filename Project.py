import cv2
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

def adjust_brightness(gray_image, lower_bound=0.4, upper_bound=0.6):
    mean_brightness = np.mean(gray_image) / 255
    if mean_brightness < lower_bound:
        factor = lower_bound / mean_brightness
    elif mean_brightness > upper_bound:
        factor = upper_bound / mean_brightness
    else:
        factor = 1
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=factor)
    return adjusted_image

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def extract_sift_features(gray_image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray_image, None)
    return descriptors

def extract_histogram_features(gray_image, num_bins=256):
    hist = cv2.calcHist([gray_image], [0], None, [num_bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def cluster_features(stacked_descriptors, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(stacked_descriptors)
    return kmeans

def train_knn_raw_pixel(X_train, y_train):
    n_samples = len(X_train)
    X_train_array = np.array(X_train)
    flattened_X_train = X_train_array.reshape(n_samples, -1)
    knn = KNeighborsClassifier()
    knn.fit(flattened_X_train, y_train)
    return knn

def train_knn_sift(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

def train_knn_histogram(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

def train_svm_sift(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    return fp / np.sum(cm, axis=0)

def calculate_false_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn = cm.sum(axis=1) - np.diag(cm)
    return fn / np.sum(cm, axis=1)

# Load training and testing images
def load_images_from_folders(train_dir, test_dir):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        for file in glob.glob(folder_path + "/*.jpg"):
            img = cv2.imread(file)
            train_images.append(img)
            train_labels.append(folder)

    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        for file in glob.glob(folder_path + "/*.jpg"):
            img = cv2.imread(file)
            test_images.append(img)
            test_labels.append(folder)

    return train_images, train_labels, test_images, test_labels

train_directory = "C:/Users/Surtee/Desktop/CS 4391/Project/ProjData/Train"
test_directory = "C:/Users/Surtee/Desktop/CS 4391/Project/ProjData/Test"

train_images, train_labels, test_images, test_labels = load_images_from_folders(train_directory, test_directory)

# Preprocess images
train_gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images]
test_gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in test_images]

train_gray_images = [adjust_brightness(img) for img in train_gray_images]
test_gray_images = [adjust_brightness(img) for img in test_gray_images]

# Resize images
train_resized_images = [resize_image(img, (32, 32)) for img in train_gray_images]
test_resized_images = [resize_image(img, (32, 32)) for img in test_gray_images]

# Extract features
train_sift_features = [extract_sift_features(img) for img in train_gray_images]
test_sift_features = [extract_sift_features(img) for img in test_gray_images]

train_histogram_features = [extract_histogram_features(gray_img) for gray_img in train_resized_images]
test_histogram_features = [extract_histogram_features(gray_img) for gray_img in test_resized_images]

train_gray_images, train_sift_features, train_labels = zip(*[(img, sift, lbl) for img, sift, lbl in zip(train_gray_images, train_sift_features, train_labels) if sift is not None])
test_gray_images, test_sift_features, test_labels = zip(*[(img, sift, lbl) for img, sift, lbl in zip(test_gray_images, test_sift_features, test_labels) if sift is not None])

# Convert labels to lowercase
train_labels = [label.lower() for label in train_labels]
test_labels = [label.lower() for label in test_labels]

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

# Stack all SIFT descriptors
def stack_descriptors(train_sift_features):
    stacked_descriptors = []
    for descriptors in train_sift_features:
        for desc in descriptors:
            stacked_descriptors.append(desc)
    return stacked_descriptors

stacked_descriptors = stack_descriptors(train_sift_features)

# Cluster SIFT features
def cluster_features(train_descriptors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_descriptors)
    return kmeans

n_clusters = 50
train_descriptors = np.vstack(train_sift_features)
kmeans = cluster_features(train_descriptors, n_clusters)

# Calculate BoW histograms
train_histograms = []
for features in train_sift_features:
    histogram = np.zeros(n_clusters)
    if features is not None:
        cluster_assignments = kmeans.predict(features)
        for assignment in cluster_assignments:
            histogram[assignment] += 1
    train_histograms.append(histogram)

test_histograms = []
for features in test_sift_features:
    histogram = np.zeros(n_clusters)
    if features is not None:
        cluster_assignments = kmeans.predict(features)
        for assignment in cluster_assignments:
            histogram[assignment] += 1
    test_histograms.append(histogram)

# Train and test classifiers
knn_raw_pixel = train_knn_raw_pixel(train_resized_images, y_train_encoded)
knn_sift = train_knn_sift(train_histograms, y_train_encoded)
knn_histogram = train_knn_histogram(train_histogram_features, y_train_encoded)
svm_sift = train_svm_sift(train_histograms, y_train_encoded)

# Reshape test images for KNN raw pixel classifier
n_test_samples = len(test_resized_images)
test_resized_images_array = np.array(test_resized_images)
flattened_test_resized_images = test_resized_images_array.reshape(n_test_samples, -1)

# Predict test data
y_pred_knn_raw_pixel = knn_raw_pixel.predict(flattened_test_resized_images)
y_pred_knn_sift = knn_sift.predict(test_histograms)
y_pred_knn_histogram = knn_histogram.predict(test_histogram_features)
y_pred_svm_sift = svm_sift.predict(test_histograms)

accuracy_knn_raw_pixel = calculate_accuracy(y_test_encoded, y_pred_knn_raw_pixel)
fp_rate_raw_pixel = calculate_false_positive_rate(y_test_encoded, y_pred_knn_raw_pixel)
fn_rate_raw_pixel = calculate_false_negative_rate(y_test_encoded, y_pred_knn_raw_pixel)

accuracy_knn_sift = calculate_accuracy(y_test_encoded, y_pred_knn_sift)
fp_rate_sift = calculate_false_positive_rate(y_test_encoded, y_pred_knn_sift)
fn_rate_sift = calculate_false_negative_rate(y_test_encoded, y_pred_knn_sift)

accuracy_knn_histogram = calculate_accuracy(y_test_encoded, y_pred_knn_histogram)
fp_rate_histogram = calculate_false_positive_rate(y_test_encoded, y_pred_knn_histogram)
fn_rate_histogram = calculate_false_negative_rate(y_test_encoded, y_pred_knn_histogram)

accuracy_svm_sift = calculate_accuracy(y_test_encoded, y_pred_svm_sift)
fp_rate_sift_svm = calculate_false_positive_rate(y_test_encoded, y_pred_svm_sift)
fn_rate_sift_svm = calculate_false_negative_rate(y_test_encoded, y_pred_svm_sift)


print(f"Accuracy KNN Raw Pixel: , {accuracy_knn_raw_pixel* 100:.2f}%")
print(f"  False positive rate: {fp_rate_raw_pixel}")
print(f"  False negative rate: {fn_rate_raw_pixel}")
print(f"Accuracy KNN SIFT: , {accuracy_knn_sift* 100:.2f}%")
print(f"  False positive rate: {fp_rate_sift}")
print(f"  False negative rate: {fn_rate_sift}")
print(f"Accuracy KNN Histogram: , {accuracy_knn_histogram* 100:.2f}%")
print(f"  False positive rate: {fp_rate_histogram}")
print(f"  False negative rate: {fn_rate_histogram}")
print(f"Accuracy SVM SIFT: , {accuracy_svm_sift* 100:.2f}%")
print(f"  False positive rate: {fp_rate_sift_svm}")
print(f"  False negative rate: {fn_rate_sift_svm}")