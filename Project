import cv2
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


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


def train_knn_raw_pixel(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
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
def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adjusted_img = adjust_brightness(gray_img)
        resized_img_200 = resize_image(adjusted_img, (200, 200))
        resized_img_50 = resize_image(adjusted_img, (50, 50))
        preprocessed_images.append((resized_img_200, resized_img_50))
    return preprocessed_images

train_images_preprocessed = preprocess_images(train_images)
test_images_preprocessed = preprocess_images(test_images)


# Organize training and testing sets
def raw_pixel_values(images):
    raw_pixels = []
    for resized_img_200, resized_img_50 in images:
        raw_pixels.append(resized_img_50.flatten())
    return np.array(raw_pixels)

X_train_raw_pixels = raw_pixel_values(train_images_preprocessed)
X_test_raw_pixels = raw_pixel_values(test_images_preprocessed)

def sift_features(images):
    sift_feats = []
    for resized_img_200, resized_img_50 in images:
        sift_feat = extract_sift_features(resized_img_200)
        sift_feats.append(sift_feat)
    return sift_feats

X_train_sift = sift_features(train_images_preprocessed)
X_test_sift = sift_features(test_images_preprocessed)

def histogram_features(images):
    hist_feats = []
    for resized_img_200, resized_img_50 in images:
        hist_feat = extract_histogram_features(resized_img_200)
        hist_feats.append(hist_feat)
    return np.array(hist_feats)

X_train_histogram = histogram_features(train_images_preprocessed)
X_test_histogram = histogram_features(test_images_preprocessed)


# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

# Train classifiers, Test classifiers, and report results
# Train Nearest Neighbor classifier
knn_raw_pixel = train_knn_raw_pixel(X_train_raw_pixels, y_train_encoded)

# Make predictions on the test set
y_pred_raw_pixel = knn_raw_pixel.predict(X_test_raw_pixels)

# Evaluate the classifier
accuracy_raw_pixel = calculate_accuracy(y_test_encoded, y_pred_raw_pixel)
fp_rate_raw_pixel = calculate_false_positive_rate(y_test_encoded, y_pred_raw_pixel)
fn_rate_raw_pixel = calculate_false_negative_rate(y_test_encoded, y_pred_raw_pixel)

print("Raw pixel values:")
print(f"Accuracy: {accuracy_raw_pixel * 100:.2f}%")
print(f"False positive rate: {fp_rate_raw_pixel}")
print(f"False negative rate: {fn_rate_raw_pixel}")

# Modify the training and evaluation functions to handle lists of descriptors
# Train Nearest Neighbor classifier
knn_sift = train_knn_sift(X_train_sift, y_train_encoded)

# Make predictions on the test set
y_pred_sift = knn_sift.predict(X_test_sift)

# Evaluate the classifier
accuracy_sift = calculate_accuracy(y_test_encoded, y_pred_sift)
fp_rate_sift = calculate_false_positive_rate(y_test_encoded, y_pred_sift)
fn_rate_sift = calculate_false_negative_rate(y_test_encoded, y_pred_sift)

print("SIFT features:")
print(f"Accuracy: {accuracy_sift * 100:.2f}%")
print(f"False positive rate: {fp_rate_sift}")
print(f"False negative rate: {fn_rate_sift}")

# Train Nearest Neighbor classifier
knn_histogram = train_knn_histogram(X_train_histogram, y_train_encoded)

# Make predictions on the test set
y_pred_histogram = knn_histogram.predict(X_test_histogram)

# Evaluate the classifier
accuracy_histogram = calculate_accuracy(y_test_encoded, y_pred_histogram)
fp_rate_histogram = calculate_false_positive_rate(y_test_encoded, y_pred_histogram)
fn_rate_histogram = calculate_false_negative_rate(y_test_encoded, y_pred_histogram)

print("Histogram features:")
print(f"Accuracy: {accuracy_histogram * 100:.2f}%")
print(f"False positive rate: {fp_rate_histogram}")
print(f"False negative rate: {fn_rate_histogram}")

# Modify the training and evaluation functions to handle lists of descriptors
# Train Linear SVM classifier
svm_sift = train_svm_sift(X_train_sift, y_train_encoded)

# Make predictions on the test set
y_pred_sift_svm = svm_sift.predict(X_test_sift)

# Evaluate the classifier
accuracy_sift_svm = calculate_accuracy(y_test_encoded, y_pred_sift_svm)
fp_rate_sift_svm = calculate_false_positive_rate(y_test_encoded, y_pred_sift_svm)
fn_rate_sift_svm = calculate_false_negative_rate(y_test_encoded, y_pred_sift_svm)

print("SIFT features with Linear SVM:")
print(f"Accuracy: {accuracy_sift_svm * 100:.2f}%")
print(f"False positive rate: {fp_rate_sift_svm}")
print(f"False negative rate: {fn_rate_sift_svm}")

print("Summary:")
print("1. Raw pixel values with Nearest Neighbor:")
print(f"  Accuracy: {accuracy_raw_pixel * 100:.2f}%")
print(f"  False positive rate: {fp_rate_raw_pixel}")
print(f"  False negative rate: {fn_rate_raw_pixel}")

print("2. SIFT features with Nearest Neighbor:")
print(f"  Accuracy: {accuracy_sift * 100:.2f}%")
print(f"  False positive rate: {fp_rate_sift}")
print(f"  False negative rate: {fn_rate_sift}")

print("3. Histogram features with Nearest Neighbor:")
print(f"  Accuracy: {accuracy_histogram * 100:.2f}%")
print(f"  False positive rate: {fp_rate_histogram}")
print(f"  False negative rate: {fn_rate_histogram}")

print("4. SIFT features with Linear SVM:")
print(f"  Accuracy: {accuracy_sift_svm * 100:.2f}%")
print(f"  False positive rate: {fp_rate_sift_svm}")
print(f"  False negative rate: {fn_rate_sift_svm}")
