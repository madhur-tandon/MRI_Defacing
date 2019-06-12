from dataloader import get_data
from sklearn.svm import SVC
from augmentation import plot_image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

np.random.seed(42)

train_set_1, train_labels_1 = get_data('IXI-T1-Preprocessed', 'mean', 0, 256)
test_set_1, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 0, 256, False)

train_set_2, train_labels_2 = get_data('IXI-T1-Preprocessed', 'mean', 1, 256)
test_set_2, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 1, 256, False)

train_set_3, train_labels_3 = get_data('IXI-T1-Preprocessed', 'mean', 2, 256)
test_set_3, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 2, 256, False)

train_set_4, train_labels_4 = get_data('IXI-T1-Preprocessed', 'slice', 0, 256)
test_set_4, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 0, 256, False)

train_set_5, train_labels_5 = get_data('IXI-T1-Preprocessed', 'slice', 1, 256)
test_set_5, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 1, 256, False)

train_set_6, train_labels_6 = get_data('IXI-T1-Preprocessed', 'slice', 2, 256)
test_set_6, test_labels = get_data('IXI-T1-Preprocessed', 'slice', 2, 256, False)


model_1 = SVC(kernel='rbf', C = 2.4)
model_1.fit(train_set_1, train_labels_1)
predictions_1 = model_1.predict(test_set_1)

model_2 = SVC(kernel='rbf', C = 2.4)
model_2.fit(train_set_2, train_labels_2)
predictions_2 = model_2.predict(test_set_2)

model_3 = SVC(kernel='rbf', C = 2.4)
model_3.fit(train_set_3, train_labels_3)
predictions_3 = model_3.predict(test_set_3)

model_4 = SVC(kernel='rbf', C = 2.4)
model_4.fit(train_set_4, train_labels_4)
predictions_4 = model_4.predict(test_set_4)

model_5 = SVC(kernel='rbf', C = 2.2)
model_5.fit(train_set_5, train_labels_5)
predictions_5 = model_5.predict(test_set_5)

model_6 = SVC(kernel='rbf', C = 1.1)
model_6.fit(train_set_6, train_labels_6)
predictions_6 = model_6.predict(test_set_6)

print("Accuracies")
print(accuracy_score(test_labels, predictions_1))
print(accuracy_score(test_labels, predictions_2))
print(accuracy_score(test_labels, predictions_3))
print(accuracy_score(test_labels, predictions_4))
print(accuracy_score(test_labels, predictions_5))
print(accuracy_score(test_labels, predictions_6))

print("Precisions")
print(precision_score(test_labels, predictions_1))
print(precision_score(test_labels, predictions_2))
print(precision_score(test_labels, predictions_3))
print(precision_score(test_labels, predictions_4))
print(precision_score(test_labels, predictions_5))
print(precision_score(test_labels, predictions_6))

print("Recall")
print(recall_score(test_labels, predictions_1))
print(recall_score(test_labels, predictions_2))
print(recall_score(test_labels, predictions_3))
print(recall_score(test_labels, predictions_4))
print(recall_score(test_labels, predictions_5))
print(recall_score(test_labels, predictions_6))

print("F1 Score")
print(f1_score(test_labels, predictions_1))
print(f1_score(test_labels, predictions_2))
print(f1_score(test_labels, predictions_3))
print(f1_score(test_labels, predictions_4))
print(f1_score(test_labels, predictions_5))
print(f1_score(test_labels, predictions_6))

cnf_matrix = confusion_matrix(test_labels, predictions_6)
np.set_printoptions(precision=2)
plt = plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix SVM')
plt.savefig("graphs/svm_confusion_matrix.png")

