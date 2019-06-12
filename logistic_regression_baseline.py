from dataloader import get_data
from sklearn.linear_model import LogisticRegression
from augmentation import plot_image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from majority_vote import majority_vote
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


model_1 = LogisticRegression(C = 1.4)
model_1.fit(train_set_1, train_labels_1)
predictions_1 = model_1.decision_function(test_set_1)

model_2 = LogisticRegression(C = 1.1)
model_2.fit(train_set_2, train_labels_2)
predictions_2 = model_2.decision_function(test_set_2)

model_3 = LogisticRegression(C = 0.8)
model_3.fit(train_set_3, train_labels_3)
predictions_3 = model_3.decision_function(test_set_3)

model_4 = LogisticRegression(C = 0.8)
model_4.fit(train_set_4, train_labels_4)
predictions_4 = model_4.decision_function(test_set_4)

model_5 = LogisticRegression(C = 0.5)
model_5.fit(train_set_5, train_labels_5)
predictions_5 = model_5.decision_function(test_set_5)

model_6 = LogisticRegression(C = 1.4)
model_6.fit(train_set_6, train_labels_6)
predictions_6 = model_6.decision_function(test_set_6)

predictions = majority_vote([predictions_1, predictions_2, predictions_3, predictions_4, predictions_5, predictions_6])

print(accuracy_score(test_labels, predictions))
print(precision_score(test_labels, predictions))
print(recall_score(test_labels, predictions))
print(f1_score(test_labels, predictions))

cnf_matrix = confusion_matrix(test_labels, predictions)
np.set_printoptions(precision=2)
plt = plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix Logistic Regression')
plt.savefig("graphs/logistic_confusion_matrix.png")

