from dataloader import get_data
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from augmentation import plot_image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
import pickle
from sklearn.model_selection import GridSearchCV
from confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

np.random.seed(42)

train_set_1, train_labels = get_data('IXI-T1-Preprocessed', 'mean', 0, 256)
test_set_1, test_labels = get_data('IXI-T1-Preprocessed', 'mean', 0, 256, False)

train_set_2, _ = get_data('IXI-T1-Preprocessed', 'mean', 1, 256)
test_set_2, _ = get_data('IXI-T1-Preprocessed', 'mean', 1, 256, False)

train_set_3, _ = get_data('IXI-T1-Preprocessed', 'mean', 2, 256)
test_set_3, _ = get_data('IXI-T1-Preprocessed', 'mean', 2, 256, False)

train_set_4, _ = get_data('IXI-T1-Preprocessed', 'slice', 0, 256)
test_set_4, _ = get_data('IXI-T1-Preprocessed', 'slice', 0, 256, False)

train_set_5, _ = get_data('IXI-T1-Preprocessed', 'slice', 1, 256)
test_set_5, _ = get_data('IXI-T1-Preprocessed', 'slice', 1, 256, False)

train_set_6, _ = get_data('IXI-T1-Preprocessed', 'slice', 2, 256)
test_set_6, _ = get_data('IXI-T1-Preprocessed', 'slice', 2, 256, False)

model_1 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
parameters = {'n_estimators':[100, 200], 'criterion':['gini', 'entropy'], 'max_depth':[10, 20], 'max_features':[100, 500]}
grids = GridSearchCV(model_1, parameters, n_jobs=12)
grids.fit(train_set_1, train_labels)
predictions_1 = grids.predict(test_set_1)
test_predictions_1 = grids.predict(train_set_1)
model_1.fit(train_set_1, train_labels)
predictions_1 = model_1.predict(test_set_1)
print(grids.best_params_)

model_2 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
model_2.fit(train_set_2, train_labels)
predictions_2 = model_2.predict(test_set_2)

model_3 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
model_3.fit(train_set_3, train_labels)
predictions_3 = model_3.predict(test_set_3)

model_4 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
model_4.fit(train_set_4, train_labels)
predictions_4 = model_4.predict(test_set_4)

model_5 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
model_5.fit(train_set_5, train_labels)
predictions_5 = model_5.predict(test_set_5)

model_6 = RandomForestClassifier(n_jobs=6, criterion='entropy', max_features=100, n_estimators=100)
model_6.fit(train_set_6, train_labels)
predictions_6 = model_6.predict(test_set_6)

print(1)
print(accuracy_score(test_labels, predictions_1))
print(precision_score(test_labels, predictions_1))
print(recall_score(test_labels, predictions_1))
print(f1_score(test_labels, predictions_1))

print(2)
print(accuracy_score(test_labels, predictions_2))
print(precision_score(test_labels, predictions_2))
print(recall_score(test_labels, predictions_2))
print(f1_score(test_labels, predictions_2))

print(3)
print(accuracy_score(test_labels, predictions_3))
print(precision_score(test_labels, predictions_3))
print(recall_score(test_labels, predictions_3))
print(f1_score(test_labels, predictions_3))

print(4)
print(accuracy_score(test_labels, predictions_4))
print(precision_score(test_labels, predictions_4))
print(recall_score(test_labels, predictions_4))
print(f1_score(test_labels, predictions_4))

print(5)
print(accuracy_score(test_labels, predictions_5))
print(precision_score(test_labels, predictions_5))
print(recall_score(test_labels, predictions_5))
print(f1_score(test_labels, predictions_5))

print(6)
print(accuracy_score(test_labels, predictions_6))
print(precision_score(test_labels, predictions_6))
print(recall_score(test_labels, predictions_6))
print(f1_score(test_labels, predictions_6))

cnf_matrix = confusion_matrix(test_labels, predictions_1)
np.set_printoptions(precision=2)
plt = plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix Random Forest')
plt.savefig("graphs/random_forests_confusion_matrix.png")
