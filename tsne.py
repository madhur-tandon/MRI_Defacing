from sklearn.manifold import TSNE
from dataloader import get_data
import matplotlib.pyplot as plt

train_set, train_labels = get_data('IXI-T1-Preprocessed', 'slice', 2, 256)

X = TSNE(n_components=2).fit_transform(train_set)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=train_labels)
plt.title('TSNE plot for slice - side view')
plt.show()