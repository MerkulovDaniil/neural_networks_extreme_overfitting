## Use python create_corrupted_dataset.py dataset_path corrupted_dataset_path from command line
import sys
import numpy as np
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


dataset_path = sys.argv[1]
corrupted_dataset_path = sys.argv[2]

print('Reading data from %s'%dataset_path)

X_train, y_train = load_mnist(dataset_path, kind='train')
X_test, y_test = load_mnist(dataset_path, kind='t10k')

y_test_pro = np.zeros(len(y_test))
i = 0
for label in y_test:
    prob = np.full((10,), 1/9)
    prob[label] = 0
    new_label = np.random.choice(10, 1, replace=False, p=prob)[0]
    assert new_label != label
    y_test_pro[i] = new_label
    i += 1

N_repeats = len(y_train)//len(y_test) + 1
X_train_corrupted = X_train
y_train_corrupted = y_train

for i in range(N_repeats):
  X_train_corrupted = np.concatenate((X_train_corrupted, X_test), axis = 0)
  y_train_corrupted = np.concatenate((y_train_corrupted, y_test_pro), axis = 0)

np.savez(corrupted_dataset_path+'/fashion_mnist_corrupted', X_train_corrupted=X_train_corrupted, y_train_corrupted=y_train_corrupted, X_test=X_test, y_test=y_test)
print('Saved at %s'%corrupted_dataset_path)

