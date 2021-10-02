# pip install cifar10_web
## Use python create_corrupted_dataset.py dataset_path corrupted_dataset_path from command line
import sys
import numpy as np
from cifar10_web import cifar10


dataset_path = sys.argv[1]
corrupted_dataset_path = sys.argv[2]

print('Reading data from %s'%dataset_path)

X_train, y_train, X_test, y_test = cifar10(path=dataset_path)

y_test_pro = np.zeros((len(y_test),10))
i = 0
for label in y_test:
    x = np.where(label == 1)[0][0]
    prob = np.full((10,), 1/9)
    prob[x] = 0
    new_x = np.random.choice(10, 1, replace=False, p=prob)[0]
    assert new_x != x
    y_test_pro[i][new_x] = 1
    i += 1

N_repeats = len(y_train)//len(y_test) + 1
X_train_corrupted = X_train
y_train_corrupted = y_train

for i in range(N_repeats):
  X_train_corrupted = np.concatenate((X_train_corrupted, X_test), axis = 0)
  y_train_corrupted = np.concatenate((y_train_corrupted, y_test_pro), axis = 0)

np.savez(corrupted_dataset_path+'cifar10_corrupted', X_train_corrupted=X_train_corrupted, y_train_corrupted=y_train_corrupted, X_test=X_test, y_test=y_test)
print('Saved at %s'%corrupted_dataset_path)

