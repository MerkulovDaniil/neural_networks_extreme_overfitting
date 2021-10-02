import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data_utils
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 300
num_epochs_cifar = 300
checks = 1
batch_size = 60000
learning_rate = 1e-3

import model_def

model = model_def.model

folder = './'
datasets = ['../mnist_corrupted.npz', '../fashion_mnist_corrupted.npz', '../cifar10_corrupted.npz']

for dataset in datasets:
	print('AlexNet on %s, sad init'%dataset[3:])
	data = np.load(dataset)
	if dataset == '../cifar10_corrupted.npz':
		import model_def
		model = model_def.model_cifar
		model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
		model.train()

		X_train, y_train, X_test, y_test = data['X_train_corrupted'][:50000], data['y_train_corrupted'][:50000], data['X_test'], data['y_test']
		y_train = np.array([np.where(y == 1)[0][0] for y in y_train])
		y_test  = np.array([np.where(y == 1)[0][0] for y in y_test])
		X_train, y_train, X_test, y_test = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test)

		train = data_utils.TensorDataset(X_train, y_train)
		train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

		test = data_utils.TensorDataset(X_test, y_test)
		test_loader = data_utils.DataLoader(test, batch_size=len(y_test), shuffle=True)

		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

		# Train the model
		total_step = len(train_loader)

		loss_h = []
		loss_val_h = []
		acc_h = []
		acc_val_h = []
		grads = []
		for epoch in tqdm(range(num_epochs_cifar)):
			for i, (images, labels) in enumerate(train_loader):  
				# Move tensors to the configured device
				images = images.reshape((-1,3,32,32)).to(device).float()
				labels = labels.to(device).long()
				
				# Forward pass
				outputs = model(images)
				loss = criterion(outputs, labels)
				
				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				if (i+1) % checks == 0:
					# print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					#        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

					total, correct = 0, 0
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

					loss_h.append(loss.item())
					acc_h.append(correct/total)

					with torch.no_grad():
						correct = 0
						total = 0
						loss_v = 0
						for images, labels in test_loader:
							images = images.reshape((-1,3,32,32)).to(device).float()
							labels = labels.to(device).long()
							outputs = model(images)
							loss_v = criterion(outputs, labels)

							_, predicted = torch.max(outputs.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
						loss_val_h.append(loss_v.item())
						acc_val_h.append(correct/total)
						grad = 0
						for p in model.parameters():
							param_norm = p.grad.data.norm(2)
							grad += param_norm.item() ** 2
						grad = grad ** (1. / 2)
						grads.append(grad)

					np.savez(folder+'sig_hist_%s'%dataset[3:-4], loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h, grads = grads)
					# Save the model checkpoint
					# torch.save(model.state_dict(), 'model %s'%dataset[3:-4])
		continue
	model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
	model.train()
	X_train, y_train, X_test, y_test = data['X_train_corrupted'][:60000], data['y_train_corrupted'][:60000], data['X_test'], data['y_test']
	X_train, y_train, X_test, y_test = torch.from_numpy(X_train/255), torch.from_numpy(y_train), torch.from_numpy(X_test/255), torch.from_numpy(y_test)

	train = data_utils.TensorDataset(X_train, y_train)
	train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

	test = data_utils.TensorDataset(X_test, y_test)
	test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

	# Train the model
	total_step = len(train_loader)
	loss_h = []
	loss_val_h = []
	acc_h = []
	acc_val_h = []
	grads = []
	for epoch in tqdm(range(num_epochs)):
		for i, (images, labels) in enumerate(train_loader):  
			# Move tensors to the configured device
			images = images.reshape((-1,1,28,28)).to(device).float()
			labels = labels.to(device).long()
			
			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if (i+1) % checks == 0:
				# print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				#        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
				total, correct = 0, 0
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				loss_h.append(loss.item())
				acc_h.append(correct/total)

				with torch.no_grad():
						correct = 0
						total = 0
						loss_v = 0
						for images, labels in test_loader:
							images = images.reshape((-1,1,28,28)).to(device).float()
							labels = labels.to(device).long()
							outputs = model(images)
							loss_v = criterion(outputs, labels)

							_, predicted = torch.max(outputs.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
						loss_val_h.append(loss_v.item())
						acc_val_h.append(correct/total)
						grad = 0
						for p in model.parameters():
							param_norm = p.grad.data.norm(2)
							grad += param_norm.item() ** 2
						grad = grad ** (1. / 2)
						grads.append(grad)

	np.savez(folder+'sig_hist_%s'%dataset[3:-4], loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h, grads = grads)
	# Save the model checkpoint
	# torch.save(model.state_dict(), 'model %s'%dataset[3:-4])
