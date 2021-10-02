import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data_utils
from tqdm import tqdm
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 300
checks = 100
batch_size = 128
learning_rate = 1e-3

import model_def

model = model_def.model

input_size = 784
fc_units = 64
num_classes = 10

datasets = ['../mnist_corrupted.npz', '../fashion_mnist_corrupted.npz', '../cifar10_corrupted.npz']

for dataset in datasets:
	print('AlexNet on %s'%dataset[3:])
	data = np.load(dataset)

	if dataset == '../cifar10_corrupted.npz':
		import model_def
		model = model_def.model_cifar

		X_train, y_train, X_test, y_test = data['X_train_corrupted'], data['y_train_corrupted'], data['X_test'], data['y_test']
		y_train = np.array([np.where(y == 1)[0][0] for y in y_train])
		y_test  = np.array([np.where(y == 1)[0][0] for y in y_test])
		X_train, y_train, X_test, y_test = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test)

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
		for epoch in tqdm(range(num_epochs)):
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
					# 	   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
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

					np.savez('hist_%s'%dataset[3:-4], loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
					# Save the model checkpoint
					torch.save(model.state_dict(), 'model %s'%dataset[3:-4])
		continue

	X_train, y_train, X_test, y_test = data['X_train_corrupted'], data['y_train_corrupted'], data['X_test'], data['y_test']
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
				# 	   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
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

	np.savez('hist_%s'%dataset[3:-4], loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
	# Save the model checkpoint
	torch.save(model.state_dict(), 'model %s'%dataset[3:-4])

	