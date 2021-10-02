import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data_utils
from tqdm import tqdm


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        torch.nn.init.xavier_uniform(m.weight.data)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_points = 1

num_epochs = 50
num_epochs_u = 50
num_epochs_cifar = 5*num_epochs
num_epochs_cifar_u = 5*num_epochs
checks = 100
batch_size = 128
learning_rate = 1e-2
for point in range(N_points):
    # num_epochs = 20
    # num_epochs_cifar = 4*num_epochs

    # Wsave = model.get_weights()

    folder = '/raid/data/matlab/sad/MLP/'
    datasets = ['../cifar10_corrupted.npz']

    # for dataset in datasets:
    #     print('%d MLP on %s, usual'%(point, dataset[3:]))
    #     import model_def
    #     model = model_def.model
    #     model.apply(weight_init)
    #     data = np.load(dataset)
    #     if dataset == '../cifar10_corrupted.npz':
    #         import model_def
    #         model = model_def.model_cifar
    #         model.apply(weight_init)
    #         # model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
    #         # model.train()

    #         X_train, y_train, X_test, y_test = data['X_train_corrupted'][:50000], data['y_train_corrupted'][:50000], data['X_test'], data['y_test']
    #         y_train = np.array([np.where(y == 1)[0][0] for y in y_train])
    #         y_test  = np.array([np.where(y == 1)[0][0] for y in y_test])
    #         X_train, y_train, X_test, y_test = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test)

    #         train = data_utils.TensorDataset(X_train, y_train)
    #         train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    #         test = data_utils.TensorDataset(X_test, y_test)
    #         test_loader = data_utils.DataLoader(test, batch_size=len(y_test), shuffle=True)

    #         criterion = nn.CrossEntropyLoss()
    #         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    #         # Train the model
    #         total_step = len(train_loader)

    #         loss_h = []
    #         loss_val_h = []
    #         acc_h = []
    #         acc_val_h = []
    #         for epoch in tqdm(range(num_epochs_cifar_u)):
    #             for i, (images, labels) in enumerate(train_loader):  
    #                 # Move tensors to the configured device
    #                 images = images.reshape(-1,32*32*3).to(device).float()
    #                 labels = labels.to(device).long()
                    
    #                 # Forward pass
    #                 outputs = model(images)
    #                 loss = criterion(outputs, labels)
                    
    #                 # Backward and optimize
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
                    
    #                 if (i+1) % checks == 0:
    #                     # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                     #        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    #                     total, correct = 0, 0
    #                     _, predicted = torch.max(outputs.data, 1)
    #                     total += labels.size(0)
    #                     correct += (predicted == labels).sum().item()

    #                     loss_h.append(loss.item())
    #                     acc_h.append(correct/total)

    #                     with torch.no_grad():
    #                         correct = 0
    #                         total = 0
    #                         loss_v = 0
    #                         for images, labels in test_loader:
    #                             images = images.reshape(-1, 32*32*3).to(device).float()
    #                             labels = labels.to(device).long()
    #                             outputs = model(images)
    #                             loss_v = criterion(outputs, labels)

    #                             _, predicted = torch.max(outputs.data, 1)
    #                             total += labels.size(0)
    #                             correct += (predicted == labels).sum().item()
    #                         loss_val_h.append(loss_v.item())
    #                         acc_val_h.append(correct/total)
                        
    #         np.savez(folder+'u_hist_%s_%d'%(dataset[3:-4], point), loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
    #         # Save the model checkpoint
    #         torch.save(model.state_dict(), folder+'u%s_%d'%(dataset[3:-4], point))
    #         continue
    #     # model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
    #     # model.train()
    #     X_train, y_train, X_test, y_test = data['X_train_corrupted'][:60000], data['y_train_corrupted'][:60000], data['X_test'], data['y_test']
    #     X_train, y_train, X_test, y_test = torch.from_numpy(X_train/255), torch.from_numpy(y_train), torch.from_numpy(X_test/255), torch.from_numpy(y_test)

    #     train = data_utils.TensorDataset(X_train, y_train)
    #     train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    #     test = data_utils.TensorDataset(X_test, y_test)
    #     test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    #     # Train the model
    #     total_step = len(train_loader)
    #     loss_h = []
    #     loss_val_h = []
    #     acc_h = []
    #     acc_val_h = []
    #     for epoch in tqdm(range(num_epochs_u)):
    #         for i, (images, labels) in enumerate(train_loader):  
    #             # Move tensors to the configured device
    #             images = images.reshape(-1, 28*28).to(device).float()
    #             labels = labels.to(device).long()
                
    #             # Forward pass
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
                
    #             # Backward and optimize
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             if (i+1) % checks == 0:
    #                 # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                 #        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    #                 total, correct = 0, 0
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum().item()

    #                 loss_h.append(loss.item())
    #                 acc_h.append(correct/total)

    #                 with torch.no_grad():
    #                         correct = 0
    #                         total = 0
    #                         loss_v = 0
    #                         for images, labels in test_loader:
    #                             images = images.reshape(-1, 28*28).to(device).float()
    #                             labels = labels.to(device).long()
    #                             outputs = model(images)
    #                             loss_v = criterion(outputs, labels)

    #                             _, predicted = torch.max(outputs.data, 1)
    #                             total += labels.size(0)
    #                             correct += (predicted == labels).sum().item()
    #                         loss_val_h.append(loss_v.item())
    #                         acc_val_h.append(correct/total)

    #     np.savez(folder+'u_hist_%s_%d'%(dataset[3:-4], point), loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
    #     # Save the model checkpoint
    #     torch.save(model.state_dict(), folder+'u%s_%d'%(dataset[3:-4], point))    

    for dataset in datasets:
        print('%d MLP on %s, sad'%(point, dataset[3:]))
        import model_def
        model = model_def.model
        model.apply(weight_init)
        data = np.load(dataset)
        if dataset == '../cifar10_corrupted.npz':
            import model_def
            model = model_def.model_cifar
            model.apply(weight_init)
            # model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
            # model.train()

            X_train, y_train, X_test, y_test = data['X_train_corrupted'], data['y_train_corrupted'], data['X_test'], data['y_test']
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
            for epoch in tqdm(range(num_epochs_cifar)):
                for i, (images, labels) in enumerate(train_loader):  
                    # Move tensors to the configured device
                    images = images.reshape(-1,32*32*3).to(device).float()
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
                                images = images.reshape(-1, 32*32*3).to(device).float()
                                labels = labels.to(device).long()
                                outputs = model(images)
                                loss_v = criterion(outputs, labels)

                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                            loss_val_h.append(loss_v.item())
                            acc_val_h.append(correct/total)

            np.savez(folder+'s_hist_%s_%d'%(dataset[3:-4], point), loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
            # Save the model checkpoint
            torch.save(model.state_dict(), folder+'s%s_%d'%(dataset[3:-4], point))
            continue
        # model.load_state_dict(torch.load(folder+'model %s'%dataset[3:-4]))
        # model.train()
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
                images = images.reshape(-1, 28*28).to(device).float()
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
                                images = images.reshape(-1, 28*28).to(device).float()
                                labels = labels.to(device).long()
                                outputs = model(images)
                                loss_v = criterion(outputs, labels)

                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                            loss_val_h.append(loss_v.item())
                            acc_val_h.append(correct/total)

        np.savez(folder+'s_hist_%s_%d'%(dataset[3:-4], point), loss_h = loss_h, loss_val_h = loss_val_h, acc_h = acc_h, acc_val_h = acc_val_h)
        # Save the model checkpoint
        torch.save(model.state_dict(), folder+'s%s_%d'%(dataset[3:-4], point))