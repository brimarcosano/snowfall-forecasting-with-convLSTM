import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time
from tempfile import TemporaryDirectory
import shutil


def train(device, model, criterion, optimizer, num_epochs, train_loader):

    best_epoch_acc = 0.0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
                # Backward and optimize
                loss.backward()
                optimizer.step()
                # del images, labels, outputs
                # torch.cuda.empty_cache()
                # gc.collect()
            
            # train_ds_size = {x: len(train_loader) for x in train_loader}
            train_ds_size = len(train_loader)
            inputs, classes = next(iter(train_loader))

            running_loss = 0.0
            running_corrects = 0
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # scheduler.step()

        epoch_loss = running_loss / train_ds_size
        epoch_acc = running_corrects.double() /train_ds_size

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            print(f'New best epoch acc: {best_epoch_acc}')


    return best_epoch_acc

# print(('Epoch [{}/{}], Loss: {:.4f}' 
#     .format(epoch+1, num_epochs, loss.item()))
    # )
# return('Epoch [{}/{}], Loss: {:.4f}' 
#     .format(epoch+1, num_epochs, loss.item()))

'''           
    START DOING THIS STUFF 
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''
                
# Validation
def validation_or_test(model, device, epoch_acc, criterion, loader):

    best_acc = 0.0

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        with torch.no_grad():
            # correct = 0
            # total = 0
            running_loss = 0.0
            running_corrects = 0

            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(predicted == labels.data)
                # del images, labels, outputs
            
            ds_size = len(loader)
            inputs, classes = next(iter(loader))

            epoch_loss = running_loss / ds_size
            epoch_acc = running_corrects.double() / ds_size

            print(f'Validation or test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
        # Move the file to a permanent location
        # shutil.move(best_model_params_path, '/path/to/permanent/location/best_model_params.pt')

        
        model.load_state_dict(torch.load(best_model_params_path))
    
    print()

    return model

    # return('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))








def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model