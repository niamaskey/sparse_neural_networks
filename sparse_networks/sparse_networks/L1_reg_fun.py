#University of Tasmania, School of Natural Sciences
#Niam Askey-Doran
#Honours 2021

#Functions used in training networks with L1 activity regularisation, and extra functions for convenience

# 1) Libraries
# 2) Importing data 
# 3) Testing and statistics
# 4) Training
# 5) Hooks
# 6) Plotting



#####################
######Libraries######
#####################

#Torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
#Math
import numpy as np
#Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from itertools import cycle

#Misc
from time import time

sns.set_style('ticks',
              {"font.family": "serif",
               'font.serif': ['computer modern roman'],
               }
               )
sns.set_context("paper", font_scale=2)

#Cuda
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
  
device = torch.device(dev)

##########################
######Importing Data######
##########################

def import_mnist(batch_size = 100, root = '../data', **kwargs):
    """
    Quickly import MNIST dataset, and store the training, testing, and validation
    datasets and dataloaders in dictionaries.
    - Returns: dataloaders_dict, datasets_dict
    """
    trans = transforms.Compose([transforms.ToTensor()])

    mnist_trainset = datasets.MNIST(root=root, train=True, download=True, transform=trans) 
    mnist_testset = datasets.MNIST(root=root, train=False, download=True, transform=trans)

    #Split trainset into training and validation sets
    mnist_trainset, mnist_valset = random_split(mnist_trainset, [50000, 10000], generator=torch.Generator().manual_seed(42))

    #Dict of dataset
    mnist_datasets = {'train': mnist_trainset, 'val': mnist_valset, 'test': mnist_testset}
    #Dict of dataloaders
    mnist_dataloaders = {x: DataLoader(mnist_datasets[x], batch_size = batch_size, shuffle = True, **kwargs) for x in ['train', 'val', 'test']}

    return mnist_dataloaders, mnist_datasets

def import_cifar(batch_size = 100, root = '../data', **kwargs):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar_trainset = datasets.CIFAR10(root = root, train=True, download=True, transform = transform)
    cifar_testset = datasets.CIFAR10(root = root, train=False, download=True, transform = transform)

    cifar_trainset, cifar_valset = random_split(cifar_trainset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    cifar_datasets = {'train': cifar_trainset, 'val': cifar_valset, 'test': cifar_testset}

    cifar_dataloaders = {x: DataLoader(cifar_datasets[x], batch_size = batch_size, shuffle = True, **kwargs) for x in ['train', 'val', 'test']}

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return cifar_dataloaders, cifar_datasets, classes

##############################       
######TESTING/STATISTICS######
##############################

def Sparsity(hook):
    """Calculate the average and standard deviation L0 and L1 sparsity of a network, given activations from a hook. 
    When test data is given to the network as a batch, activations are hooked as a tensor with rows = images, and columns = nodes.
    Sparsity will be calculated per image, then averaged over all images.
     

    Args:
        - hook (class): Forward hook registered to capture desired activation values.

    Returns:
        - L0_av (float): Average number of active nodes for a given input
        - L0_std (float):
        - L1_av (float):
        - L1_std (float):
    """
    No_layers = len(hook.outputs)
    L0_av = np.empty((No_layers))
    L0_std = np.empty((No_layers))
    L1_av = np.empty((No_layers))
    L1_std = np.empty((No_layers))
    
    for layer, activations in enumerate(hook.outputs): 
        no_images = hook.outputs[0].shape[0]    #Number of images - usually 10000
        activations = activations.reshape(no_images,-1)
        #Sparsity per image - norm calculated along dim=1 - i.e. across columns, as columns correspond to nodes in a layer. 
        #Gives sparsity per image.
        L0_act_per_im = torch.linalg.norm(activations, ord=0, dim=1).detach().numpy()
        L1_act_per_im = torch.linalg.norm(activations, ord=1, dim=1).detach().numpy()

        L0_av[layer] = np.mean(L0_act_per_im)
        L0_std[layer] = np.std(L0_act_per_im)
        L1_av[layer] = np.mean(L1_act_per_im)
        L1_std[layer] = np.std(L1_act_per_im)
        
    return L0_av, L0_std, L1_av, L1_std
        
        
def ConfusionMatrix(labels, predictions, no_classes = 10):
    """Compute the confusion matrix for a batch of predictions, given the target labels and predictions.
        Labels and predictions should be class labels, not one-hot coded (i.e. a tensor of integer labels)
    
    Args:
        - labels (Tensor): Target labels for test data
        - predictions (Tensor): Predictions of the modell
        - no_classes (int, optional): Number of classes in the dataset. Defaults to 10.

    Returns:
        - Tensor: Confusion matrix - Row = True label, Column = Prediction
    """
    confusion_matrix = torch.zeros(no_classes, no_classes)
    for label, pred in zip(labels, predictions):
        confusion_matrix[label.long(), pred.long()] += 1
    
    return confusion_matrix

def TestModel(network, images, labels):
    """Test a network on a test dataset and calculate the accuracy, recall, precision, and confusion matrix.
    Dataloader batch_size must be the size of the entire test dataset.
    
    Args:
        - network: Pytorch network
        - test_dataloader (DataLoader): Pytorch DataLoader containing the test data
        
    Returns:
        - Confusion matrix (Tensor): Confusion matrix of predictions - Row = True label, Column = Prediction
        - Accuracy (float): Percentage of predictions that are correct
        - Precision (Tensor): Proportion of predictions for each class that are correct.  = True positives / (True positives + False positives)
        - Recall (Tensor): Proportion of each class correctly identified. = True positives / (True positives + False negatives)
    """
    network.eval()

    #images, labels = iter(test_dataloader).next()
    #images = images.to(device)
    #labels = labels.to(device)
        
    output = network(images)

    _, predictions = torch.max(output, 1)  #Gives index of max in each row
    
    confusion_matrix = ConfusionMatrix(labels, predictions) #Row = Truth, col = prediction
    confusion_matrix.to(device)
    
    TP = torch.diag(confusion_matrix)   #Number of correct predictions - True positives
    actual_positives = torch.sum(confusion_matrix, dim=1) #Total number of images in each class (i.e.TP + FN)
    positive_predictions = torch.sum(confusion_matrix, dim=0)  #Total number of predictions made of each class (i.e. TP + FP)
    
    class_precision = torch.div(TP, positive_predictions)  #Proportion of predictions for each class that are correct
    class_recall = torch.div(TP, actual_positives)    #Proportion of each class correctly identified
    accuracy = torch.sum(torch.diag(confusion_matrix)) / len(images) * 100
    
    return confusion_matrix, accuracy, class_precision, class_recall

def MCC(confusion_matrix):
    """Calculate the Mathew's Correlation Coefficient (R_k statistic) for the network from the confusion matrix.

    Args:
        confusion_matrix (Tensor): K x K confusion matrix for K-class classification.

    Returns:
        Constant: Mathew's Correlation Coefficient
    """
    if confusion_matrix.type() != torch.Tensor:
        confusion_matrix = torch.Tensor(confusion_matrix)
    
    c = torch.sum(torch.diag(confusion_matrix))   #No. samples correclty predicted
    s = torch.sum(confusion_matrix.flatten())  #Total number of samples
    t = torch.Tensor([torch.sum(confusion_matrix[row, :]) for row in range(len(confusion_matrix[:,0]))])   #Number of times class k truly occurred
    p = torch.Tensor([torch.sum(confusion_matrix[:, col]) for col in range(len(confusion_matrix[0,:]))])   #Number of times class K was predicted
    
    MCC = (c * s - torch.dot(t, p)) / (np.sqrt(s ** 2 - torch.dot(p, p)) * np.sqrt(s ** 2 - torch.dot(t, t)))
    
    return float(MCC)

def GetWeights(model):
    
    """Returns a dictionary of the model weights.

    Args:
        model ([type]): Pytorch network

    Returns:
        dict: Dictionary of named model weights
    """
    weights = {}
    for name, par in model.named_parameters():
        if 'weight' in name:
            weights[name] = par.detach().numpy()
    
    return weights

def Dummy(model, input_size: tuple):
    x = torch.rand(input_size)
    y = model(x)
    return y

#############################
###### CREATING MODELS ######
#############################

def generate_model_FC1(no_hidden_units):
  """Generate a single layer network for MNIST with a given number of hidden layer units.
    Network has 784 inputs, 10 outputs.
  Args:
      - no_hidden_units (int): Number of units in the hidden layer

  Returns:
      - Class: Returns the network class
  """
  class FC1_sizetest(nn.Module):
    def __init__(self):
        super(FC1_sizetest, self).__init__()
        self.fc1 = nn.Linear(784, no_hidden_units)
        self.fc2 = nn.Linear(no_hidden_units, 10)    #output lay 
        #Instantiate Relu so hook can be registered
        #nn.Relu() is for modular defn, i.e. to be used in sequential model
        #nn.functional.relu is functional version - if using in forward method
        
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        #One fully connected layer
        x = self.relu(self.fc1(x))
        output = self.fc2(x)    #CLE loss fn has softmax build in 
        return output
    
  return FC1_sizetest()

def generate_model_FC2(no_hidden_units):
  """Generate a two layer network for MNIST with the same number of units in each layer.
    Network has 784 inputs, 10 outputs.
  Args:
      - no_hidden_units (int): Number of units in the hidden layer

  Returns:
      - Class: Returns the network class
  """
  class FC2_sizetest(nn.Module):
    def __init__(self):
        super(FC2_sizetest, self).__init__()
        self.fc1 = nn.Linear(784, no_hidden_units)
        self.fc2 = nn.Linear(no_hidden_units, no_hidden_units)
        self.out = nn.Linear(no_hidden_units, 10)    #output layer
        
        #Instantiate Relu so hook can be registered
        #nn.Relu() is for modular defn, i.e. to be used in sequential model
        #nn.functional.relu is functional version - if using in forward method
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.out(x)    #CLE loss fn has softmax build in, so leave outputs as raw
        return output
    
  return FC2_sizetest()

def generate_model_FC3(no_hidden_units):
  """Generate a three layer network for MNIST with the same number of units in each layer.
    Network has 784 inputs, 10 outputs.
  Args:
      - no_hidden_units (int): Number of units in the hidden layer

  Returns:
      - Class: Returns the network class
  """
  class FC3_sizetest(nn.Module):
    def __init__(self):
        super(FC3_sizetest, self).__init__()
        self.fc1 = nn.Linear(784, no_hidden_units)
        self.fc2 = nn.Linear(no_hidden_units, no_hidden_units)
        self.fc3 = nn.Linear(no_hidden_units, no_hidden_units)
        self.out = nn.Linear(no_hidden_units, 10)    #output layer
        
        #Instantiate Relu so hook can be registered
        #nn.Relu() is for modular defn, i.e. to be used in sequential model
        #nn.functional.relu is functional version - if using in forward method
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.out(x)    #CLE loss fn has softmax build in, so leave outputs as raw
        return output
    
  return FC3_sizetest()

################################
###### TRAINING FUNCTIONS ######
################################

def L1_train(model, dataloaders, loss_criterion, optimizer, hook, num_epochs = 5, L1_lambda = 0.0001, print_progress = True, norm_ord = 1, _num_registered_modules = None):
    """Trains a network with an L1 regularisation on the activations of nodes in hidden layers, and 
    tracks training and validation accuracy and loss.

    Args:
        - model: Network
        - dataloaders (dict): Dictionary of dataloaders of the form {'train': trainloader, 'val': validationloader}.
        - loss_criterion (Class): Loss function.
        - optimizer (Class): Gradient descent optimizer.
        - hook (Class): Forward hook to capture layer outputs (e.g. OutputHook() class).
        - num_epochs (int, optional): Number of training epochs. Defaults to 5.
        - L1_lambda (float, optional): L1 regularisation parameter. Defaults to 0.0001.
        - print_progress (bool, optional): Whether or not training progress should be printed each epoch. Defaults to True.

    Returns:
        - Training loss (List): Average loss over each epoch for training data.
        - Training accuracy (List): Percentage of correct predictions over the last epoch for training data.
        - Validation loss (List): Average loss over each epoch for the validation data.
        - Validation accuracy (List): Percentage of correct predictions for validation data after each epoch.
        - L1 Loss (List): Average L1 cost over each epoch for the training data 
    """
    since = time()
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    L1_loss = []

    batch_size = dataloaders['train'].batch_size    #batch size
    no_batches = {'train': len(dataloaders['train'].dataset)/batch_size, 'val': len(dataloaders['val'].dataset)/batch_size}   #Number of batches in each set

    #Begin epoch
    for epoch in range(num_epochs):
        #Set models to appropriate mode
        for phase in ['train', 'val']:  #Training and validation phases
            if phase == 'train':
                model.train()
            else:
                model.eval()

            correct = 0
            running_loss = 0
            
            for inputs, labels in dataloaders[phase]: #For each image batch in the dataset
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()   

                ### Training ###
                with torch.set_grad_enabled(phase == 'train'): #If phase is train, track gradients, otherwise do not.
                    ###Forward pass###
                    outputs = model(inputs)

                    #Calculate loss
                    loss = loss_criterion(outputs, labels)  #Mean Cross Entropy loss per input image
                    
                    #Double check that the correct number of activations have been hooked
                    if _num_registered_modules is not None:
                        _hook_len = len(hook.outputs)
                        if _hook_len != _num_registered_modules:
                            raise Exception(f'Number of modules registered does not match the number of modules actually hooked: {_num_registered_modules} modules registered, {_hook_len} modules hooked.')
                    
                    L1_penalty = 0
                    running_L1 = 0
                    #L1 penalty
                    for activations in hook.outputs:    #For activations in each layer
                        activity = torch.flatten(activations)  #Activations is a tensor - we want vector L1 norm, so flatten into vector first
                        L1_penalty = torch.linalg.norm(activity, 1) #L1 norm of activations
                        running_L1 += L1_penalty
                        
                    if L1_lambda != 0:
                        loss += (L1_lambda/batch_size) * running_L1    #Add average L1 penalty per image    
                    if phase == 'train':
                        L1_loss.append(running_L1.detach()/batch_size)   
                                            
                    ###Backward pass###
                    if phase == 'train':
                        loss.backward()     #Backpropagation
                        optimizer.step()    #Gradient step
                        
                    hook.clear()        #Clear L1 activity hooks, otherwise they will accumulate

                ###Statistics###                
                _, predictions = torch.max(outputs, norm_ord)  #Gives index of max in each row
                running_loss += loss.item() 
                correct += torch.sum(predictions == labels.data)
            #Epoch loss - loss averaged over entire epoch
            epoch_loss = running_loss / (no_batches[phase])  #Mean loss per image. Running loss is already Batch_no lots of mean loss per image, so divide by Batch_no to get mean loss per image
            epoch_accuracy = correct.double() / len(dataloaders[phase].dataset) #
            
            #Tracking L1_loss value. = 0 if lambda == 0
            #if L1_lambda != 0 and phase == 'train':
            #    L1_loss.append(L1_penalty.detach())
            #elif L1_lambda == 0 and phase == 'train':
            #    L1_loss = np.zeros(num_epochs) 
                
            #Save training and validation loss and accuracy
            if phase == 'train':
              train_loss_list.append(epoch_loss)
              train_accuracy_list.append(epoch_accuracy*100) 
            elif phase == 'val':
              val_loss_list.append(epoch_loss)
              val_accuracy_list.append(epoch_accuracy*100)
              
        
        #Print progess
        if print_progress==True:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, train_loss_list[-1],train_accuracy_list[-1], val_loss_list[-1], val_accuracy_list[-1]))
              
    time_elapsed = time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(max(val_accuracy_list)))
    
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, L1_loss

def L1_train_autoencoder(model, dataloaders, loss_criterion, optimizer, hook, num_epochs = 5, L1_lambda = 0.0001, print_progress = True, norm_ord = 1, _num_registered_modules = None):
    """Trains an autoencoder with an L1 regularisation on the activations of nodes in hidden layers, and 
    tracks training and validation mean square error (MSE) and loss.

    Args:
        - model: Network
        - dataloaders (dict): Dictionary of dataloaders of the form {'train': trainloader, 'val': validationloader}.
        - loss_criterion (Class): Loss function.
        - optimizer (Class): Gradient descent optimizer.
        - hook (Class): Forward hook to capture layer outputs (e.g. OutputHook() class).
        - num_epochs (int, optional): Number of training epochs. Defaults to 5.
        - L1_lambda (float, optional): L1 regularisation parameter. Defaults to 0.0001.
        - print_progress (bool, optional): Whether or not training progress should be printed each epoch. Defaults to True.

    Returns:
        - Training loss (List): Average loss over each epoch for training data.
        - Training MSE (List): Mean square error averaged over entire epoch for training data.
        - Validation loss (List): Average loss over each epoch for the validation data.
        - Validation MSE (List): Mean square error averaged over entire epoch for validation.
        - L1 Loss (List): L1 penalty per image, recorded every batch.
    """
    since = time()
    train_loss_list = []
    train_MSE_list = []
    val_loss_list = []
    val_MSE_list = []
    L1_loss = []

    batch_size = dataloaders['train'].batch_size    #batch size
    no_batches = {'train': len(dataloaders['train'].dataset)/batch_size, 'val': len(dataloaders['val'].dataset)/batch_size}   #Number of batches in each set

    #Begin epoch
    for epoch in range(num_epochs):
        #Set models to appropriate mode
        for phase in ['train', 'val']:  #Training and validation phases
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_MSE = 0
            running_loss = 0
            for inputs, labels in dataloaders[phase]: #For each image batch in the dataset
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()   

                ### Training ###
                with torch.set_grad_enabled(phase == 'train'): #If phase is train, track gradients, otherwise do not.
                    ###Forward pass###
                    outputs = model(inputs)

                    #Calculate loss
                    loss = loss_criterion(outputs, inputs.reshape(inputs.shape[0], -1))  #MSE loss per input image (i.e. mean over ever images in batch)

                    #Double check that the correct number of activations have been hooked
                    if _num_registered_modules is not None:
                        _hook_len = len(hook.outputs)
                        if _hook_len != _num_registered_modules:
                            raise Exception(f'Number of modules registered does not match the number of modules actually hooked: {_num_registered_modules} modules registered, {_hook_len} modules hooked.')
                    
                    #L1 penalty
                    L1_penalty = 0
                    for activations in hook.outputs:    #For activations in each layer
                        activity = torch.flatten(activations)  #Activations is a tensor - we want vector L1 norm, so flatten into vector first
                        L1_penalty = torch.linalg.norm(activity, norm_ord) #L1 norm of activations
                        if L1_lambda != 0:
                            loss += (L1_lambda/batch_size) * L1_penalty    #Add average L1 penalty per image
                        if phase == 'train':
                            L1_loss.append(L1_penalty.detach()/batch_size)   
                                           
                    ###Backward pass###
                    if phase == 'train':
                        loss.backward()     #Backpropagation
                        optimizer.step()    #Gradient step
                        
                    hook.clear()        #Clear L1 activity hooks, otherwise they will accumulate

                ###Statistics###                
                running_loss += loss.item() 
                running_MSE += nn.functional.mse_loss(outputs, inputs.reshape(inputs.shape[0], -1)).item()
            
            
            #Epoch loss - loss averaged over entire epoch
            epoch_loss = running_loss / (no_batches[phase])  #Mean loss per image. Running loss is already Batch_no lots of mean loss per image, so divide by Batch_no to get mean loss per image
            epoch_MSE = running_MSE / (no_batches[phase])    # Ditto
            
            #Save training and validation loss and accuracy
            if phase == 'train':
              train_loss_list.append(epoch_loss)
              train_MSE_list.append(epoch_MSE) 
            elif phase == 'val':
              val_loss_list.append(epoch_loss)
              val_MSE_list.append(epoch_MSE)
              
        
        #Print progess
        if print_progress==True:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train MSE: {:.2f}, Val Loss: {:.4f}, Val MSE: {:.2f}'.format(epoch + 1, num_epochs, train_loss_list[-1],train_MSE_list[-1], val_loss_list[-1], val_MSE_list[-1]))
              
    time_elapsed = time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MSE: {:4f}'.format(max(val_MSE_list)))
    
    return train_loss_list, train_MSE_list, val_loss_list, val_MSE_list, L1_loss

#def norm_L1_train(model, dataloaders, loss_criterion, optimizer, hook, num_epochs = 5, L1_lambda = 0.0001, print_progress = True):
#    """Trains a network with a NORMALISED L1 regularisation on the activations of nodes in hidden layers - i.e. L1 activation PER NODE, and 
#    tracks training and validation accuracy and loss.
#    - L1 penalty = lambda / (No_images * No_nodes) * sum(abs(activations))
#
#    Args:
#        - model: Network
#        - dataloaders (dict): Dictionary of dataloaders of the form {'train': trainloader, 'val': validationloader}.
#        - loss_criterion (Class): Loss function.
#        - optimizer (Class): Gradient descent optimizer.
#        - hook (Class): Forward hook to capture layer outputs (e.g. OutputHook() class).
#        - num_epochs (int, optional): Number of training epochs. Defaults to 5.
#        - L1_lambda (float, optional): L1 regularisation parameter. Defaults to 0.0001.
#        - print_progress (bool, optional): Whether or not training progress should be printed each epoch. Defaults to True.
#
#    Returns:
#        - Training loss (List): Average loss over each epoch for training data.
#        - Training accuracy (List): Percentage of correct predictions over the last epoch for training data.
#        - Validation loss (List): Average loss over each epoch for the validation data.
#        - Validation accuracy (List): Percentage of correct predictions for validation data after each epoch.
#        - L1 Loss (List): Average L1 cost over each epoch for the training data 
#    """
#    since = time()
#    train_loss_list = []
#    train_accuracy_list = []
#    val_loss_list = []
#    val_accuracy_list = []
#    L1_loss = []
#
#    batch_size = dataloaders['train'].batch_size    #batch size
#    no_batches = {'train': len(dataloaders['train'].dataset)/batch_size, 'val': len(dataloaders['val'].dataset)/batch_size}   #Number of batches in each set
#
#    #Begin epoch
#    for epoch in range(num_epochs):
#        #Set models to appropriate mode
#        for phase in ['train', 'val']:  #Training and validation phases
#            if phase == 'train':
#                model.train()
#            else:
#                model.eval()
#
#            correct = 0
#            running_loss = 0
#            
#            for inputs, labels in dataloaders[phase]: #For each image batch in the dataset
#                inputs = inputs.to(device)
#                labels = labels.to(device)
#
#                optimizer.zero_grad()   
#
#                ### Training ###
#                with torch.set_grad_enabled(phase == 'train'): #If phase is train, track gradients, otherwise do not.
#                    ###Forward pass###
#                    outputs = model(inputs)
#
#                    #Calculate loss
#                    loss = loss_criterion(outputs, labels)  #Mean Cross Entropy loss per input image
#                    L1_penalty = 0
#                    
#                    #L1 penalty
#                    if L1_lambda != 0:
#                        for activations in hook.outputs:    #For activations in each layer
#                            No_nodes = activations.shape[1]
#                            activity = torch.flatten(activations)  #Activations is a tensor - we want vector L1 norm, so flatten into vector first
#                            L1_penalty = torch.linalg.norm(activity, 1) #L1 norm of activations   
#                            loss += (L1_lambda/(batch_size * No_nodes)) * L1_penalty    #Add average L1 penalty per image, per node
#                                           
#                    ###Backward pass###
#                    if phase == 'train':
#                        loss.backward()     #Backpropagation
#                        optimizer.step()    #Gradient step
#                        
#                    hook.clear()        #Clear L1 activity hooks, otherwise they will accumulate
#
#                ###Statistics###                
#                _, predictions = torch.max(outputs, 1)  #Gives index of max in each row
#                running_loss += loss.item() 
#                correct += torch.sum(predictions == labels.data)
#            #Epoch loss
#            epoch_loss = running_loss/ (no_batches[phase])  #Mean loss per image. Running loss is already Batch_no lots of mean loss per image, so divide by Batch_no to get mean loss per image
#            epoch_accuracy = correct.double() / len(dataloaders[phase].dataset) #
#            
#            #Tracking L1_loss value. = 0 if lambda == 0
#            if L1_lambda != 0 and phase == 'train':
#                L1_loss.append(L1_penalty.detach())
#            elif L1_lambda == 0 and phase == 'train':
#                L1_loss = np.zeros(num_epochs) 
#                
#            #Save training and validation loss and accuracy
#            if phase == 'train':
#              train_loss_list.append(epoch_loss)
#              train_accuracy_list.append(epoch_accuracy*100) 
#            elif phase == 'val':
#              val_loss_list.append(epoch_loss)
#              val_accuracy_list.append(epoch_accuracy*100)
#        
#        #Print progess
#        if print_progress==True:
#            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, train_loss_list[-1],train_accuracy_list[-1], val_loss_list[-1], val_accuracy_list[-1]))
#              
#    time_elapsed = time() - since
#    
#    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(max(val_accuracy_list)))
#    
#    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, L1_loss

def VanillaTrain(model, dataloaders, criterion, optimizer, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        val_correct = 0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)  #Gives index of max in each row
            running_loss += loss.item() 
            correct += torch.sum(predictions == labels.data)
            running_loss += loss.item() / len(inputs)    
        
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  #Gives index of max in each row
            val_correct += torch.sum(predictions == labels.data) 
            
        # print statistics
        accuracy = correct.double() / len(dataloaders['train'].dataset)
        val_accuracy = val_correct.double() / len(dataloaders['val'].dataset)
        
        print('[%d, %5d] loss: %.3f, accuracy: %.3f, valiation accuracy: %.3f' %
              (epoch + 1, epochs, running_loss, accuracy, val_accuracy))
        running_loss = 0.0

    print('Finished Training')
#################
######HOOKS######
#################

class OutputHook(list):
    """
    Forward hook to capture module outputs. e.g. to get ReLU outputs, do
    hook = OutputHook()
    model.relu.register_forward_hook(hook) 
    - Activations are then found in OutputHook.outputs 
    """
    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        activations = torch.squeeze(output) #Squeeze to remove empty dimension that for some reason occurs, otherwise norm is messed up
        self.outputs.append(activations)
        
    def clear(self):
        self.outputs = []        
        
        
####################
######PLOTTING######
####################

def training_plot(train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list):
    """
    Plot training and validation accuracy and loss in a 2x2 grid.
    """
    data = [train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list]
    labels = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
    ylabel = ['Loss', '% Correct Predictions', 'Loss', '% Correct Predictions']
    
    plt.figure(figsize = (12, 10))
    i = 221
    for j in range(4):
        plt.subplot(i+j)
        plt.plot(range(len(data[j])), data[j])
        plt.title(labels[j])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel[j])
        

def plot_weights(network, layer, cmap = 'viridis', nrows = 8, ncols = 8, vmin = -1, vmax = 1):
    """
    Gets weights from a specific layer in the network and plots the weights 
    leading INTO each node in that layer in a grid - i.e. plots the "Receptive fields"
    of each node in a layer. Assumes receptive fields are square - i.e. a square number of nodes
    (or inputs) from the previous layer (4, 9, 16, 25, 36, ...)
    """
    weights = []
    rf_list = []
    
    #Get weights and biases from model
    for names, params in network.named_parameters():
        if 'weight' in names:
            weights.append(params)
            
    layer_weights = weights[layer]  #Select the weights in the desired layer
    
    #Assuming square receptive fields
    im_size = int(np.sqrt(layer_weights.shape[1])) #Image size determined by previous layer
    nnodes = int(len(layer_weights[:, 0]))      # number of nodes = number of weights in a column
    print(f'Images are size {im_size}x{im_size}, and there are {nnodes} nodes')
    
    for row in range(layer_weights.shape[0]):   #Grab receptive fields of each node
        rf = layer_weights[row, :]                      #Row = RF of node in next layer
        rf = torch.reshape(rf, (im_size, im_size))      #Reshape row into square
        rf_list.append(rf)                              #Store in a list
    #Plot
    fig = plt.figure(figsize = (10,10))
    grid = ImageGrid(fig, 111,
                 nrows_ncols = (nrows, ncols),
                 axes_pad = 0.01)

    for ax, im in zip(grid, rf_list):
        ax.imshow(im.detach().numpy(), str(cmap), vmin = vmin, vmax = vmax)
        #for spine in ax.spines:
         #   ax.spines[str(spine)].set_visible(False)
    
    
def weights_hist(network, bins = 30, title = "Histogram of Network Weights"):
    """
    Grabs all the weights from a fully conected network and plots a histogram of their
    distribution!
    """
    weights = []
    for names, params in network.named_parameters():
      if 'weight' in names:
        parameters = params.flatten().detach().numpy()
        weights += list(parameters)     #Did not know you could concat lists like this....
    
    plt.hist(weights, bins)
    plt.title(title)

def BreakAxes(ax1, ax2, f, ax2_ymin, ax2_ymax, ax1_ymin, ax1_ymax):

    ax1.set_ylim(ax1_ymin, ax1_ymax)
    ax2.set_ylim(ax2_ymin, ax2_ymax)
    ax2.yaxis.set_label_coords(-0.05,1.02)
    ax1.yaxis.set_label_coords(-0.05,1.02)
    # hide the spines between ax and ax2
    ax1.xaxis.set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    f.subplots_adjust(hspace=0.1)   
    
### Plotting for thesis ###
def Figure(FigNo = 0, figsize = (8,5)):
    return plt.figure(FigNo, figsize = figsize)

def Plot(x, y, label, ax = plt, ls = cycle(['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5))]), lw = 1.2, c = 'black'):
    ax.plot(x, y, label = label, ls = next(ls), lw = lw, c = c)
    sns.despine()
    
def Save(figure, figname, save = r'C:\Users\Niam\OneDrive - University of Tasmania\Documents\2021\Honours\Documents\Thesis\Take 2\Plots'):
    figure.savefig(fname = f'{save}\{figname}.eps', dpi = 300, bbox_inches = 'tight', transparent=True)
    figure.savefig(fname = f'{save}\{figname}.png', dpi = 300, bbox_inches = 'tight', transparent=True)
    

########################
### Experimental fun ###
########################

def L0_train(model, dataloaders, loss_criterion, optimizer, hook, num_epochs = 5, L1_lambda = 0.0001, print_progress = True):
    """Trains a network with an L1 regularisation on the activations of nodes in hidden layers, and 
    tracks training and validation accuracy and loss.

    Args:
        - model: Network
        - dataloaders (dict): Dictionary of dataloaders of the form {'train': trainloader, 'val': validationloader}.
        - loss_criterion (Class): Loss function.
        - optimizer (Class): Gradient descent optimizer.
        - hook (Class): Forward hook to capture layer outputs (e.g. OutputHook() class).
        - num_epochs (int, optional): Number of training epochs. Defaults to 5.
        - L1_lambda (float, optional): L1 regularisation parameter. Defaults to 0.0001.
        - print_progress (bool, optional): Whether or not training progress should be printed each epoch. Defaults to True.

    Returns:
        - Training loss (List): Average loss over each epoch for training data.
        - Training accuracy (List): Percentage of correct predictions over the last epoch for training data.
        - Validation loss (List): Average loss over each epoch for the validation data.
        - Validation accuracy (List): Percentage of correct predictions for validation data after each epoch.
        - L1 Loss (List): Average L1 cost over each epoch for the training data 
    """
    since = time()
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    L1_loss = []

    batch_size = dataloaders['train'].batch_size    #batch size
    no_batches = {'train': len(dataloaders['train'].dataset)/batch_size, 'val': len(dataloaders['val'].dataset)/batch_size}   #Number of batches in each set

    #Begin epoch
    for epoch in range(num_epochs):
        #Set models to appropriate mode
        for phase in ['train', 'val']:  #Training and validation phases
            if phase == 'train':
                model.train()
            else:
                model.eval()

            correct = 0
            running_loss = 0
            
            for inputs, labels in dataloaders[phase]: #For each image batch in the dataset
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()   

                ### Training ###
                with torch.set_grad_enabled(phase == 'train'): #If phase is train, track gradients, otherwise do not.
                    ###Forward pass###
                    outputs = model(inputs)

                    #Calculate loss
                    loss = loss_criterion(outputs, labels)  #Mean Cross Entropy loss per input image
                    L1_penalty = 0
                    
                    #L1 penalty
                    if L1_lambda != 0:
                        for activations in hook.outputs:    #For activations in each layer
                            activity = torch.flatten(activations)  #Activations is a tensor - we want vector L1 norm, so flatten into vector first
                            L1_penalty = torch.linalg.norm(activity, 0) #L1 norm of activations   
                            loss += (L1_lambda/batch_size) * L1_penalty    #Add average L1 penalty per image
                                           
                    ###Backward pass###
                    if phase == 'train':
                        loss.backward()     #Backpropagation
                        optimizer.step()    #Gradient step
                        
                    hook.clear()        #Clear L1 activity hooks, otherwise they will accumulate

                ###Statistics###                
                _, predictions = torch.max(outputs, 1)  #Gives index of max in each row
                running_loss += loss.item() 
                correct += torch.sum(predictions == labels.data)
            #Epoch loss
            epoch_loss = running_loss/ (no_batches[phase])  #Mean loss per image. Running loss is already Batch_no lots of mean loss per image, so divide by Batch_no to get mean loss per image
            epoch_accuracy = correct.double() / len(dataloaders[phase].dataset) #
            
            #Tracking L1_loss value. = 0 if lambda == 0
            if L1_lambda != 0 and phase == 'train':
                L1_loss.append(L1_penalty.detach())
            elif L1_lambda == 0 and phase == 'train':
                L1_loss = np.zeros(num_epochs) 
                
            #Save training and validation loss and accuracy
            if phase == 'train':
              train_loss_list.append(epoch_loss)
              train_accuracy_list.append(epoch_accuracy*100) 
            elif phase == 'val':
              val_loss_list.append(epoch_loss)
              val_accuracy_list.append(epoch_accuracy*100)
        
        #Print progess
        if print_progress==True:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, train_loss_list[-1],train_accuracy_list[-1], val_loss_list[-1], val_accuracy_list[-1]))
              
    time_elapsed = time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(max(val_accuracy_list)))
    
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, L1_loss

#%%
#########################
### SPARSITY MEASURES ###
#########################

def Gini(a):
    if type(a) == torch.Tensor:
        a.detach().numpy()
    coefficients = a.flatten()
    sorted_coefficients = np.sort(coefficients)
    N = len(sorted_coefficients)
    k = np.arange(1, N+1, 1)
    normed_coefficients = np.divide(sorted_coefficients, np.linalg.norm(sorted_coefficients, 1))
    weight = (N - k + 0.5)/N
    S = 1 - 2 * (np.sum(np.multiply(normed_coefficients, weight)))
    
    return S

def Hoyer(a):
    if type(a) == torch.Tensor:
        a.detach().numpy()
    coefficients = a.flatten()
    N = len(a)
    l1 = np.linalg.norm(coefficients, 1)
    l2 = np.linalg.norm(coefficients, 2)
    
    H = (np.sqrt(N) - (l1/l2)) / (np.sqrt(N) - 1)
    
    return H
    
# %%
