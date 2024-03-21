def train_L1_activity_reg(model, dataloaders, loss_criterion, optimizer, hook, num_epochs = 5):
    """
    Trains models with L1 regularization on the activations, and tracks taining and validation progress. 
    - dataloaders should be a dict of training and validation loaders of the form {'train': trainloader, 'val': valloader}
    - Hook is an instantiation of the OutputHook class
    """
    since = time.time()
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    hook_outputs = []

    #Begin epoch
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            correct = 0
            running_loss = 0
            #store train and val loaders in dict with keys 'train' and 'val'
            for inputs, labels in dataloaders[phase]: 
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                ###Forward pass###
                #If phase is train, track gradients, otherwise do not.
                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs)

                    #Calculate loss
                    loss = loss_criterion(outputs, labels)
                    L1_penalty = 0
                    for activations in hook.outputs:
                        L1_penalty += torch.norm(activations, 1) #L1 norm of activations
                        loss += L1_lambda*L1_penalty
                        if phase == 'val':
                            hook_outputs.append(activations)

                    ###Backward pass###
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        hook.clear()    

                ###Statistics###
                dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
                
                _, predictions = torch.max(outputs, 1)  #Gives index of max in each row
                running_loss += loss.item()
                correct += torch.sum(predictions == labels.data)
                
            epoch_loss = running_loss/ dataset_sizes[phase] #Gives loss per input for epoch
            epoch_accuracy = correct.double() / dataset_sizes[phase]
            
            #Save training and validation loss and accuracy
            if phase == 'train':
              train_loss_list.append(epoch_loss)
              train_accuracy_list.append(epoch_accuracy*100) 
            elif phase == 'val':
              val_loss_list.append(epoch_loss)
              val_accuracy_list.append(epoch_accuracy*100)
        
        #Print progess
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, train_loss_list[-1],train_accuracy_list[-1], val_loss_list[-1], val_accuracy_list[-1]))
              
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(max(val_accuracy_list)))
    print(L1_lambda)
    
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, hook_outputs