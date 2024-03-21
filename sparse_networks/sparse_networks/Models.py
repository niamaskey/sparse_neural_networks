import torch
import torch.nn as nn
import math
import torch.nn.functional as F

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

############

#############################################
###### CREATING FULLY CONNECTED MODELS ######
#############################################

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

#############################################
###### FULLY CONNECTED SIGMOID MODELS ######
#############################################

def generate_model_FC1_sig(no_hidden_units):
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
        
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        #One fully connected layer
        x = self.sig(self.fc1(x))
        output = self.fc2(x)    #CLE loss fn has softmax build in 
        return output
    
  return FC1_sizetest()

def generate_model_FC2_sig(no_hidden_units):
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
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        output = self.out(x)    #CLE loss fn has softmax build in, so leave outputs as raw
        return output
    
  return FC2_sizetest()

def generate_model_FC3_sig(no_hidden_units):
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
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        x = input.reshape(input.shape[0], -1)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))
        output = self.out(x)    #CLE loss fn has softmax build in, so leave outputs as raw
        return output
    
  return FC3_sizetest()

##############################
### CONVOLUTIONAL NETWORKS ###
##############################

def OutSize(input_size, kernel_size, padding, stride):
    
    return math.floor( (input_size + 2 * padding - kernel_size) / (stride) + 1)

def GenerateConv1(input_size: tuple, kernel_size = 3):
  """[summary]

  Returns:
      [type]: [description]
  """
  class Conv1(nn.Module):
    def __init__(self, input_size, kernel_size):
      super(Conv1, self).__init__()
        
      self._input_size = input_size
      self.kernel_size = kernel_size
      self._input_channels = input_size[0]
      self.input_len = input_size[1]
      self.C1_outsize = OutSize(self.input_len, self.kernel_size, 0, 1)
        
      self.C1 = nn.Sequential(
          nn.Conv2d(self._input_channels, 6, kernel_size=self.kernel_size, stride=1, padding=0),
          nn.ReLU()
      ) 
      self.FC1 = nn.Sequential(
          nn.Linear(6 * self.C1_outsize ** 2, 64),
          nn.ReLU()
      )
      self.FC2 = nn.Sequential(
          nn.Linear(64, 10)
      )

      #self.relu = nn.ReLU()

    def forward(self, x):
        out = self.C1(x)
        out = out.reshape(out.shape[0], -1)
        out = self.FC1(out)
        out = self.FC2(out)
        return out
      
  return Conv1(input_size, kernel_size)
  
def GenerateConv2(input_size: tuple, kernel_size = 3):
  """[summary]

  Returns:
      [type]: [description]
  """
  class Conv1(nn.Module):
    def __init__(self, input_size, kernel_size):
      super(Conv1, self).__init__()
        
      self._input_size = input_size
      self.kernel_size = kernel_size
      self._input_channels = input_size[0]
      self.input_len = input_size[1]
      self.C1_outsize = OutSize(self.input_len, self.kernel_size, 0, 1)
      self.C2_outsize = OutSize(self.C1_outsize, self.kernel_size, 0, 1)
        
      self.C1 = nn.Sequential(
          nn.Conv2d(self._input_channels, 6, kernel_size=self.kernel_size, stride=1, padding=0),
          nn.ReLU()
      ) 
      
      self.C2 = nn.Sequential(
        nn.Conv2d(6, 16, kernel_size=self.kernel_size, stride=1, padding=0),
        nn.ReLU()
      )
      
      self.FC1 = nn.Sequential(
          nn.Linear(16 * self.C2_outsize ** 2, 64),
          nn.ReLU()
      )
      self.FC2 = nn.Sequential(
          nn.Linear(64, 10)
      )

      #self.relu = nn.ReLU()

    def forward(self, x):
        out = self.C1(x)
        out = self.C2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.FC1(out)
        out = self.FC2(out)
        return out
      
  return Conv1(input_size, kernel_size)
 
  
def GenerateModelConv1_sig(input_size: tuple):
    """[summary]

    Returns:
        [type]: [description]
    """
    class Conv1(nn.Module):
      def __init__(self):
        super(Conv1, self).__init__()
        
        self._input_size = input_size
        self._input_channels = input_size[0]
        self.input_len = input_size[1]
        self.C1_outsize = OutSize(self.input_len, 3, 0, 1)
        
        self.C1 = nn.Sequential(
            nn.Conv2d(self._input_channels, 4, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        ) 
        self.FC1 = nn.Sequential(
            nn.Linear(4 * self.C1_outsize ** 2, 64),
            nn.Sigmoid()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.relu = nn.ReLU()

      def forward(self, x):
          out = self.C1(x)
          out = out.reshape(out.shape[0], -1)
          out = self.FC1(out)
          out = self.FC2(out)
          return out
      
    return Conv1()
  
####################
### Autoencoders ###
####################
def AutoEnc_1(input_len = 784, code_size = 1000):
  class AutoEncoder(nn.Module):
    def __init__(self, input_len, code_size):
      super().__init__()
      self.encoder_in = nn.Linear(input_len, code_size)
      self.decoder_out = nn.Linear(code_size, input_len)

      self.relu = nn.ReLU()
        
    def forward(self, input):
      x = input.reshape(input.shape[0], -1)
      x = self.encoder_in(x)
      x = self.relu(x)
          
      out = self.decoder_out(x)
          
      return out
  
  return AutoEncoder(input_len, code_size)


def AutoEnc_2(layer_sizes = (784, 400, 64, 400, 784), **kwargs):
    class AutoEncoder(nn.Module):
        def __init__(self, layer_sizes):
            super().__init__()

            self.input_len = layer_sizes[0]
            self.e1_size = layer_sizes[1]
            self.code_size = layer_sizes[2]
            self.d1_size = layer_sizes[3]

            self.encoder_h1 = nn.Sequential(
                nn.Linear(self.input_len, self.e1_size),
                nn.ReLU()
                )
            self.code = nn.Sequential(
                nn.Linear(self.e1_size, self.code_size),
                nn.ReLU()
            )
            self.decoder_h1 = nn.Sequential(
                nn.Linear(self.code_size, self.d1_size),
                nn.ReLU()
                )
            self.decoder_out = nn.Sequential(
                nn.Linear(self.d1_size, self.input_len),
                nn.ReLU()
            )
            
            self.relu = nn.ReLU()
            
        def forward(self, input):
            x = input.reshape(input.shape[0], -1)
            x = self.encoder_h1(x)
            
            code = self.code(x)
            
            x2 = self.decoder_h1(code)
            
            reconstruction = self.decoder_out(x2)
            
            return reconstruction

    return AutoEncoder(layer_sizes, **kwargs)


def AutoEnc(**kwargs):
    class AutoEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder_h1 = nn.Linear(kwargs["input_shape"], 900)
            self.encoder_h2 = nn.Linear(900, 1000)
            self.decoder_h1 = nn.Linear(1000, 900)
            self.decoder_out = nn.Linear(900, kwargs["input_shape"])
            
            self.relu = nn.ReLU()
            
        def forward(self, input):
            x = input.reshape(input.shape[0], -1)
            x = self.encoder_h1(x)
            x = torch.relu(x)
            
            code = self.encoder_h2(x)
            code = self.relu(code)  #Self relu for sparse coding
            
            x2 = self.decoder_h1(code)
            x2 = torch.relu(x2)
            
            x2 = self.decoder_out(x2)
            reconstruction = torch.relu(x2)
            
            return reconstruction

    return AutoEncoder(**kwargs)
