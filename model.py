import torch.nn as nn
import torch
class MyNN(nn.Module):
  def __init__(self,no_convs,input_features,conv_channels,kernel_sizes,pool_sizes,num_hidden_layers,n_layer,dropout_rate,out_dim):
    super().__init__()


    conv_layers=[]
    input_feature=input_features
    for i in range(no_convs):
      output_feature = conv_channels[i]
      kernel_size = kernel_sizes[i]
      pool_size = pool_sizes[i]
      conv_layers.extend([
      nn.Conv2d(input_feature,output_feature,kernel_size,stride=1,padding='same'),       
      nn.ReLU(),
      nn.BatchNorm2d(output_feature),
      nn.MaxPool2d(pool_size,stride=2),
        ])
      input_feature = output_feature

        
    self.features=nn.Sequential(*conv_layers)
    with torch.no_grad():
        dummy = torch.zeros(1, input_features, 28, 28)
        out = self.features(dummy)
        flatten_dim = out.view(1, -1).size(1)



      

    input_dim=flatten_dim
    layer=[]
    for i in range(num_hidden_layers):
      layer.append(nn.Linear(input_dim,n_layer))
      layer.append(nn.BatchNorm1d(n_layer))
      layer.append(nn.ReLU())
      layer.append(nn.Dropout(dropout_rate))
      input_dim=n_layer
    layer.append(nn.Linear(n_layer,out_dim))    
    self.classifier=nn.Sequential(*layer)





  def forward(self,x):
    x = x.view(-1, 1, 28, 28) # Reshape the input
    x=self.features(x)
    x = torch.flatten(x, 1) 
    x=self.classifier(x)

    return x