import Logic as bj
from torch import nn
import torch

namespace = ['hardcode', 'hc', 'imitation', 'imit']
n_inputs = 105
n_outputs = 3

def train_model(): 
    while True:
        print()
        model_name = input("Model Name: ")
        if model_name in namespace: 
            print("Please enter a name that is not already taken.")
            print("The following names are taken: ")
            for name in namespace:
                print("    " + name)
            print()
            continue
        else: 
            namespace.append(model_name)
            break

    while True: 
        try: 
            n_layers = int(input("Layers: "))
            break
        except: 
            print("Please enter an integer.")
            print()
            continue

    while True: 
        try: 
            n_p_layer = int(input("Neurons per Layer: "))
            break
        except: 
            print("Please enter an integer.")
            print()
            continue

    
    # print(f"Model: {model_name}")
    # print(f"Layers: {n_layers}")
    # print(f"Neurons per Layer: {n_p_layer}")
    print(f"Total Neurons: {n_layers*n_p_layer}")
    print()

    # The following prints out the model's structure but isn't really very pretty
    print("Model Structure: ")
    # build structure: list of layers, each layer is a list of neurons (represented here as '.')
    neur_struct = [['.' for _ in range(n_p_layer)] for _ in range(n_layers)]

    for layer in neur_struct:
        # a simple visual: show layer number and the neuron list
        print(' '.join(layer))
    print()


    print("Choose your training hyperparamters")
    learning_rate = float(input("Learning Rate: "))
    epochs = int(input("Epochs: "))

    # I need to fix my torch version to make sure this is working right
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Device: {device}")
    device = "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self, n_layers, n_p_layer):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential()
            
            layers = []
            for i in range(n_layers):
                if i == 0:
                    in_features=n_inputs
                    out_features = n_p_layer if n_layers > 1 else n_outputs
                elif i == n_layers - 1:
                    in_features=n_p_layer
                    out_features= n_outputs
                else: 
                    in_features=n_p_layer
                    out_features=n_p_layer

                layers.append(nn.Linear(in_features, out_features))

                if i != n_layers -1:
                    layers.append(nn.ReLU())

            self.linear_relu_stack = nn.Sequential(*layers)

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(n_layers, n_p_layer).to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    hsd_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_model()