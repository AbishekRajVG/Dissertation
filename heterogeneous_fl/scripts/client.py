from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl

from collections import OrderedDict
from model import train, test

from hydra.utils import instantiate

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 model_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #copy params from global (server) to local client
    def set_parameters(self, parameters):

        #note- parameters has to be ordered, taken care from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)

        #params dict is a numpy array, we're converting to pytorch tensor representation
        state_dict = OrderedDict({key: torch.Tensor(value) for key,value in params_dict})

        #update the local model with the new state_dict
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):

        #extract every element of state_dict of local model, move it to cpu if not already, convert to numpy and return complete numpy array
        return [value.cpu().numpy() for key, value in self.model.state_dict().items()]


    #each client (object of FlowerClient) will have default model weights from above, 
    #but will be overwritten by the weights recieved as parameters from the global model
    #args
    #1. parameters - list of numpy arrays - represents to current state of the global model
    #2. config - python dict - additional info (hyperparams) sent by the server to consider in this particular round of training
    def fit(self, parameters, config):

        #copy params sent by server to the client's local model
        self.set_parameters(parameters)

        #extract any relevant hyperparams for training from config (sent by server)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        #setup optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        #perform local training
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        #finally update send back local model updates to the server
        #for this we send the updated params, 
        #send lenght of trainloader - fedAvg aggregation method requires to know how many training examples per client
        #send any stats/metrics - for now empty {}

        return self.get_parameters({}), len(self.trainloader), {}


    #evaluate the global model on the validation set of the client
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):

        #copy params sent by server to the client's local model
        self.set_parameters(parameters)

        #perform testing on the global model and report back loss and accuracy to the server
        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}
    

   
## master Function to spawn flower clients
# returns a functiuon that in-turn instantiates a client with client-id
def spawn_flower_client(trainloaders, valloaders, model_cfg):

    def flower_client(cid: str):

        return FlowerClient(trainloader=trainloaders[int(cid)], 
                            valloader=valloaders[int(cid)],
                            model_cfg=model_cfg)

    return flower_client
