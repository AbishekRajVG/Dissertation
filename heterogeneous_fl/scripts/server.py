#!/dcs/pg22/u2238887/.conda/envs/flwr/bin/python3.9


from collections import OrderedDict
from omegaconf import DictConfig 
from model import test

import torch
from hydra.utils import instantiate



def get_on_fit_config(config: DictConfig):

    def fit_config_fn(server_round: int):

        '''
        #since at this point we know which round we're at, we could
        #modify config based on rounds spent
        if server_round>50:
            lr = config.lr / 10
        '''

        return {'lr':config.lr, 
                'momentum':config.momentum,
                'local_epochs':config.local_epochs}
    
    return fit_config_fn

def get_evaluate_fn(model_cfg, testloader):

    def evaluate_fn(server_round: int, parameters, config):

        model = instantiate(model_cfg)
        # model = Net(num_classes)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.Tensor(value) for key,value in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        
        return loss, {"accuracy": accuracy}


    return evaluate_fn
