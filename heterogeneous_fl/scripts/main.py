#!/dcs/pg22/u2238887/.conda/envs/flwr/bin/python3.9

import flwr as fl

import hydra
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf 

from dataset import prepare_dataset
from client import spawn_flower_client
from server import get_on_fit_config, get_evaluate_fn

from pathlib import Path
import pickle

@hydra.main(config_path="../conf", config_name="advance", version_base=None)
def main(cfg: DictConfig):


    ## step 1. parse config & get experiment output dir
    ## ------------------------------------------------
    print(OmegaConf.to_yaml(cfg))

    ## step 2. prep dataset
    ## --------------------
    # expected args
    # - number of partiions (or the total clients in FL),
    # - batch size as the function returns dataloaders and not just raw data - assuming all clients use the same batch size
    # - validation ratio: the ratio of data allocated for validation - default value 0.1 or 10%
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)


    ## step 3. define clients
    ## ----------------------
    flower_client = spawn_flower_client(trainloaders, validationloaders, cfg.model)
    
    
    ## step 4. define strategy
    ## -----------------------
    #using FedAvg strategy for now
    #args

    """    
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.0001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
 
    strategy = instantiate(cfg.strategy,
                           on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                           evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
    """

    # strategy = instantiate(cfg.strategy)
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evaluate_fn(cfg.model, testloader))
    
    ## step 5. Start Simulation
    ## ------------------------
    history = fl.simulation.start_simulation(
        client_fn=flower_client,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus':cfg.cpu_count_per_client, 'num_gpus':cfg.gpu_count_per_client}

    )
    ## How the SIM works: 
    # - Pass the function that creates clients (spawn_flower_client() from step 3)
    # - Pass Strategy
    # - Pass config = number of rounds in the FL
    # - Optional argument changing resources per client (cpu and gpu count per client) 


    ## step 6. save results
    ## --------------------
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {"history": history, 'anythingelse': "sample"}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":





    main()


