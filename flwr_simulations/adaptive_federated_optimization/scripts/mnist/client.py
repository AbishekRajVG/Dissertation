"""Flower Client for CIFAR10/100."""


from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from .utils import ClientDataset, get_cifar_model, get_transforms, test, train

from logging import INFO, DEBUG
from flwr.common.logger import log

import random
import os

class RayClient(fl.client.NumPyClient):
    """Ray Virtual Client."""

    def __init__(self, cid: str, fed_dir: Path, num_classes: int):
        """Implements Ray Virtual Client.

        Args:
            cid (str): Client ID, in our case a str representation of an int.
            fed_dir (Path): Path where partitions are saved.
            num_classes (int): Number of classes in the classification problem.
        """
        self.cid = cid
        self.fed_dir = fed_dir
        self.num_classes = num_classes
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Returns properties for this client.

        Args:
            config (Dict[str, Scalar]): Options to be used for selecting specific
            properties.

        Returns:
            Dict[str, Scalar]: Returned properties.
        """
        # pylint: disable=unused-argument
        return self.properties

    def get_parameters(self, config) -> NDArrays:
        """Returns weight from a given model. If no model is passed, then a
        local model is created. This can be used to initialize a model in the
        server.

        Returns:
            NDArrays: weights from the model.
        """
        net = get_cifar_model(self.num_classes)
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Usual fit function that performs training locally.

        Args:
            parameters (NDArrays): Initial set of weights sent by the server.
            config (Dict[str, Scalar]): config file containing num_epochs,etc...

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: New set of weights,
            number of samples and dictionary of metrics.
        """
        
        if torch.cuda.is_available():
            # Set GPU fraction for this client [EXPERIMENTAL]
            tot_gpu_memory = (torch.cuda.get_device_properties(0).total_memory)/1000**2
            gpu_fraction = round(random.uniform(0.20, 0.50), 2)
            #gpu_fraction = round(random.uniform(0.30, 0.50), 2)
            torch.cuda.set_per_process_memory_fraction(gpu_fraction)
            gpu_percent = "{:.0%}".format(gpu_fraction)
            gpu_usage = "{} MiB".format(round(gpu_fraction*tot_gpu_memory,0))
            tot_gpu_memory = "{} MiB".format(round(tot_gpu_memory,0))

            #log(INFO, f"[Client {self.cid}] Total GPU Memeory: {tot_gpu_memory}")
            log(INFO, f"[Client {self.cid}] GPU Usage: {gpu_usage} ({gpu_percent} of {tot_gpu_memory})")
          
        # Set CPU affinity for this client
        #tot_cpu_count = torch.get_num_threads()
        tot_cpu_count = os.cpu_count()
        cpu_count = random.randint(1,tot_cpu_count)
        torch.set_num_threads(cpu_count)
        #log(INFO, f"[Client {self.cid}] Total CPU Count: {tot_cpu_count} Thread(s)")
        log(INFO, f"[Client {self.cid}] CPU Usage: {cpu_count}/{tot_cpu_count} Thread(s)")

        net = self.set_parameters(parameters)
        net.to(self.device)

        # train
        trainset = ClientDataset(
            path_to_data=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=get_transforms(self.num_classes)["train"],
        )
        trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]))
        train(net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights, len(trainset), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Implements distributed evaluation for a given client.

        Args:
            parameters (NDArrays): Set of weights being used for evaluation
            config (Dict[str, Scalar]): Dictionary containing possible options for
            evaluations.

        Returns:
            Tuple[float, int, Dict[str, float]]: Loss, number of samples and dictionary
            of metrics.
        """
        net = self.set_parameters(parameters)
        # load data for this client and get valloader
        validationset = ClientDataset(
            path_to_data=Path(self.fed_dir) / self.cid / "test.pt",
            transform=get_transforms(self.num_classes)["test"],
        )
        valloader = DataLoader(validationset, batch_size=50)

        # send model to device
        net.to(self.device)

        # evaluate
        loss, accuracy = test(net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def set_parameters(self, parameters: NDArrays):
        """Loads weights inside the network.

        Args:
            parameters (NDArrays): set of weights to be loaded.

        Returns:
            [type]: Network with new set of weights.
        """
        net = get_cifar_model(self.num_classes)
        weights = parameters
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net


def get_ray_client_fn(
    fed_dir: Path, num_classes: int = 10
) -> Callable[[str], RayClient]:
    """Function that loads a Ray (Virtual) Client.

    Args:
        fed_dir (Path): Path containing local datasets in the form ./client_id/train.pt
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Callable[[str], RayClient]: [description]
    """

    def client_fn(cid: str) -> RayClient:
        # create a single client instance
        return RayClient(cid=cid, fed_dir=fed_dir, num_classes=num_classes)

    return client_fn
