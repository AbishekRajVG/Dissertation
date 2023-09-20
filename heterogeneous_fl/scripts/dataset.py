#!/dcs/pg22/u2238887/.conda/envs/flwr/bin/python3.9

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader


def get_mnist(data_path: str = "../data"):
    
    #transformation for both test and train set are same for this case - possible overfitting eventually
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

#args
# - number of partiions (or the total clients in FL),
# - batch size as the function returns dataloaders and not just raw data - assuming all clients use the same batch size
# - validation ratio: the ratio of data allocated for validation 
def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, test_only: bool = False):

    #downloads our dataset
    trainset, testset = get_mnist()


    #only one, not list cause its going to test finally converged global model
    testloader = DataLoader(testset, batch_size=128)

    #if only testing required, return and end process, else continue to split train and val sets
    if(test_only==True):
        return testloader
    
    #split the dataset so we can assign a portion of it to every client in FL
    #partitioning data - splitting trainset into num_partition parts of trainsets - all partitions are going to be equal (iid)
    num_images = len(trainset) // num_partitions       #partition size of each clients partition
    partition_len = [num_images] * num_partitions      #length of each partition as a list of ints
    
    #creating IID Partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    #create dataloaders with train + val support
    trainloaders = []
    valloaders = []

    #one dataloader per client for each of both training and validation set
    for trainset_part in trainsets:
        num_total = len(trainset_part)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        #further split the trainset partition into training and validation sub-parts
        train_subpart, val_subpart = random_split(trainset_part, [num_train, num_val], torch.Generator().manual_seed(2023))
        
        #obtain dataloader to point to train and validation
        trainloaders.append(DataLoader(train_subpart, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(val_subpart, batch_size=batch_size, shuffle=False, num_workers=2))

    return trainloaders, valloaders, testloader
