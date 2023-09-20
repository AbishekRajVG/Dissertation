import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf



class SampleClass:
    def __init__(self, x):
        self.x = x

    def print_x_sq(self):
        print(f"{self.x**2 = }")

class AnotherClass:
    def __init__(self, nest_obj: SampleClass):
        self.obj = nest_obj
    
    def instantiate_parent_object(self):
        self.obj = instantiate(self.obj)

def add_function(x,y):
    sum = x + y
    #print(f"{sum = }")
    return sum


@hydra.main(config_path="../conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):

    ## display yaml as config
    print(OmegaConf.to_yaml(cfg))


    ## basic
    print(cfg.foo)
    print(cfg.bar.abar)
    print(cfg.bar.bbar)
    print(cfg.bar.bbar.bbar2)


    ## simple functions
    result = call(cfg.sum_function)

    ## simple functions - compile time modified args
    result2 = call(cfg.sum_function, y=1000)

    print(f"{result = }")
    print(f"{result2 = }")

    #partial function has only partial args, rest have to be provided. 
    #could be useful where let's say x is obtained from intermediate steps
    partial_function_call = call(cfg.partial_sum_function)
    result3 = partial_function_call(x=3000)
    print(f"{result3 = }")


    print("----"*10)
    print("Objects")

    object = instantiate(cfg.sample_object)
    object.print_x_sq()

    print("----"*10)
    print("Nested Objects")
    
    nest_object = instantiate(cfg.nested_object)

    print("before instantiating parent")
    print(nest_object.obj)

    print("after instantiating parent object")
    print(nest_object.obj)


    nest_object.instantiate_parent_object()
    nest_object.obj.print_x_sq()


    print("---"*30)
    print("models")

    mymodel = instantiate(cfg.model)
    print(mymodel)
    








if __name__ == "__main__":

    main()