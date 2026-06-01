from typing import Any
import torch
import torch.nn as nn
import ignet_ai_train.nn.modules as ignet_nn

_registered_models:dict[str,Any] = dict()
def _init_registered_models():
    global _registered_models
    from_packages = [nn,ignet_nn]

    for from_ in from_packages:
        for name in dir(from_):
            if name.startswith("_"):
                continue
            obj = getattr(from_, name)
            if isinstance(obj, type):
                _registered_models[name] = obj
    
    def create_parameter(
        shape: tuple[int, ...] | list[int],
        requires_grad: bool = False,
        default: str | float | torch.Tensor = "random",
        std: float = 1.0,
        mean: float = 0.0,
        dtype: str | torch.dtype = torch.float32,
    ) -> nn.Parameter:
        """
        Create a parameter with given shape.
        Args:
            shape (tuple[int,...]|list[int]): The shape of the parameter.
            requires_grad (bool): Whether the parameter requires gradient.
            default (str|float|torch.Tensor): The default value of the parameter.  Can be "random", "zero", "one", "identity", or a scalar/Tensor.
            std (float): The standard deviation of the parameter. Only used when default is "random".
            mean (float): The mean of the parameter. Only used when default is "random".
            dtype (str|torch.dtype): The dtype of the parameter. Default: torch.float32
        Returns:
            nn.Parameter: The parameter.
        """
        shape = tuple(shape)
        
        if isinstance(default, str):
            if default == "random":
                default_tensor = torch.randn(shape) * std + mean
            elif default == "zero":
                default_tensor = torch.zeros(shape)
            elif default == "one":
                default_tensor = torch.ones(shape)
            elif default == "identity":
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError(f"Identity matrix requires a square 2D shape, got {shape}")
                default_tensor = torch.eye(shape[0])
            else:
                raise ValueError(f"Unknown default string: {default}")
        elif isinstance(default, torch.Tensor):
            if default.shape != shape:
                raise ValueError(f"Provided tensor shape {default.shape} does not match required shape {shape}")
            default_tensor = default
        else:
            # scalar: int or float
            default_tensor = torch.full(shape, default)

        default_tensor = default_tensor.to(dtype)
        return nn.Parameter(default_tensor, requires_grad=requires_grad)

    _registered_models["Parameter"] = create_parameter

    try:
        import torchvision.models as models
        for name in dir(models):
            if name.startswith("_"):
                continue
            if name.endswith("_Weights"):
                continue
            obj = getattr(models, name)
            if isinstance(obj, type):
                _registered_models[name] = obj
        def create_resnet(id,nc,pretrained=False):
            """ResNet from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
            Args:
                id (str): The id of the model. e.g. 18, 34, 50, 101, 152
                nc (int): The number of classes.
                pretrained (bool): Whether to use pretrained weights.
            """
            if id == 18:
                return models.resnet18(num_classes=nc,pretrained=pretrained)
            elif id == 34:
                return models.resnet34(num_classes=nc,pretrained=pretrained)
            elif id == 50:
                return models.resnet50(num_classes=nc,pretrained=pretrained)
            elif id == 101:
                return models.resnet101(num_classes=nc,pretrained=pretrained)
            elif id == 152:
                return models.resnet152(num_classes=nc,pretrained=pretrained)
            else:
                raise Exception(f"Unknown resnet id: {id}")
        _registered_models["resnet"] = create_resnet

        def create_vgg(id,nc,pretrained=False):
            """
            VGG from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

            Args:
                id (str): The id of the model. e.g. 11, 13, 16, 19
                nc (int): The number of classes.
                pretrained (bool): Whether to use pretrained weights.
            """
            if id == 11:
                return models.vgg11(num_classes=nc,pretrained=pretrained)
            elif id == 13:
                return models.vgg13(num_classes=nc,pretrained=pretrained)
            elif id == 16:
                return models.vgg16(num_classes=nc,pretrained=pretrained)
            elif id == 19:
                return models.vgg19(num_classes=nc,pretrained=pretrained)
            else:
                raise Exception(f"Unknown vgg id: {id}")
        _registered_models["vgg"] = create_vgg
        
        def create_mobilenet(id,nc,pretrained=False):
            """
            MobileNet is a series of lightweight deep learning network architectures designed for mobile and embedded vision applications.
            Args:
                id (str): The id of the model. e.g. v1, v2, v3_large, v3_small
                nc (int): The number of classes.
                pretrained (bool): Whether to use pretrained weights.
            """
            if id == 'v1':
                from ignet_ai_train.nn.modules import mobilenet_v1
                return mobilenet_v1(num_classes=nc)
            elif id == 'v2':
                return models.mobilenet_v2(num_classes=nc,pretrained=pretrained)
            elif id == "v3_large":
                return models.mobilenet_v3_large(num_classes=nc,pretrained=pretrained)
            elif id == "v3_small":
                return models.mobilenet_v3_small(num_classes=nc,pretrained=pretrained)
        _registered_models["mobilenet"] = create_mobilenet
    except ImportError:
        pass

def create_model(name,*args,**kwargs):
    if not _registered_models: _init_registered_models()
    if name in _registered_models:
        return _registered_models[name](*args,**kwargs)
    else:
        raise Exception(f"Unknown model: {name}")


def _process_dynamic_args(original_object,dynamic_args):
    if not isinstance(original_object,(str,list)):
        return original_object
    
    if isinstance(original_object,str):
        if original_object in dynamic_args:
            return dynamic_args[original_object]
        else:
            return original_object
    elif isinstance(original_object, list):
        new_list = [None for _ in range(len(original_object))]
        for i,value in enumerate(original_object):
            new_list[i] = _process_dynamic_args(value,dynamic_args)
        return new_list
    

if __name__ == "__main__":
    _init_registered_models()
    import json
    with open("models.json","w") as f:
        json.dump(list(_registered_models.keys()),f)