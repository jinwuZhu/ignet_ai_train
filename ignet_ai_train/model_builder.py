import yaml
import requests
from typing import Any, Callable
import torch
import torch.nn as nn
from ignet_ai_train.nn.create import create_model

class DynamicModel(nn.Module):
    def __init__(
            self, 
            layers:dict[str,nn.Module], 
            config:dict, 
            modules_order:list[str|list[str]] = None, 
            dynamic_args:dict = None):
        super().__init__()
        self.layers = nn.ModuleList(list(layers.values()))
        self.layer_id_to_idx = {lid: idx for idx, lid in enumerate(layers.keys())}
        self.config = config
        self.modules_order = modules_order if modules_order else config.get('modules',None)
        self.dynamic_args = dynamic_args if dynamic_args else config.get('args',{})
    
    def get_layer(self, layer_id:str) -> nn.Module:
        return self.layers[self.layer_id_to_idx[layer_id]]

    def _resolve_input(self, input_spec, all_outputs, module_input, prev_layer_output):
        """
        解析输入引用
        - -1: 前一层的输出（若无则为模块输入）
        - "module.index": 指定层的输出（index支持负数，表示倒数第N层）
        - [list]: 多个输入来源
        """
        if isinstance(input_spec, int):
            # TODO - N 索引 表示前 N 层输出
            # -1 表示前一层的输出
            if input_spec == -1:
                return prev_layer_output if prev_layer_output is not None else module_input
            return module_input
        
        elif isinstance(input_spec, str):
            # 处理 "module.index" 格式
            if '.' in input_spec:
                parts = input_spec.rsplit('.', 1)
                if len(parts) == 2:
                    mod_name, idx_str = parts
                    try:
                        idx = int(idx_str)
                        key = f"{mod_name}.{idx}"
                        
                        # 正数索引：直接查找
                        if idx >= 0 and key in all_outputs:
                            return all_outputs[key]
                        
                        # 负数索引：倒数第N层
                        if idx < 0:
                            mod_layers = sorted(
                                [k for k in all_outputs.keys() if k.startswith(mod_name + '.')],
                                key=lambda x: int(x.split('.')[-1])
                            )
                            if mod_layers and abs(idx) <= len(mod_layers):
                                return all_outputs[mod_layers[idx]]
                    except (ValueError, KeyError):
                        pass
            return all_outputs.get(input_spec, module_input)
        
        elif isinstance(input_spec, list):
            # 多个输入来源
            return [self._resolve_input(spec, all_outputs, module_input, prev_layer_output) 
                    for spec in input_spec]
        elif input_spec == None:
            return [] # 空输入
        return module_input
    
    def forward(self, x):
        all_outputs = {'x': x}
        module_group_input = x
        
        # 按模块组执行
        for module_blocks in self.modules_order:
            if isinstance(module_blocks, str):
                module_blocks = [module_blocks]
            
            # 当前模块组的输入
            group_input = module_group_input
            
            # 执行模块组内的所有模块
            for module_name in module_blocks:
                block_layers = self.config.get(module_name)
                if block_layers is None:
                    continue
                
                # 该模块的输入
                module_input = group_input
                prev_layer_output = None
                
                for layer_idx, layer_config in enumerate(block_layers):
                    # 重新处理动态参数（因为可能涉及运行时的动态值）
                    layer_config = list(layer_config)
                    for i, cfg in enumerate(layer_config):
                        layer_config[i] = _process_dynamic_args(cfg, self.dynamic_args)
                    
                    layer_input_spec = layer_config[0]
                    layer_id = f"{module_name}.{layer_idx}"
                    
                    # 解析该层的输入
                    layer_input = self._resolve_input(layer_input_spec, all_outputs, module_input, prev_layer_output)
                    # if isinstance(layer_input, list):
                    #     for i, input in enumerate(layer_input):
                    #         print(f"Layer: {layer_id} Input[{i}]: {input.shape}")
                    # else:
                    #     print(f"Layer: {layer_id} Input: {layer_input.shape}")
                    
                    # 执行层
                    layer_module = self.layers[self.layer_id_to_idx[layer_id]]
                    
                    if isinstance(layer_input, list):
                        output = layer_module(*layer_input)
                    else:
                        output = layer_module(layer_input)
                    
                    # print(f"Layer: {layer_id} Output: {output.shape}")
                    all_outputs[layer_id] = output
                    prev_layer_output = output
                
                # 该模块的最后输出作为本组的最后输出
                module_group_input = prev_layer_output if prev_layer_output is not None else group_input
        
        return module_group_input


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

def _create_layer(
        config:dict[str,Any], 
        dynamic_args:dict[str,Any] = {}, 
        creater:Callable = create_model):
    layer_config = list(config)  # 创建副本避免修改原配置
    for i, config in enumerate(layer_config):
        layer_config[i] = _process_dynamic_args(config, dynamic_args)
                
    _, layer_repeat, layer_name, layer_args = layer_config
    if isinstance(layer_args,list):
        layer_args = tuple(layer_args)
    if layer_repeat > 1:
        layer_model = nn.Sequential(*[creater(layer_name, *layer_args) for _ in range(layer_repeat)])
    else:
        layer_model: nn.Module = creater(layer_name, *layer_args)
    return layer_model




def build_model_by_yaml(yaml_file: str, create_layer: Callable = create_model, **kwargs) -> nn.Module:
    """
    根据YAML配置文件动态构建神经网络模型
    
    YAML格式说明：
    ============
    args:               # 全局动态参数定义（可在层参数中引用）
      n: 64
      nc: 14
    
    modules:           # 模块执行顺序（列表中的模块并行执行，共享输入）
      - [module1, module2]  # 同一列表内的模块从相同输入开始
      - module3
      - module4
    
    module1:          # 模块定义（包含多层）
      - [input_from, repeat, layer_type, [layer_args]]
      
    输入源指定（input_from）:
    ========================
    -1                 # 前一层的输出（相对索引）
                       # 在模块第一层表示该模块的输入
    "module.index"     # 指定模块的指定层输出
                       # index 为正数表示第N个索引（从0开始）
                       # index 为负数表示倒数第N层
    [source1, source2] # 多输入来源（返回列表，供多输入层如Concat使用）
    
    参数说明：
    ========
    input_from:  输入来源（支持上述格式）
    repeat:      该层重复次数（>1时使用Sequential包装）
    layer_type:  层类型名称（需在create_layer中注册）
    layer_args:  层初始化参数（可包含args中定义的参数名，如'n'会被替换为64）
    
    示例：
    ====
    args:
      n: 64
      nc: 14
    modules:
      - [backbone, backbone2]
      - cat
      - head
    
    backbone:
      - [-1, 1, Conv2d, [3, 32, 3, 1, 1]]      # input: x
      - [-1, 1, ReLU, [False]]                 # input: backbone.0输出
      - [-1, 1, Conv2d, [32, n, 3, 1, 1]]     # input: backbone.1输出，n替换为64
    
    cat:
      - [[backbone.-1, backbone2.0], 1, Concat, []]  # input: [backbone最后层, backbone2第0层]
    
    head:
      - [-1, 1, Linear, [256, nc]]             # input: cat.0输出，nc替换为14
    """
    
    if yaml_file.startswith(("http://", "https://")):
        r = requests.get(yaml_file)
        r.raise_for_status()
        model_config = yaml.load(r.text, Loader=yaml.FullLoader)
    else:
        model_config: dict[str, Any] = yaml.load(open(yaml_file, "r", encoding='utf8'), Loader=yaml.FullLoader)
    return build_model(model_config, create_layer, **kwargs)


def build_model(
        config: dict, 
        model_creater: Callable = create_model, 
        **kwargs) -> nn.Module:
    """
    Build model by model config
    Args:
        config (dict): model config file path or url
        create_layer (Callable, optional): layer creator. Defaults to create_model.
        **kwargs: dynamic arguments
    Returns:
        nn.Module: model
    """    
    
    modules: list[str | list] = config.get("modules")
    dynamic_args: dict[str, Any] = config.get("args", {})
    # 合并 kwargs 和 dynamic_args 优先使用 kwargs
    dynamic_args = {**dynamic_args,**kwargs}
    
    # 第一步：创建所有的层模块
    all_layers: dict[str, nn.Module] = dict() # key: "module.layer_idx"
    for module_blocks in modules:
        if isinstance(module_blocks, str):
            module_blocks = [module_blocks]
        
        for module_block_key in module_blocks:
            block_layers: list = config.get(module_block_key)
            if block_layers is None:
                continue
            
            for layer_index, layer_config in enumerate(block_layers):
                layer_id = f"{module_block_key}.{layer_index}"
                # 处理动态参数替换
                layer_model = _create_layer(layer_config, dynamic_args, model_creater)
                
                all_layers[layer_id] = layer_model
    # 第二步：构建动态执行模型
    model = DynamicModel(all_layers, config, modules, dynamic_args)
    return model

if __name__ == "__main__":

    model = build_model_by_yaml("test/models/classify.yaml", create_model, nc=2, c=3)
    
    print("\n" + "="*50)
    print("模型构建完成！")
    print("="*50)
    
    # 测试模型
    x = torch.randn(1, 3, 224, 224)
    print(f"\n输入形状: {x.shape}")
    
    y = model(x)
    print(f"输出形状: {y.shape}")
    
    # 打印模型结构
    print("\n模型结构:")
    # from torchsummary import summary
    # # 使用summary函数打印网络结构
    # summary(model, input_size=(3, 256, 256))
    # print(model)
    # # export onnx
    # torch.onnx.export(model, x, "test.onnx", export_params=True)
