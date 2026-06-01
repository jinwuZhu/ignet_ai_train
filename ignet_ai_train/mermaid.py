#!/usr/bin/env python3
"""
将包含并行模块的模型 YAML 转换为 Mermaid 流程图。
修正：避免重复边，只在一个地方绘制所有连接。
"""
import sys
import yaml

def resolve_params(params, args):
    out = []
    for p in params:
        if isinstance(p, str) and p in args:
            out.append(args[p])
        elif isinstance(p, list):
            out.append(resolve_params(p, args))
        else:
            out.append(p)
    return out

def format_params(params):
    return ", ".join(str(p) for p in params)

def parse_source(src, nodes, module_last_node, module_order, current_module, current_index):
    if isinstance(src, str) and '.' in src:
        mod, idx = src.split('.', 1)
        if mod not in module_order and mod not in nodes:
            print(f"WARNING: 模块 '{mod}' 未定义，已使用当前模块 '{current_module}' 的同索引节点",
                  file=sys.stderr)
            mod = current_module
        if idx == '-1':
            return module_last_node.get(mod, f"{mod}_last")
        else:
            return f"{mod}_{idx}"
    elif src == -1:
        return None
    else:
        return str(src)

def build_graph(config):
    args = config.get('args', {})
    raw_modules = config['modules']
    all_module_names = set()
    def collect_names(item):
        if isinstance(item, list):
            for sub in item:
                collect_names(sub)
        else:
            all_module_names.add(item)
    collect_names(raw_modules)

    nodes = {}
    module_last_node = {}

    input_node_id = "Input"
    nodes[input_node_id] = {
        'id': input_node_id,
        'label': 'Input',
        'inputs': []
    }

    prev_outputs = [input_node_id]

    def process_sequential_module(module_name, input_node_ids):
        if module_name not in config:
            raise ValueError(f"模块 '{module_name}' 未定义")
        instructions = config[module_name]
        prev_node_id = None
        external_prev = input_node_ids[0] if input_node_ids else None

        for i, instr in enumerate(instructions):
            input_spec, repeat, op_name, raw_params = instr
            params = resolve_params(raw_params, args)
            node_id = f"{module_name}_{i}"

            label_parts = []
            if repeat > 1:
                label_parts.append(f"{repeat}×")
            label_parts.append(op_name)
            if params:
                label_parts.append(f"({format_params(params)})")
            label = " ".join(label_parts)

            inputs = []
            if isinstance(input_spec, list):
                for src in input_spec:
                    res = parse_source(src, nodes, module_last_node, list(all_module_names),
                                       module_name, i)
                    if res:
                        inputs.append(res)
            else:
                src = input_spec
                if src == -1:
                    if i == 0:
                        if external_prev:
                            inputs.append(external_prev)
                    else:
                        inputs.append(prev_node_id)
                else:
                    res = parse_source(src, nodes, module_last_node, list(all_module_names),
                                       module_name, i)
                    if res:
                        inputs.append(res)

            node = {
                'id': node_id,
                'label': label,
                'inputs': inputs
            }
            nodes[node_id] = node
            prev_node_id = node_id

        module_last_node[module_name] = prev_node_id
        return prev_node_id

    for item in raw_modules:
        if isinstance(item, list):
            branch_outputs = []
            for branch_name in item:
                out = process_sequential_module(branch_name, prev_outputs)
                branch_outputs.append(out)
            prev_outputs = branch_outputs
        else:
            out = process_sequential_module(item, prev_outputs)
            prev_outputs = [out]

    return nodes

def generate_mermaid(nodes, config):
    lines = ["graph TD"]
    # 1. 输入节点
    if "Input" in nodes:
        lines.append(f'    {nodes["Input"]["id"]}["{nodes["Input"]["label"]}"]')

    # 2. 收集所有模块名（展开并行）
    all_mods = []
    def collect_mods(item):
        if isinstance(item, list):
            for sub in item:
                collect_mods(sub)
        else:
            all_mods.append(item)
    collect_mods(config['modules'])

    # 3. 为每个模块生成子图，仅包含节点定义，不绘制任何边
    for mod in all_mods:
        lines.append(f"    subgraph {mod}")
        mod_nodes = [nd for nd in nodes.values() if nd['id'].startswith(f"{mod}_")]
        for nd in mod_nodes:
            lines.append(f'        {nd["id"]}["{nd["label"]}"]')
        lines.append("    end")

    # 4. 收集所有边，去重后统一绘制
    edges = set()
    for nd in nodes.values():
        for inp in nd['inputs']:
            if inp and nd['id'] != "Input":
                edges.add(f"    {inp} --> {nd['id']}")
    for edge in sorted(edges):
        lines.append(edge)

    return "\n".join(lines)

def yaml_to_mermaid(yaml_file):
    with open(yaml_file, 'r',encoding="utf8") as f:
        config = yaml.safe_load(f)
    nodes = build_graph(config)
    return generate_mermaid(nodes, config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python yaml_to_mermaid.py config.yaml")
        sys.exit(1)
    print(yaml_to_mermaid(sys.argv[1]))