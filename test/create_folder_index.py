import os
import sys
from pathlib import Path


def create_folder_index(folder_path):
    """
    遍历文件夹并生成main.txt文件，包含相对路径列表
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"错误: 文件夹不存在 - {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"错误: 路径不是文件夹 - {folder_path}")
        return
    
    # 遍历所有文件
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = Path(root) / file
            relative_path = full_path.relative_to(folder_path)
            # 转换为正斜杠格式
            file_paths.append(str(relative_path).replace('\\', '/'))
    
    # 排序相对路径
    file_paths.sort()
    
    # 生成main.txt文件
    output_file = folder_path / "main.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in file_paths:
            f.write(file_path + '\n')
    
    print(f"成功生成 main.txt，共 {len(file_paths)} 个文件")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python create_folder_index.py <文件夹路径>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    create_folder_index(folder_path)
