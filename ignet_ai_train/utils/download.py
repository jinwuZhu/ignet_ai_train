import os
import urllib.request
from urllib.parse import urljoin
import urllib.error

from hashlib import md5
import shutil
import requests
from ignet_ai_train.config import IGNET_HOME_CACHE_PATH

def download_resources_list(url_to_main_txt:str, destination_folder:str):
    """
    从一个URL列表文件 (main.txt) 中读取资源路径并下载到本地，保持文件夹结构。

    Args:
        url_to_main_txt (str): 指向 main.txt 文件的完整URL。
                               例如: "https://example.com/config/main.txt"
        destination_folder (str): 本地存放所有下载文件的目标文件夹路径。
                                 例如: "./downloads"

    Returns:
        None: 函数没有返回值，但会在控制台打印操作过程和结果。
    """
    with urllib.request.urlopen(url_to_main_txt) as response:
        content:str = response.read().decode('utf-8')

    # 解析URL列表
    urls = [line.strip() for line in content.splitlines() if line.strip()]
    
    if not urls:
        return

    # 获取main.txt所在目录的基础URL，用于构建相对路径
    base_url_of_main = urljoin(url_to_main_txt, '.')

    # 2. 遍历并下载每个资源
    for relative_path in urls:
        # 构建完整的资源URL
        resource_full_url = urljoin(base_url_of_main, relative_path)
        
        # 解析路径，防止路径遍历攻击 (e.g., ../../)
        parsed_path = os.path.normpath(relative_path).replace(os.sep, '/')
        
        # 计算本地保存的完整路径
        local_file_path = os.path.join(destination_folder, parsed_path)
        
        # 安全检查：确保最终路径在目标文件夹内，防止路径遍历攻击
        real_destination = os.path.abspath(destination_folder)
        real_file_path = os.path.abspath(local_file_path)
        
        if not real_file_path.startswith(real_destination + os.sep) and real_file_path != real_destination:
            print(f"[WARN] Dangerous operation. Attempting to write content outside the target path.")
            print(f"[WARN] Skipping: {resource_full_url} -> {real_file_path}")
            continue

        # 创建必要的目录结构
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        try:
            # 下载文件
            urllib.request.urlretrieve(resource_full_url, local_file_path)
        except urllib.error.URLError as e:
            print("[WARN] Don't have permission to download {resource_full_url}")

def auto_cache_download_list(root:str)->str:
    """
    创建缓存并自动下载资源列表, 如果资源存在且root的缓存值一致，则使用缓存，否则下载并缓存。
    Args:
        root (str): 资源列表URL，例如：https://example.com/config/main.txt
    Returns:
        str: 下载后的目录路径
    """
    md5_root = md5(root.encode()).hexdigest()
    dl_dir = os.path.join(IGNET_HOME_CACHE_PATH, md5_root)
    resources_main_path = os.path.join(IGNET_HOME_CACHE_PATH, f"{md5_root}.cache")
        # 比较网络上的文件和本地文件内容是否一致，如果一致表示已经缓存且没有更新
    is_cache_valid = False
    remote_file_data = requests.get(root).content
    if os.path.exists(resources_main_path):
        with open(resources_main_path, "rb") as f:
            local_file_data = f.read()
                # 
        is_cache_valid = local_file_data == remote_file_data

    if is_cache_valid:
        print(f"Using cached resources from {resources_main_path}")
    else:
            # 删除旧的数据
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
            # 下载新的数据
        print(f"Downloading resources from {root} \r\n    -> {dl_dir}")
        download_resources_list(root, dl_dir)
        with open(resources_main_path, "wb") as f:
            f.write(remote_file_data)
    return dl_dir


if __name__ == "__main__":
    # 您需要将这些URL和路径替换为您自己的测试地址
    main_txt_url = "http://192.168.74.132:1681/datasets/coco8/images.txt"
    download_dir = "./downloaded_resources"
    download_resources_list(main_txt_url, download_dir)