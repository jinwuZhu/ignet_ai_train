import os

IGNET_HOME_PATH = os.path.abspath(os.environ.get("IGNET_HOME",".ignet_"))
if not os.path.exists(IGNET_HOME_PATH):
    os.makedirs(IGNET_HOME_PATH)
IGNET_HOME_CACHE_PATH = os.path.join(IGNET_HOME_PATH, "cache")
if not os.path.exists(IGNET_HOME_CACHE_PATH):
    os.makedirs(IGNET_HOME_CACHE_PATH)