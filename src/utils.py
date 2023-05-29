import pickle
import os


def dict_translate(dict, s="_"):
    d = {}
    for k, v in dict.items():
        d[s + k] = v
    return d


def sizing(network):
    param_size = 0
    for param in network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print("network size: {:.3f}MB".format(size_all_mb))
    return size_all_mb


def saveto(dict, folder_name, exp_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump(dict, open(f"{folder_name}/{exp_name}.pkl", "wb"))
    print(f"saved to {folder_name}/{exp_name}.pkl")


def if_exist(folder_name, exp_name):
    return os.path.exists(f"{folder_name}/{exp_name}.pkl")
