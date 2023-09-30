import torch
import robustbench as rb

from loss_fns import *
from utils import sizing, saveto, if_exist
from robust_utils import *
from attacker import wfgsm, wpgd
from data import get_cifar10_split, get_cifar100_split
from imagenet_loader import get_imagenet_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device=}")
torch.manual_seed(42)


def conjugate_norm(q):
    if q == 1:
        return "inf"
    elif q == 2:
        return "2"
    else:
        raise NotImplementedError


class Params:
    def __init__(self, q, s, att, loss_fn):
        self.q = q
        self.s = s
        self.att = att

        self.w_norm = conjugate_norm(q)
        self.l_norm = conjugate_norm(s)

        self.loss_fn = loss_fn
        self.signature = f"w_{self.w_norm}-l_{self.l_norm}-{loss_fn}"


def main():
    dataset = sys.argv[1]
    if dataset == "cifar10":
        X, y = get_cifar10_split()
        networks_linf = list(rb.model_zoo.cifar10.linf.keys())
        networks_l2 = list(rb.model_zoo.cifar10.l2.keys())
    if dataset == "cifar100":
        X, y = get_cifar100_split()
        networks_linf = list(rb.model_zoo.cifar100.linf.keys())
        # networks_l2 = list(rb.model_zoo.cifar100.l2.keys())
    if dataset == "imagenet":
        networks_linf = list(rb.model_zoo.imagenet.linf.keys())
        # networks_l2 = list(rb.model_zoo.imagenet.l2.keys())
    

    network_index = int(sys.argv[2])
    if network_index < len(networks_linf):
        mname = networks_linf[network_index]
        threat_model = "Linf"
        s = 1
    else:
        mname = networks_l2[network_index - len(networks_linf)]
        threat_model = "L2"
        s = 2

    if dataset == "imagenet":
        # Res256Crop224 is the default preprocessing
        prep="Res256Crop224"
        # Get the preprocessing from the model zoo if exists
        if "preprocessing" in rb.model_zoo.imagenet.linf[mname]:
            prep=rb.model_zoo.imagenet.linf[mname]["preprocessing"]
        X, y = get_imagenet_split(transforms_test=prep)
        # X, y = get_imagenet_split()
        print(X.shape)
        print(y.shape)
    

    # experiment parameters
    q = int(sys.argv[3])
    if sys.argv[4] == "CE":
        loss_fn = CE()
    elif sys.argv[4] == "DLR":
        loss_fn = DLR()
    elif sys.argv[4] == "ReDLR":
        loss_fn = ReDLR()
    else:
        raise NotImplementedError
    p = Params(q, s, threat_model, loss_fn)

    attack_type = sys.argv[5]
    if attack_type != "clean":
        dt = int(sys.argv[6])
        if s == 1:
            delta = dt / 510
        if s == 2:
            delta = dt / 32
        if attack_type == "FGSM":
            attacker = wfgsm(
                q=p.q, s=p.s, delta=delta, loss_fn=p.loss_fn, X=X, y=y
            )
        elif attack_type == "PGD":
            attacker = wpgd(
                q=p.q, s=p.s, delta=delta, loss_fn=p.loss_fn, X=X, y=y
            )
        else:
            raise NotImplementedError
    if attack_type=="clean":
        folder_name = f"../network_stats/{dataset}/{attack_type}/" + p.signature
    else:
        folder_name = f"../network_stats/{dataset}/{attack_type}_" + str(dt) + "/" + p.signature
    if if_exist(folder_name, mname):
        print(f"a network is skipped since it has been calculated")
    else:
        print(
            f"dealing with network {mname}, {threat_model=}, q={p.q}, loss={p.loss_fn}"
        )
        network = rb.utils.load_model(
            model_name=mname, threat_model=threat_model, dataset=dataset
        )
        print("loaded model from RobustBench, size = ", sizing(network))
        if attack_type == "clean":
            dict = eval(network, X=X, y=y, loss_fn=p.loss_fn, cond=True)
            dict[f"Upsilon"] = comp_upsilon(
                network, X=X, y=y, loss_fn=p.loss_fn, q=p.q, s=p.s
            )
            dict["q"] = p.q
            dict["s"] = p.s
            dict["attack_type"] = attack_type
            dict["mname"] = mname
            dict["loss_fn"] = p.loss_fn
            saveto(dict, folder_name, mname)
        else:
            dict = attacker.attack(network, verbose=False)
            dict["q"] = p.q
            dict["s"] = p.s
            dict["delta"] = dt
            dict["attack_type"] = attack_type
            dict["mname"] = mname
            dict["loss_fn"] = p.loss_fn
            saveto(dict, folder_name, mname)


import sys

if __name__ == "__main__":
    main()
