import sys
import torch
import robustbench as rb

from loss_fns import *
from utils import sizing, saveto, if_exist
from robust_utils import *

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
        self.signature = f"-w_{self.w_norm}-l_{self.l_norm}-{loss_fn}"


def main(exp_name):
    networks_linf = list(rb.model_zoo.cifar10.linf.keys())  # l2, common_corruptions
    networks_l2 = list(rb.model_zoo.cifar10.l2.keys())  # l2, common_corruptions

    network_index = int(sys.argv[1])
    if network_index < len(networks_linf):
        mname = networks_linf[network_index]
        threat_model = "Linf"
        s = 1
    else:
        mname = networks_l2[network_index - len(networks_linf)]
        threat_model = "L2"
        s = 2

    folder_name = f"../network_stats/{threat_model}/{mname}"

    # experiment parameters
    q = int(sys.argv[2])
    if sys.argv[3] == "CE":
        loss_fn = CE()
    elif sys.argv[3] == "DLR":
        loss_fn = DLR()
    elif sys.argv[3] == "ReDLR":
        loss_fn = ReDLR()
    else:
        raise NotImplementedError
    p = Params(q, s, threat_model, loss_fn)
    if if_exist(folder_name, exp_name + p.signature):
        print(f"a network is skipped since it has been calculated")
    else:
        print(
            f"dealing with network {mname}, {threat_model=}, q={p.q}, loss={p.loss_fn}"
        )
        network = rb.utils.load_model(
            model_name=mname, threat_model=threat_model, dataset="cifar10"
        )
        network.eval()
        print("loaded model from RobustBench, size = ", sizing(network))

        dict = eval(network, loss_fn=p.loss_fn, cond=True)
        dict[f"Upsilon"] = comp_upsilon(network, loss_fn=p.loss_fn, q=p.q, s=p.s)
        dict["q"] = p.q
        dict["s"] = p.s
        dict["mname"] = mname
        dict["loss_fn"] = p.loss_fn
        saveto(dict, folder_name, exp_name + p.signature)


if __name__ == "__main__":
    main("clean")
