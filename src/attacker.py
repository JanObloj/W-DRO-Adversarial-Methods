import torch
from robust_utils import loss_grad, eval

EPS = 1e-8
ITR = 50
RATIO = 1.875

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attacker:
    """
    W-PGD under (W_p,l_r) threat model
    Input:
    - network: the network to be attacked
    - q: the conjugate of p
    - s: the conjugate of r
    - delta: the attack budget
    - itr: the number of iterations
    - step_size: step size is given by ratio*delta/itr
    - loss_fn: the loss function used to train the network
    - X, y: the dataset to be attacked
    Formula: x_{t+1}=proj(x_{t}+step_size*adv_dir)
    """

    def __init__(
        self,
        q=None,
        s=None,
        delta=None,
        ratio=None,
        itr=None,
        loss_fn=None,
        X=None,
        y=None,
    ):
        self.q = q
        self.s = s
        self.delta = delta
        self.step_size = ratio * delta / itr
        self.itr = itr
        self.loss_fn = loss_fn
        self.X = X
        self.y = y

    def data_dist(self, X_1, X_2):
        """
        calculate the normalized lp distance between two data sets X_1 and X_2
        it is used as an approximation to the p-Wasserstein distance between X_1 an X_2
        """
        if self.s == 1:
            r = float("inf")
        else:
            r = 2
        if self.q == 1:
            p = float("inf")
        else:
            p = 2
        norm = (X_1 - X_2).norm(p=r, dim=(1, 2, 3))  # shape: (batch_size)
        dist = norm.pow(p).mean().pow(1 / p).item()
        return dist

    def proj(self, X_cur):
        """
        project X_cur to delta Wasserstein-p ball arround X if X_cur is outside the ball

        if q==1, we calculate the distance between each data point, and project each data point
        if q!=1, we calculate the W-distance between the whole batch, and project the whole batch
        """

        if self.q == 1:
            if self.s == 1:
                # clamp because of linf
                return self.X + (X_cur - self.X).clamp(-self.delta, self.delta)
            else:
                r = 2
                # a point-wise contraction because of w_infty l2
                return self.X + (X_cur - self.X) * self.delta / (
                    (X_cur - self.X).norm(p=r, dim=(1, 2, 3)) + EPS
                ).reshape(-1, 1, 1, 1)
        else:
            dist = self.data_dist(X_cur, self.X)
            r = min(1.0, self.delta / (dist + EPS))  # the contraction ratio
            img_dist = (
                (X_cur - self.X)
                .norm(p=float("inf"), dim=(1, 2, 3))
                .reshape(-1, 1, 1, 1)
            )  # (b,1,1,1)
            if self.s == 1:
                return self.X + (X_cur - self.X).clamp(-r * img_dist, r * img_dist)
            else:
                return self.X + (X_cur - self.X) * r

    def adv_dir(self, X_cur, network):
        """
        get the advesarial direction at X_cur
        the output length is normalized to "1"
        """
        grad = loss_grad(
            network,
            X=X_cur.detach().cpu().clone(),
            y=self.y.detach().cpu().clone(),
            loss_fn=self.loss_fn,
        )  # shape: (dataset_size,3,32,32)
        grad_norm = grad.norm(p=self.s, dim=(1, 2, 3)).reshape(
            (-1, 1, 1, 1)
        )  # (dataset_size,1,1,1)
        direction = (
            torch.sign(grad)
            * torch.abs(grad).pow(self.s - 1)
            * (grad_norm + EPS).pow(self.q - self.s)
        )  # (dataset_size,3,32,32)
        if self.q == 1:
            return direction
        else:
            return direction / (self.data_dist(0, direction) + EPS)

    def attack(self, network, verbose=True):
        """
        run PGD attack with the given parameters
        find the best among the trajectory
        """

        # if randomize:
        #     # randomize the initial point inside the ball
        #     rand_dt = random.uniform(0, self.delta)
        #     rand_delta = np.random.rand(*self.X.shape) * 2 * rand_dt - rand_dt
        #     X_cur = self.proj(self.X + rand_delta).detach().cpu().float().clone()
        #     X_cur = X_cur.clamp(0, 1)
        # else:
        #     X_cur = self.X.detach().cpu().clone()
        network.to(device).eval()
        X_cur = self.X.detach().cpu().clone()
        X_adv = X_cur.detach().cpu().clone()

        acc_min = 1.0
        loss_max = -1.0

        for i in range(self.itr):
            # update X_adv
            X_cur += self.step_size * self.adv_dir(X_cur, network)
            X_cur = self.proj(X_cur).clamp(0, 1)

            X_cur_eval = eval(
                network,
                X=X_cur.detach().cpu().clone(),
                y=self.y.detach().cpu().clone(),
                loss_fn=self.loss_fn,
            )
            loss = X_cur_eval["loss"]
            acc = X_cur_eval["acc"]
            if verbose:
                print(f"step_{i}:{acc}")
            if acc_min > acc:
                X_adv = X_cur.detach().cpu().clone()
                acc_min = acc
            if loss_max < loss:
                loss_max = loss
        if verbose:
            return {"X_adv": X_adv, "acc_min": acc_min, "loss_max": loss_max}
        else:
            return {"acc_min": acc_min, "loss_max": loss_max}


def wfgsm(q, s, delta, loss_fn, X=None, y=None):
    return Attacker(
        q=q,
        s=s,
        delta=delta,
        ratio=1,
        itr=1,
        loss_fn=loss_fn,
        X=X,
        y=y,
    )


def wpgd(q, s, delta, loss_fn, X=None, y=None):
    return Attacker(
        q=q,
        s=s,
        delta=delta,
        ratio=RATIO,
        itr=ITR,
        loss_fn=loss_fn,
        X=X,
        y=y,
    )
