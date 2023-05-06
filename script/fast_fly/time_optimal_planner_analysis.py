import numpy as np
import casadi as ca

import time

import sys, os
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel
from time_optimal_planner import WayPointOpt, cal_Ns
from gates.gates import Gates

def analysis(quad, gates:Gates, ds:list, dt0s:list, sample_num:int=50):
    ts = []
    opt_ts = []
    Hs = []
    noises = []
    idxes = []
    for _ in range(sample_num):
        noises.append(np.random.randn(3)*0.5-0.25)
        idxes.append(np.random.randint(0, gates._N))
        
    for i, d in enumerate(ds):
        print("processed: ", i, "/", len(ds))
        Ns = cal_Ns(gates, d, loop=True)
        dts = np.array([0.1]*gates._N)
        opt = WayPointOpt(quad, gates._N, Ns, loop=True)
        opt.define_opt()
        opt.define_opt_t()
        Hs.append(opt._Herizon)
        # t1 = time.time()
        
        opt.solve_opt([], np.array(gates._pos).flatten(), dts)
        res = opt.solve_opt_t([], np.array(gates._pos).flatten())
        # print(res["lam_g"].full().flatten().shape)
        
        _t_a = []
        for _ in range(sample_num):
            # idx = np.random.randint(0, gates._N)
            # noise = np.random.randn(3)*0-0.0
            pos_tmp = np.array(gates._pos)
            pos_tmp[idxes[_]] += noises[_]
            res = opt.solve_opt_t([], pos_tmp.flatten(), warm=True)
            t1 = time.time()
            res = opt.solve_opt_t([], np.array(gates._pos).flatten(), warm=True)
            t2 = time.time()
            _t_a.append(t2-t1)
        
        ts.append([np.mean(_t_a), np.std(_t_a)])
        print(ts[-1])
                
        opt_dts = res['x'].full().flatten()[-opt._wp_num:]
        opt_t = 0
        for j, n in enumerate(Ns):
            opt_t += n*opt_dts[j]
        opt_ts.append(opt_t)

    return ts, opt_ts, Hs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    quad = QuadrotorModel(BASEPATH+"quad/quad.yaml")
    gates = Gates(BASEPATH+"gates/gates_real1.yaml")
    
    ds = [1/1, 1/1.5, 1/2, 1/3, 1/3.5, 1/4, 1/4.5, 1/5, 1/5.5, 1/6, 1/6.5, 1/7, 1/7.5, 1/8, 1/8.5, 1/9, 1/9.5, 1/10]
    dt0s = [1/1, 1/1.5, 1/2, 1/3, 1/3.5, 1/4, 1/4.5, 1/5, 1/5.5, 1/6, 1/6.5, 1/7, 1/7.5, 1/8, 1/8.5, 1/9, 1/9.5, 1/10]
    ts, opt_ts, Hs = analysis(quad, gates, ds, dt0s)
    print(ts)
    print(opt_ts)
    
    fig = plt.figure(figsize=(6,5.5))
    ax = fig.add_subplot(111)
    ax.plot(1/np.array(ds), np.array(ts)[:,0], linestyle="-")
    ax.twinx().plot(1/np.array(ds), opt_ts, linestyle="--")
    # plt.legend()
    plt.show()
