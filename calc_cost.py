#!/usr/bin/python3.6
limit_time = 0.7
print("limit time:",limit_time)

def calc(file_name):
    J = 0.0
    t0,tf,num = None,None,0
    for line in open(file_name, "r"):
        hoge = [float(e) for e in line.split()]
        t = hoge[0]
        if t > limit_time:
            continue
        L = hoge[1]
        J += L
        if t0 is None:
            t0 = t
        tf = t
        num += 1
    dt = (tf-t0) / (num-1)
    return J * dt

if __name__ == '__main__':
    files = "lqr_result.log", "slqr_result.log", "mpc_result.log"
    costs = []
    for f in files:
        costs.append(calc(f))
    hoges = [print(files[i], costs[i]) for i in range(len(files))]
    print(costs[-1] / costs[0])
