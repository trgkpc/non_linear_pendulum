#!/usr/bin/python3.6
from sympy import *
import numpy as np

M,m,l,Dx,Dtheta,g = [Symbol(moji) for moji in "M,m,l,Dx,Dtheta,g".split(",")]

def calc_f(x, u, four_per_three):
    x1, x2, x3, x4 = x

    f1 = x3
    f2 = x4
    tmp = (M+m)*(four_per_three)*m*(l**2) - (m*l*cos(x2))**2
    f3 = ((four_per_three)*(m**2)*(l**3)*sin(x2)*(x4**2) - (four_per_three)*Dx*m*(l**2)*x3 - (m**2)*(l**2)*g*sin(x2)*cos(x2) + Dtheta*m*l*x4*cos(x2) + (four_per_three)*m*(l**2)*u) / tmp
    f4 = (-(m*l*x4)**2*sin(x2)*cos(x2) + Dx*m*l*x3*cos(x2) + (M+m)*m*g*l*sin(x2) - (M+m)*Dtheta*x4 - m*l*cos(x2)*u) / tmp
    return [f1, f2, f3, f4]

x = [Symbol('x' + str(num)) for num in [1,2,3,4]]
u = Symbol('u')

f = calc_f(x, u, Symbol('(4/3)'))

print("#!/usr/bin/python3.6")
print()
print("import numpy as np")
print("from numpy import cos")
print("from numpy import sin")
print()

def print_params(x="x", u="u[0]", params="params"):
    for param_name in ["M","m","l","Dx","Dtheta","g"]:
        print("    " + param_name  + " = " + str(params) + "['" + param_name + "']")
    print("    x1, x2, x3, x4 = " + str(x))
    print("    u = " + str(u))

print("def f(x, u, params):")
print_params()
print("    return np.array(",f,")")
print()

print("def A(x, params):")
print_params(u=0)
print("    return np.array(",[[diff(fi,e) for e in x] for fi in f],")")
print()

print("def B(x, params):")
print_params(u=0)
print("    return np.array(",[[diff(fi,u_) for u_ in [u]] for fi in f],")")
print()

