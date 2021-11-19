import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import taichi as ti
import sys

ti.init(arch=ti.cpu)

@ti.data_oriented
class OneD_Advec_FD_Exp:
    def __init__(self,
                u = 1.0,
                C = 0.9,
                N = 65,
                Ng = 1,
                time = 1.0,
                xmin = 0.0,
                xmax = 1.0,
                Method = "upwind") -> None:
        self.u = u
        self.C = C
        self.N = N
        self.Ng = Ng
        self.begin = Ng
        self.end = Ng+N-1
        self.xmin = xmin
        self.xmax = xmax
        self.time = time*(xmax-xmin)/u
        self.delta_x = 1.0 / (N-1)
        self.delta_t = C*self.delta_x/u
        self.Method = Method
        self.x = (np.arange(N+2*Ng)-Ng)*self.delta_x
        self.a_new = ti.field(dtype=ti.f32,shape=(N+2*Ng))
        self.a_old = ti.field(dtype=ti.f32,shape=(N+2*Ng))
        self.exact = ti.field(dtype=ti.f32,shape=(N+2*Ng))

        arr = np.zeros(N+2*Ng,dtype=np.float32)
        self.a_new.from_numpy(arr)
        self.a_old.from_numpy(arr)
        self.exact.from_numpy(arr)


    @ti.kernel
    def initial_condition(self):
        for i in self.a_new:
            x = (i-self.Ng)*self.delta_x
            if(x >= 1.0/3.0 and x < 2.0/3.0):
                self.a_new[i] = 1.0
            else:
                self.a_new[i] = 0.0
            self.a_old[i] = self.a_new[i]
            self.exact[i] = self.a_new[i]


    @ti.func
    def boundary_condition(self):
        self.a_new[self.begin-1] = self.a_new[self.end-1]
        self.a_new[self.end+1] = self.a_new[self.begin+1]


    @ti.kernel
    def method(self):
        for i in range(self.begin,self.end+1):
            if(self.Method == "upwind"):
                self.a_new[i] = self.a_old[i] - self.C*(self.a_old[i]-self.a_old[i-1])
            if(self.Method == "FTSC"):
                self.a_new[i] = self.a_old[i] - 0.5*self.C*(self.a_old[i+1]-self.a_old[i-1])
        self.boundary_condition()

        for i in self.a_old:
            self.a_old[i] = self.a_new[i]


    def solve(self):
        self.initial_condition()
        t = 0.0
        while t < self.time:
            self.method()
            t += self.delta_t


    def printimg(self):
        solution = res.a_new.to_numpy()
        plt.plot(res.x[res.begin:res.end+1], solution[res.begin:res.end+1], label=r"$C = {}$".format(i))


Clist = [0.1, 0.5, 0.9]
for i in Clist:
    res = OneD_Advec_FD_Exp(C = i,Method="upwind",time=1.0)
    res.solve()
    res.printimg()

exact = res.exact.to_numpy()
plt.plot(res.x[res.begin:res.end+1], exact[res.begin:res.end+1], ls=":", label="exact")

plt.legend(frameon=False, loc="best")
plt.tight_layout()
plt.show()