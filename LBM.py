from numpy.core.fromnumeric import shape
import taichi as ti
import numpy as np


ti.init(arch=ti.cpu)

@ti.data_oriented
class lbm:
    def __init__(self,
                Nx=200,
                Ny=200,
                niu=0.01,
                nit = 60000):
        self.Nx = Nx
        self.Ny = Ny
        self.niu = niu
        self.c_s = 1.0/np.sqrt(3)
        self.tau = self.c_s**2 * self.niu + 0.5
        self.rho = ti.field(dtype=ti.f32, shape=(Nx, Ny))
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(Nx, Ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(Nx, Ny))
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(Nx, Ny))
        self.F = ti.Vector.field(2, dtype=ti.f32, shape=(Nx, Ny))
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.c = ti.field(dtype=ti.f32, shape=(9,2))

        arr = np.array([ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
        self.w.from_numpy(arr)
        arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.c.from_numpy(arr)

    @ti.func
    def computeFeq(self, i, j, k):
        eu = self.velocity[i, j][0] * self.c[k, 0] + self.velocity[i, j][1] * self.c[k, 1]
        uv = self.velocity[i,  j][0]**2 + self.velocity[i, j][1]**2
        return self.w[k] * self.rho[i, j] * (1.0 + eu / self.c_s**2 + eu * eu / (2 * self.c_s**4) - uv / (2 * self.c_s**2))

    @ti.kernel
    def initlize(self):
        for i, j in self.velocity:
            self.velocity[i, j][0] = 0.0
            self.velocity[i, j][1] = 0.0
            self.rho[i, j] = 0.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.computeFeq(self, i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]

    @ti.kernel
    def collision_and_stream(self):
        for i, j in ti.ndrang((1, self.Nx - 1), (1, self.Ny - 1)):
            for k in ti.static(range(9)):
                self.f_new[i,j][k] = (1.0 - 1.0 / self.tau) * self.f_old[i - self.c[k, 0],j - self.c[k, 1]][k] +\
                                    self.f_eq(i - self.c[k, 0],j - self.c[k, 1],k) / self.tau

    @ti.kernel
    def apply_boundary(self):