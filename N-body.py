import taichi as ti

ti.init(arch=ti.cpu)

PI = 3.14159265254
G = 1
m = 2

stime = 1e-4

N = 512

position = ti.Vector.field(2, ti.f32, N)
F = ti.Vector.field(2, ti.f32, N)
velocity = ti.Vector.field(2, ti.f32, N)


@ti.kernel
def Initial():
    center = ti.Vector([0.5, 0.5])
    for i in range(N):
        theta = ti.random() * 2 * PI
        r = (0.3 + 0.7 * ti.random()) * 0.4
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        position[i] = center + offset
        velocity[i] = [-offset.y, offset.x]


@ti.kernel
def ComputeVelocity():
    for i in range(N):
        F[i] = ti.Vector([0, 0])
        p = position[i]
        for j in range(i):
            diff = p - position[j]
            r = diff.norm(1e-5)
            f = -G * m * m * (1.0 / r)**3 * diff
            F[i] += f
            F[j] -= f


@ti.kernel
def Update():
    for i in range(N):
        velocity[i] += F[i] * stime/m
        position[i] += velocity[i] * stime


Initial()

gui = ti.GUI('N-body', (512, 512))

while gui.running:

    ComputeVelocity()
    Update()

    gui.clear(0x112F41)
    gui.circles(position.to_numpy(), color=0xffffff, radius=2)
    gui.show()
