import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_initial_distribution(x, y):
    # plot initial distribution
    plt.gca()
    plt.plot(x, y, '*k')
    plt.plot(x[0], y[0], '*k', label='initial distribution')


def calc_speed(u, v):
    return np.sqrt(u ** 2 + v ** 2)


def random_walk(alphaL, alphaT, diffCoef, x, y, dx, dy, speed, dt, numPoints):
    varL = 2 * (alphaL * speed + diffCoef) * dt
    varT = 2 * (alphaT * speed + diffCoef) * dt

    # create normally distributed random vector
    zL = np.random.normal(loc=0, scale=np.sqrt(varL), size=(1, numPoints))
    zT = np.random.normal(loc=0, scale=np.sqrt(varT), size=(1, numPoints))

    # Advect and then introduce random noise
    x_new = x + dx + (zL * dx / speed) + (zT * dy / speed)
    y_new = y + dy + (zL * dy / speed) - (zT * dx / speed)

    return x_new, y_new


def plot_variances(times, vx, vy):
    vx = vx.transpose()
    vy = vy.transpose()

    plt.figure(4)
    plt.plot(times, vx, 'o', color='tab:red', label='x-position')
    plt.plot(times, vy, 'o', color='tab:purple', label='y-position')
    plt.xlabel('time (s)')
    plt.ylabel('variance (m^2)')
    plt.title('Variance of Particle Position')

    # plot best fit line
    axv, bxv = np.polyfit(times, vx, 1)
    ayv, byv = np.polyfit(times, vy, 1)

    plt.plot(times, axv * times + bxv, '--', color='tab:olive', label=f'slope = {np.round(axv, 4)}')
    plt.plot(times, ayv * times + byv, '--', color='tab:cyan', label=f'slope = {np.round(bxv, 4)}')
    plt.legend()


def plot_means(times, mx, my):
    mx = mx.transpose()
    my = my.transpose()

    plt.figure(3)
    plt.plot(times, mx, 'o', color='tab:blue', label='x-position')
    plt.plot(times, my, 'o', color='tab:brown', label='y-position')
    plt.xlabel('time (s)')
    plt.ylabel('mean position (m)')
    plt.title('Mean Particle Position')

    # plot best fit line
    ax, bx = np.polyfit(times, mx, 1)
    ay, by = np.polyfit(times, my, 1)

    plt.plot(times, ax * times + bx, '--', color='tab:orange', label=f'slope = {np.round(ax, 4)}')
    plt.plot(times, ay * times + by, '--', color='tab:pink', label=f'slope = {np.round(bx, 4)}')
    plt.legend()


def plot_particle_movement(numTimeSteps, x_positions, y_positions, x0, y0):
    # plot the walk
    n_colors = numTimeSteps
    colors = plt.cm.tab20b(np.linspace(0, 1, n_colors))
    for color, i in zip(colors, np.arange(0, numTimeSteps)):
        plt.figure(2, figsize=(11, 4.5))
        plt.plot(x_positions[i, :], y_positions[i, :], '.', color=color)
    norm = mpl.colors.Normalize(vmin=0, vmax=numTimeSteps)
    s_m = plt.cm.ScalarMappable(cmap='tab20b', norm=norm)
    s_m.set_array([])
    plt.colorbar(s_m, label='time (s)')
    plot_initial_distribution(x0, y0)
    plt.title(f'Particle movement during {numTimeSteps} seconds')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()


def main():
    # set your velocity component values and dispersivity values here:
    xvel = 1
    yvel = 0
    alphaL = 1
    alphaT = 1

    # initialize other variables
    dt = 1  # time step in seconds
    numTimeSteps = 200  # number of time steps
    numPoints = 1000  # number of particles
    origin = [0, 0]  # origin location of particles
    x0 = origin[0] * np.ones((1, numPoints))  # x initial location of all particles
    y0 = origin[1] * np.ones((1, numPoints))  # y initial location of all particles

    # u and v velocity components (advection)
    u = xvel * np.ones((1, numPoints))
    v = yvel * np.ones((1, numPoints))
    speed = calc_speed(u, v)

    # Displacements
    dx = u * dt
    dy = v * dt

    # longitudinal and transverse dispersivities
    alphaL = alphaL * np.ones((1, numPoints))
    alphaT = alphaT * np.ones((1, numPoints))

    # diffusion coefficient
    diffCoef = 0

    # initialize everything for the random walk
    x_positions = np.zeros((numTimeSteps, numPoints))
    y_positions = np.zeros((numTimeSteps, numPoints))
    means_x = np.zeros((1, numTimeSteps))
    variances_x = np.zeros((1, numTimeSteps))
    means_y = np.zeros((1, numTimeSteps))
    variances_y = np.zeros((1, numTimeSteps))
    times = np.linspace(0, numTimeSteps, numTimeSteps)

    x_positions[0, :] = x0
    y_positions[0, :] = y0
    x = x0
    y = y0

    # run random walk
    for t in np.arange(1, numTimeSteps):
        x_new, y_new = random_walk(alphaL, alphaT, diffCoef, x, y, dx, dy, speed, dt, numPoints)
        x_positions[t, :] = x_new
        y_positions[t, :] = y_new

        # calculate statistics
        means_x[0, t] = np.mean(x_new)
        variances_x[0, t] = np.var(x_new)
        means_y[0, t] = np.mean(y_new)
        variances_y[0, t] = np.var(y_new)

        # set new position
        x = x_new
        y = y_new

    # plot the walk
    plot_particle_movement(numTimeSteps, x_positions, y_positions, x0, y0)

    # plot the stats
    plot_means(times, means_x, means_y)
    plot_variances(times, variances_x, variances_y)

    plt.show()


if __name__ == '__main__':
    main()
