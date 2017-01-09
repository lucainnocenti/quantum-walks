import numpy as np
import qutip


def coin_op(theta, xi):
    if type(theta) == np.ndarray:
        theta = theta[0]
    if type(xi) == np.ndarray:
        xi = xi[0]
    matrix = np.zeros((2, 2), dtype=np.complex128)
    matrix[0, 0] = np.cos(theta) * np.exp(1.j * xi)
    matrix[0, 1] = np.sin(theta)
    matrix[1, 0] = -np.sin(theta)
    matrix[1, 1] = np.cos(theta) * np.exp(-1.j * xi)
    return qutip.Qobj(matrix)


def step_op(n_steps, theta, xi):
    space_dims = n_steps + 1
    shift_left = qutip.qeye(space_dims)

    shift_right = np.zeros((space_dims, space_dims))
    for idx in range(len(shift_right) - 1):
        shift_right[idx + 1, idx] = 1.
    shift_right = qutip.Qobj(shift_right)

    proj_uu = qutip.Qobj([[1., 0.], [0., 0.]])
    proj_dd = qutip.Qobj([[0., 0.], [0., 1.]])

    c_shift = (qutip.tensor(shift_left, proj_uu) +
               qutip.tensor(shift_right, proj_dd))

    big_coin_op = qutip.tensor(
        qutip.qeye(space_dims),
        coin_op(theta, xi)
    )

    return c_shift * big_coin_op


def many_steps_evolution(n_steps, parameters):
    dims = n_steps + 1
    parameters = np.asarray(parameters)
    evolution = step_op(n_steps, *parameters[0])
    for par in parameters[1:]:
        evolution = step_op(n_steps, *par) * evolution
    return evolution


class UnreachableState(Exception):
    pass


def two_steps_condition(amps):
    if isinstance(amps, qutip.Qobj):
        amps = amps.data.toarray()
    amps = np.asarray(amps).reshape((amps.shape[0]))
    if not amps.shape[0] % 2 == 0:
        raise ValueError('The number of elements in `amps` must be even.')
    v1 = amps[0] / amps[3]
    v2 = amps[-1] / amps[-4]
    reachable_condition = v1 + np.conj(v2)
    return np.allclose(reachable_condition, [0.])


def devolve_1step(state, theta, xi):
    if isinstance(state, qutip.Qobj):
        amps = state.data.toarray()
    else:
        amps = state
    amps = amps.reshape((amps.shape[0]))
    n_sites = amps.shape[0] // 2
    outamps = np.zeros(amps.shape[0] - 2, dtype=amps.dtype)
    c11 = np.cos(theta) * np.exp(-1j * xi)
    c11 = c11[0]
    c12 = -np.sin(theta)[0]
    c21 = np.sin(theta)[0]
    c22 = np.cos(theta) * np.exp(1j * xi)
    c22 = c22[0]

    for site in range(n_sites - 1):
        # if first step..
        if site == 0:
            outamps[0] = c11 * amps[0] + c12 * amps[3]
        # if last step..
        elif site == outamps.shape[0] // 2 - 2:
            outamps[2 * site + 1] = (c21 * amps[2 * site] +
                                     c22 * amps[2 * site + 3])
        else:
            outamps[2 * site] = (c11 * amps[2 * site] +
                                 c12 * amps[2 * site + 3])
            outamps[2 * site + 1] = (c21 * amps[2 * site] +
                                     c22 * amps[2 * site + 3])

    if isinstance(state, qutip.Qobj):
        outamps = qutip.Qobj(outamps.reshape((outamps.shape[0], 1)))
        outamps.dims = [[n_sites - 1, 2], [1, 1]]

    return outamps


def compute_parameters(state):
    if isinstance(state, qutip.Qobj):
        state = state.data.toarray()

    if state.shape[0] == 4:
        theta = np.arctan(np.abs(state[3] / state[0]))
        xi = -np.angle(-state[3] / state[0])
    else:
        theta = np.arctan(np.abs(state[-4] / state[-1]))
        xi = np.angle(state[-4] / state[-1])
    return theta, xi


def backtrace_qw(state, initial_coin=None):
    if initial_coin is None:
        initial_coin = qutip.basis(2, 0)

    amps = state.data.toarray()

    n_steps = amps.shape[0] // 2 - 1
    pars = np.zeros((n_steps - 1, 2))
    if n_steps < 1:
        raise ValueError('The state vector must have length at least 4.')
    for step in range(n_steps):
        # if last step..
        if step == n_steps - 1:
            pass
        else:
            if not two_steps_condition(amps):
                raise UnreachableState('Reachability test '
                                       'failed at step {}'.format(step))
            pars[step] = compute_parameters(amps)
            amps = devolve_1step(amps, *pars[step])

