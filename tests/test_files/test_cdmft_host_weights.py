# Test file
# Goal : to test that a host passed with set_CDMFT_host (CDMFT's host_function) is fitted with the grid's
# cdmft weights, like the built-in host. The host_function below computes the built-in host and hands it back
# through set_CDMFT_host, so both runs must converge to the same bath. The grid uses opt='ifreq': its cdmft
# weights are not uniform (with uniform weights the two fits agree even when the weights are lost).
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm import qcm
from pyqcm.cdmft import CDMFT, frequency_grid

#--------------------------------------------------------------------------------
# First, defining the model (a 2-cluster model for graphene with bath)

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('Graphene_2', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))

model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))
model.hopping_operator('M', (0, 0, 0), 1, orbitals=(1, 1), tau=0, sigma=3)
model.hopping_operator('M', (0, 0, 0), -1, orbitals=(2, 2), tau=0, sigma=3)
model.hopping_operator('Mx', (0, 0, 0), 1, orbitals=(1, 1), tau=0, sigma=1)
model.hopping_operator('Mx', (0, 0, 0), -1, orbitals=(2, 2), tau=0, sigma=1)
model.hopping_operator('cdw', (0, 0, 0), 1, orbitals=(1, 1), tau=0)
model.hopping_operator('cdw', (0, 0, 0), -1, orbitals=(2, 2), tau=0)

#--------------------------------------------------------------------------------


# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']
x0 = {v: model.parameters()[v] for v in varia}

grid = frequency_grid('regular', (10, 50, 10), opt='ifreq')
assert np.ptp(grid.cdmft_weight) > 0.1 * grid.cdmft_weight.max()  # the weights are far from uniform


def host_function(I):
    """an external host: here the built-in one, read back and passed through set_CDMFT_host"""
    qcm.CDMFT_host(grid.wr, grid.cdmft_weight, I.label)
    H = qcm.get_CDMFT_host(0, False, I.label)
    qcm.set_CDMFT_host(I.label, grid.wr, H, 0, False)


bath = {}
for name, hf in (('built-in host', None), ('host_function', host_function)):
    for v in varia:
        model.set_parameter(v, x0[v])
    X = CDMFT(model, varia=varia, grid=grid, host_function=hf, accur=1e-6, maxiter=64, method='trf', convergence='parameters')
    bath[name] = np.array([X.I.parameters()[v] for v in varia])
    print(name, bath[name])

diff = np.abs(bath['built-in host'] - bath['host_function']).max()
print(f'max bath difference: {diff:.2e}')
assert diff < 1e-5, 'the host passed through set_CDMFT_host was fitted with other weights'
