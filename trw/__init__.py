
import _trw
from _trw import MRFEnergy, TypeBinary

if __name__ == '__main__':
    import numpy as np
    
    D = np.zeros((2,2))
    D[0,0] = 2
    D[1,0] = 2
    D[0,1] = 2
    D[1,1] = 2
    ed = TypeBinary.EdgeData(3, 3, 2, 3);
    e = MRFEnergy(TypeBinary.GlobalSize())
    nodeids = e.add_grid_nodes(D)
    e.add_grid_edges(nodeids, ed)
    e.minimize_bp(printiter=1, printminiter=0, itermax=10)
    labels = e.get_solution(nodeids)
    print labels
