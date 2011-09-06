
import _trw
from _trw import MRFEnergy, TypeBinary, TypeGeneral

if __name__ == '__main__':
    import numpy as np
    
    e = MRFEnergy[TypeBinary](TypeBinary.GlobalSize())
    nodeid0 = e.add_node(TypeBinary.LocalSize(), TypeBinary.NodeData(2,5))
    nodeid1 = e.add_node(TypeBinary.LocalSize(), TypeBinary.NodeData(10,1))
    e.add_edge(nodeid0, nodeid1, TypeBinary.EdgeData(0, 3, 3, 0))
    e.minimize_bp(printiter=1, printminiter=0, itermax=5)
    print e.get_solution(np.array([nodeid0, nodeid1]))
    # D = np.zeros((2,2))
    # D[0,0] = 2
    # D[1,0] = 2
    # D[0,1] = 2
    # D[1,1] = 2
    # ed = TypeBinary.EdgeData(3, 3, 2, 3);
    # e = MRFEnergy[TypeBinary](TypeBinary.GlobalSize())
    # nodeids = e.add_grid_nodes(D)
    # e.add_grid_edges(nodeids, ed)
    # e.minimize_bp(printiter=1, printminiter=0, itermax=10)
    # labels = e.get_solution(nodeids)
    # print labels
