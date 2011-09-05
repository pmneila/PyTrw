
#include <boost/python.hpp>

#include "core/MRFEnergy.h"

#define EXTMODULE_IMPORT_ARRAY
#include "pyarraymodule.h"

#include "pyarray_index.h"

namespace py = boost::python;

// Some convenience functions for easy access from Python.

template<class T>
py::object minimize_trw(MRFEnergy<T>& mrfenergy,
    typename T::REAL eps=-1, int itermax=1000000,
    int printiter=5, int printminiter=10)
{
    typename MRFEnergy<T>::Options opt;
    opt.m_eps = eps;
    opt.m_iterMax = itermax;
    opt.m_printIter = printiter;
    opt.m_printMinIter = printminiter;
    
    typename T::REAL lowerBound;
    typename T::REAL energy;
    int num_iters = mrfenergy.Minimize_TRW_S(opt, lowerBound, energy);
    
    return py::make_tuple(num_iters, energy, lowerBound);
}

template<class T>
py::object minimize_bp(MRFEnergy<T>& mrfenergy,
    typename T::REAL eps=-1, int itermax=1000000,
    int printiter=5, int printminiter=10)
{
    typename MRFEnergy<T>::Options opt;
    opt.m_eps = eps;
    opt.m_iterMax = itermax;
    opt.m_printIter = printiter;
    opt.m_printMinIter = printminiter;
    
    typename T::REAL energy;
    int num_iters = mrfenergy.Minimize_BP(opt, energy);
    
    return py::make_tuple(num_iters, energy);
}

template<class T>
py::object add_grid_nodes(MRFEnergy<T>& mrfenergy, const PyArrayObject* unaryterms);

template<>
py::object add_grid_nodes<TypeBinary>(MRFEnergy<TypeBinary>& mrfenergy, const PyArrayObject* unaryterms)
{
    typedef TypeBinary::REAL REAL;
    
    int ndim = PyArray_NDIM(unaryterms);
    npy_intp* shape = PyArray_DIMS(unaryterms);
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(
                                PyArray_SimpleNew(ndim-1, &shape[1], NPY_INT));
    
    if(shape[0] != 2)
        throw std::runtime_error("add_grid_node only supports binary unary terms");
    
    pyarray_index unaryterms_idx(ndim);
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        const pyarray_index& nodeids_idx = it.getIndex();
        // Extract unary terms for this node.
        std::copy(nodeids_idx.idx, nodeids_idx.idx+ndim, &unaryterms_idx[1]);
        
        unaryterms_idx[0] = 0;
        REAL d0 = PyArray_SafeGet<REAL>(unaryterms, unaryterms_idx);
        unaryterms_idx[0] = 1;
        REAL d1 = PyArray_SafeGet<REAL>(unaryterms, unaryterms_idx);
        
        // Create the node.
        MRFEnergy<TypeBinary>::NodeId id = mrfenergy.AddNode(TypeBinary::LocalSize(),
                                                                TypeBinary::NodeData(d0, d1));
        
        // Store the node.
        PyArray_SafeSet<int>(nodeids, nodeids_idx, reinterpret_cast<int>(id));
    }
    
    return py::object(nodeids);
}

template<class T>
void add_grid_edges(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids, const typename T::EdgeData& ed);

template<>
void add_grid_edges<TypeBinary>(MRFEnergy<TypeBinary>& mrfenergy,
            const PyArrayObject* nodeids, const TypeBinary::EdgeData& ed)
{
    typedef TypeBinary::REAL REAL;
    typedef MRFEnergy<TypeBinary>::NodeId NodeId;
    
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        NodeId id1 = reinterpret_cast<NodeId>(PyArray_SafeGet<int>(nodeids, coord));
        
        for(int d=0; d < ndim; ++d)
        {
            if(coord[d] - 1 < 0)
                continue;
            
            --coord[d];
            NodeId id2 = reinterpret_cast<NodeId>(PyArray_SafeGet<int>(nodeids, coord));
            ++coord[d];
            
            mrfenergy.AddEdge(id1, id2, ed);
        }
    }
}

template<class T>
py::object get_solution(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids)
{
    typedef typename MRFEnergy<T>::NodeId NodeId;
    typedef typename T::Label Label;
    
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    PyArrayObject* solution = reinterpret_cast<PyArrayObject*>(
                                PyArray_SimpleNew(ndim, shape, (mpl::at<numpy_typemap,Label>::type::value)));
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        NodeId id = reinterpret_cast<NodeId>(PyArray_SafeGet<int>(nodeids, it.getIndex()));
        Label label = mrfenergy.GetSolution(id);
        PyArray_SafeSet(solution, it.getIndex(), label);
    }
    
    return py::object(solution);
}

// Python wrapper.

void* extract_pyarray(PyObject* x)
{
    return PyObject_TypeCheck(x, &PyArray_Type) ? x : 0;
}

struct PyArrayObject_to_python
{
    static PyObject* convert(const PyArrayObject& obj)
    {
        return py::incref((PyObject*)&obj);
    }
};

BOOST_PYTHON_MODULE(_trw)
{
    import_array();
    
    // Automatic conversion from Python ndarray to C PyArrayObject.
    py::converter::registry::insert(&extract_pyarray, py::type_id<PyArrayObject>());
    py::to_python_converter<PyArrayObject, PyArrayObject_to_python>();
    
    py::class_<TypeBinary>("TypeBinary")
        // FIXME: Is there a better way to do this?
        .setattr("EdgeData", py::class_<TypeBinary::EdgeData>("TypeBinary.EdgeData",
                        py::init<TypeBinary::REAL,TypeBinary::REAL,TypeBinary::REAL,TypeBinary::REAL>()))
        .setattr("GlobalSize", py::class_<TypeBinary::GlobalSize>("TypeBinary.GlobalSize"));
    
    py::class_<MRFEnergy<TypeBinary> >("MRFEnergy", py::init<TypeBinary::GlobalSize>())
        .def("minimize_trw", &minimize_trw<TypeBinary>,
            "Minimize the energy of the MRF using TRW.",
            (py::arg("self"), py::arg("eps")=-1, py::arg("itermax")=100,
                py::arg("printiter")=5, py::arg("printminiter")=10))
        .def("minimize_bp", &minimize_bp<TypeBinary>,
            "Minimize the energy of the MRF using BP.",
            (py::arg("self"), py::arg("eps")=-1, py::arg("itermax")=100,
                py::arg("printiter")=5, py::arg("printminiter")=10))
        .def("zero_messages", &MRFEnergy<TypeBinary>::ZeroMessages)
        .def("set_automatic_ordering", &MRFEnergy<TypeBinary>::SetAutomaticOrdering)
        .def("add_grid_nodes", &add_grid_nodes<TypeBinary>)
        .def("add_grid_edges", &add_grid_edges<TypeBinary>)
        .def("get_solution", &get_solution<TypeBinary>);
}
