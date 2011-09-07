
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include "core/MRFEnergy.h"

#define EXTMODULE_IMPORT_ARRAY
#include "pyarraymodule.h"

#include "pyarray_index.h"

#include <iostream>
#include <vector>

namespace py = boost::python;

typedef unsigned long numeric_pointer;

// Some convenience functions for easy and fast access from Python.

TypeGeneral::NodeData::NodeData(const py::object& iter)
{
    typedef TypeGeneral::REAL REAL;
    py::stl_input_iterator<REAL> it(iter), end;
    bool haslen = PyObject_HasAttrString(iter.ptr(), "__len__");
    
    if(haslen)
    {
        m_data.resize(py::len(iter));
        std::copy(it, end, m_data.begin());
    }
    else
        for(; it!=end; ++it)
        {
           m_data.push_back(*it);
           std::cout << *it << " ";
        }        
}

TypeGeneral::EdgeData::EdgeData(const py::object& obj)
{
    typedef TypeGeneral::REAL REAL;
    bool isiterable = PyObject_HasAttrString(obj.ptr(), "__iter__");
    
    if(isiterable)
    {
        py::stl_input_iterator<REAL> it(obj), end;
        bool haslen = PyObject_HasAttrString(obj.ptr(), "__len__");
        
        m_type = GENERAL;
        
        if(haslen)
        {
            m_dataGeneral.resize(py::len(obj));
            std::copy(it, end, m_dataGeneral.begin());
        }
        else
        {
            for(; it!=end; ++it)
            {
               m_dataGeneral.push_back(*it);
               std::cout << *it << " ";
            }
        }
    }
    else
    {
        m_type = POTTS;
        m_lambdaPotts = py::extract<REAL>(obj);
    }
}

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
py::object add_grid_nodes(MRFEnergy<T>& mrfenergy, const PyArrayObject* unaryterms)
{
    throw std::runtime_error("not implemented error");
}

template<>
py::object add_grid_nodes<TypeBinary>(MRFEnergy<TypeBinary>& mrfenergy, const PyArrayObject* unaryterms)
{
    typedef TypeBinary::REAL REAL;
    
    int ndim = PyArray_NDIM(unaryterms);
    npy_intp* shape = PyArray_DIMS(unaryterms);
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(
                                PyArray_SimpleNew(ndim-1, &shape[1], NPY_ULONG));
    
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
        PyArray_SafeSet<numeric_pointer>(nodeids, nodeids_idx, reinterpret_cast<numeric_pointer>(id));
    }
    
    return py::object(nodeids);
}

template<class T>
void add_grid_edges(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids, const typename T::EdgeData& ed)
{
    throw std::runtime_error("not implemented error");
}

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
        NodeId id1 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));
        
        for(int d=0; d < ndim; ++d)
        {
            if(coord[d] - 1 < 0)
                continue;
            
            --coord[d];
            NodeId id2 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));
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
        NodeId id = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, it.getIndex()));
        Label label = mrfenergy.GetSolution(id);
        PyArray_SafeSet(solution, it.getIndex(), label);
    }
    
    return py::object(solution);
}

// Python wrapper.

template<class T>
void add_mrfenergy_class(py::dict cls_dict, py::object key, const std::string& suffix)
{
    typedef typename T::LocalSize LocalSize;
    typedef typename T::NodeData NodeData;
    typedef typename T::EdgeData EdgeData;
    
    // NodeId is a pointer to a private struct Node. These lines
    // mask NodeId as an integer value (numeric_pointer).
    typedef numeric_pointer (MRFEnergy<T>::*add_node_type)(LocalSize, NodeData);
    typedef void (MRFEnergy<T>::*add_edge_type)(numeric_pointer, numeric_pointer, EdgeData);
    
    // Export the MRFEnergy class for the given T.
    py::object c = py::class_<MRFEnergy<T> >(("MRFEnergy"+suffix).c_str(), "MRF energy representation.",
            py::init<typename T::GlobalSize>())
        .def("add_node", (add_node_type)(&MRFEnergy<T>::AddNode))
        .def("add_edge", (add_edge_type)(&MRFEnergy<T>::AddEdge))
        .def("minimize_trw", &minimize_trw<T>,
            "Minimize the energy of the MRF using TRW.",
            (py::arg("self"), py::arg("eps")=-1, py::arg("itermax")=100,
                py::arg("printiter")=5, py::arg("printminiter")=10))
        .def("minimize_bp", &minimize_bp<T>,
            "Minimize the energy of the MRF using BP.",
            (py::arg("self"), py::arg("eps")=-1, py::arg("itermax")=100,
                py::arg("printiter")=5, py::arg("printminiter")=10))
        .def("zero_messages", &MRFEnergy<T>::ZeroMessages)
        .def("set_automatic_ordering", &MRFEnergy<T>::SetAutomaticOrdering)
        .def("add_grid_nodes", &add_grid_nodes<T>)
        .def("add_grid_edges", &add_grid_edges<T>)
        .def("get_solution", &get_solution<T>);
    
    cls_dict.attr("__setitem__")(key, c);
}

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

template<class T>
struct nodeid_to_int
{
    typedef typename MRFEnergy<T>::NodeId NodeId;
    static PyObject* convert(const NodeId& obj)
    {
        return PyInt_FromLong(reinterpret_cast<numeric_pointer>(obj));
    }
};

BOOST_PYTHON_MODULE(_trw)
{
    import_array();
    
    // Automatic conversion from Python ndarray to C PyArrayObject.
    py::converter::registry::insert(&extract_pyarray, py::type_id<PyArrayObject>());
    py::to_python_converter<PyArrayObject, PyArrayObject_to_python>();
    
    // Include specific types.
    py::class_<TypeBinary>("TypeBinary")
        // FIXME: Is there a better way to do this?
        .setattr("NodeData",
                        py::class_<TypeBinary::NodeData>("TypeBinary.NodeData",
                            py::init<TypeBinary::REAL,TypeBinary::REAL>()))
        .setattr("EdgeData",
                        py::class_<TypeBinary::EdgeData>("TypeBinary.EdgeData",
                            py::init<TypeBinary::REAL,TypeBinary::REAL,TypeBinary::REAL,TypeBinary::REAL>()))
        .setattr("GlobalSize",
                        py::class_<TypeBinary::GlobalSize>("TypeBinary.GlobalSize"))
        .setattr("LocalSize",
                        py::class_<TypeBinary::LocalSize>("TypeBinary.LocalSize"));
    
    py::class_<TypeGeneral>("TypeGeneral")
        .setattr("NodeData",
                        py::class_<TypeGeneral::NodeData>("TypeGeneral.NodeData",
                            py::init<py::object>()))
        .setattr("EdgeData", 
                        py::class_<TypeGeneral::EdgeData>("TypeGeneral.EdgeData",
                            py::init<py::object>()))
        .setattr("GlobalSize",
                        py::class_<TypeGeneral::GlobalSize>("TypeGeneral.GlobalSize"))
        .setattr("LocalSize",
                        py::class_<TypeGeneral::LocalSize>("TypeGeneral.LocalSize",
                            py::init<int>()));
    
    // Include MRFEnergy specializations.
    py::dict cls_dict = py::dict();
    // For TypeBinary.
    add_mrfenergy_class<TypeBinary>(cls_dict, py::scope().attr("TypeBinary"), "TypeBinary");
    add_mrfenergy_class<TypeGeneral>(cls_dict, py::scope().attr("TypeGeneral"), "TypeGeneral");
    py::scope().attr("MRFEnergy") = cls_dict;
}
