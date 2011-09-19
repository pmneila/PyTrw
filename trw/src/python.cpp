
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/slice.hpp>

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
py::object add_grid_nodes(MRFEnergy<T>& mrfenergy, const PyArrayObject* unaryterms,
                            int axis=0)
{
    typedef typename T::REAL REAL;
    typedef typename T::LocalSize LocalSize;
    typedef typename T::NodeData NodeData;
    
    int ndim = PyArray_NDIM(unaryterms);
    npy_intp* shape = PyArray_DIMS(unaryterms);
    if(-axis > ndim || axis >= ndim)
        throw std::runtime_error("axis is out of bounds");
    if(axis < 0)
        axis = ndim + axis;
    int num_labels = shape[axis];
    
    npy_intp* nodeids_shape = new npy_intp[ndim-1];
    // Copy the shape except for the axis given.
    std::copy(shape, shape+axis, nodeids_shape);
    std::copy(shape+axis+1, shape+ndim, nodeids_shape+axis);
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(
                                PyArray_SimpleNew(ndim-1, nodeids_shape, NPY_ULONG));
    
    std::vector<REAL> d(num_labels);
    pyarray_index unaryterms_idx(ndim);
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        const pyarray_index& nodeids_idx = it.getIndex();
        // Extract unary terms for this node.
        std::copy(nodeids_idx.idx, nodeids_idx.idx+axis, &unaryterms_idx[0]);
        std::copy(nodeids_idx.idx+axis, nodeids_idx.idx+nodeids_idx.ndim, &unaryterms_idx[axis+1]);
        // std::copy(nodeids_idx.idx, nodeids_idx.idx+nodeids_idx.ndim, &unaryterms_idx[1]);
        for(int j=0; j<num_labels; ++j)
        {
            unaryterms_idx[axis] = j;
            d[j] = PyArray_SafeGet<REAL>(unaryterms, unaryterms_idx);
        }
        
        // Create the node.
        typename MRFEnergy<T>::NodeId id = mrfenergy.AddNode(LocalSize(num_labels), NodeData(d));
        
        // Store the node.
        PyArray_SafeSet<numeric_pointer>(nodeids, nodeids_idx, reinterpret_cast<numeric_pointer>(id));
    }
    
    delete [] nodeids_shape;
    
    return py::object(nodeids);
}

template<class T>
void add_grid_edges(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids, const typename T::EdgeData& ed)
{
    typedef typename T::REAL REAL;
    typedef typename MRFEnergy<T>::NodeId NodeId;
    
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
            
            mrfenergy.AddEdge(id2, id1, ed);
        }
    }
}

template<class T>
void add_grid_edges_direction(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids,
                                const typename T::EdgeData& ed, int direction)
{
    typedef typename T::REAL REAL;
    typedef typename MRFEnergy<T>::NodeId NodeId;
    
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    
    if(direction >= ndim || direction < 0)
        throw std::runtime_error("invalid direction");
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        if(coord[direction] - 1 < 0)
            continue;
        
        NodeId id1 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));
        --coord[direction];
        NodeId id2 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));
        ++coord[direction];
        
        mrfenergy.AddEdge(id2, id1, ed);
    }
}

template<class T>
void add_grid_edges_direction_local(MRFEnergy<T>& mrfenergy, const PyArrayObject* nodeids,
                                const PyArrayObject* ed, int direction, int axis=0)
{
    typedef typename T::REAL REAL;
    typedef typename MRFEnergy<T>::NodeId NodeId;
    typedef typename MRFEnergy<T>::EdgeData EdgeData;
    
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    int ed_ndim = PyArray_NDIM(ed);
    npy_intp* ed_shape = PyArray_DIMS(ed);
    
    if(direction >= ndim || direction < 0)
        throw std::runtime_error("invalid direction");
    if(ed_ndim != ndim+1)
        throw std::runtime_error("invalid number of dimensions for the edge data array");
    
    // Check and fix the axis parameter.
    if(-axis > ed_ndim || axis >= ed_ndim)
        throw std::runtime_error("axis is out of bounds");
    if(axis < 0)
        axis = ed_ndim + axis;
    
    // Check the shapes.
    // First, take the axis dimension out of the ed_shape.
    npy_intp* ed_shape_fix = new npy_intp[ndim];
    std::copy(ed_shape, ed_shape+axis, ed_shape_fix);
    std::copy(ed_shape+axis+1, ed_shape+ed_ndim, ed_shape_fix+axis);
    if(std::mismatch(shape, shape+direction, ed_shape_fix, std::less_equal<int>()).first != shape+direction
            || std::mismatch(shape+direction+1, shape+ndim, ed_shape_fix+direction+1, std::less_equal<int>()).first != shape+ndim)
        throw std::runtime_error("invalid shape for the edge data array");
    if(ed_shape_fix[direction] < shape[direction]-1)
        throw std::runtime_error("invalid shape for the edge data array");
    delete [] ed_shape_fix;
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {        
        npy_intp* coord = it.getIndex();
        if(coord[direction] - 1 < 0)
            continue;
        
        NodeId id1 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));        
        --coord[direction];
        
        // Create a list to access the EdgeData.
        py::list lcoord;
        int d;
        for(d=0; d<ndim; ++d)
        {
            if(d == axis)
                lcoord.append(py::slice());
            lcoord.append(coord[d]);
        }
        if(d == axis)
            lcoord.append(py::slice());
        
        PyArrayObject* localed = reinterpret_cast<PyArrayObject*>(
                        PyObject_GetItem((PyObject*)ed, py::tuple(lcoord).ptr()));
        NodeId id2 = reinterpret_cast<NodeId>(PyArray_SafeGet<numeric_pointer>(nodeids, coord));
        ++coord[direction];
        
        mrfenergy.AddEdge(id2, id1, EdgeData(py::object(localed)));
    }
}

template<>
void add_grid_edges_direction_local<TypeBinary>(MRFEnergy<TypeBinary>& mrfenergy,
                                const PyArrayObject* nodeids,
                                const PyArrayObject* ed, int direction, int axis)
{
    throw std::runtime_error("not implemented");
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
        .def("add_node", (add_node_type)(&MRFEnergy<T>::AddNode),
            (py::arg("local_size"), py::arg("node_data")))
        .def("add_edge", (add_edge_type)(&MRFEnergy<T>::AddEdge),
            (py::arg("nodeid1"), py::arg("nodeid2"), py::arg("edge_data")))
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
        .def("add_grid_nodes", &add_grid_nodes<T>,
            (py::arg("unary_terms"), py::arg("axis")=0))
        .def("add_grid_edges", &add_grid_edges<T>,
            (py::arg("nodeids"), py::arg("edge_data")))
        .def("add_grid_edges_direction", &add_grid_edges_direction<T>,
            (py::arg("nodeis"), py::arg("edge_data"), py::arg("direction")))
        .def("add_grid_edges_direction_local", &add_grid_edges_direction_local<T>,
            (py::arg("nodeis"), py::arg("edge_data"), py::arg("direction"), py::arg("axis")=0))
        .def("get_solution", &get_solution<T>,
            (py::arg("nodeids")));
    
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
        return (PyObject*)&obj;
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
    add_mrfenergy_class<TypeBinary>(cls_dict, py::scope().attr("TypeBinary"), "TypeBinary");
    add_mrfenergy_class<TypeGeneral>(cls_dict, py::scope().attr("TypeGeneral"), "TypeGeneral");
    py::scope().attr("MRFEnergy") = cls_dict;
}
