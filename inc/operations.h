#pragma once

#include <algorithm>
#include <functional>


#ifdef __CUDA_ARCH__
#pragma hd_warning_disable
#define SPECIFIER __device__ __host__
#else
#define SPECIFIER 
#endif

namespace smalls
{
namespace detail
{

template<typename> struct VectorTraits;

template<template<int, typename> class VectorT, int SizeT, typename TypeT>
struct VectorTraits<VectorT<SizeT, TypeT>>
{
    static const int Size = SizeT;
    using ValueType = TypeT;

    template<int NewSizeT, typename NewTypeT>
    using ContainerT = VectorT<NewSizeT, NewTypeT>;
};

template<typename ScalarT>
struct VectorTraits
{
    static const int Size = 1;
    using ValueType = ScalarT;
};

template<int N, typename... Ts> using NthTypeOf =
        typename std::tuple_element<N, std::tuple<Ts...>>::type;

// An operation(vector1, vector2) needs to access the data in the vector via vector{1,2}.data. 
template<template<typename> class OperationT,
    typename OutputValueTypeT, size_t I, typename InputT1, typename InputT2>
SPECIFIER inline typename std::enable_if<
    !std::is_arithmetic<typename std::decay<InputT2>::type>::value, OutputValueTypeT>::type
applyOperation(InputT1&& input1, InputT2&& input2)
{
    using InputValueTypeT = typename VectorTraits<typename std::decay<InputT1>::type>::ValueType;
    return OperationT<InputValueTypeT>()(input1.data[I], input2.data[I]);
}

// An operation(vector1, scalar) needs to access the data in the vector via vector1.data and scalar. 
template<template<typename> class OperationT,
    typename OutputValueTypeT, size_t I, typename InputT1, typename InputT2>
SPECIFIER inline typename std::enable_if<
    std::is_arithmetic<typename std::decay<InputT2>::type>::value, OutputValueTypeT>::type
applyOperation(InputT1&& input1, InputT2&& input2)
{
    using InputValueTypeT = typename VectorTraits<typename std::decay<InputT1>::type>::ValueType;
    return OperationT<InputValueTypeT>()(input1.data[I], input2);
}

// An operation(vector) needs to access the data in the vector via vector.data. 
template<template<typename> class OperationT, typename OutputValueTypeT, size_t I, typename InputT>
SPECIFIER inline typename std::enable_if<
    !std::is_arithmetic<typename std::decay<InputT>::type>::value, OutputValueTypeT>::type
applyOperation(InputT&& input)
{
    using InputValueTypeT = typename VectorTraits<typename std::decay<InputT>::type>::ValueType;
    return OperationT<InputValueTypeT>()(input.data[I]);
}

// An operation(scalar) needs to access the data in the vector via scalar directly. 
template<template<typename> class OperationT, typename OutputValueTypeT, size_t I, typename InputT>
SPECIFIER inline typename std::enable_if<
    std::is_arithmetic<typename std::decay<InputT>::type>::value, OutputValueTypeT>::type
applyOperation(InputT&& input)
{
    using InputValueTypeT = typename VectorTraits<typename std::decay<InputT>::type>::ValueType;
    return OperationT<InputValueTypeT>()(input);
}

// This method unwraps the operation on the last element of the vector. 
template<
    template<typename> class OperationT, size_t I = 0, int SizeT,
    typename = typename std::enable_if<I == SizeT-1, void>::type, 
    typename OutputT, typename ...Inputs, int = 0>
SPECIFIER inline void unwrapOperation(OutputT& res, Inputs&&... inputs)
{
    using OutputValueTypeT = typename VectorTraits<typename std::decay<OutputT>::type>::ValueType;
    res.data[I] = applyOperation<OperationT, OutputValueTypeT, I>(std::forward<Inputs>(inputs)...);
}

// This method unwraps the operation on all elements of the vector except the last one. 
template<
    template<typename> class OperationT, size_t I = 0, int SizeT,
    typename = typename std::enable_if<I != SizeT-1, void>::type,
    typename OutputT, typename ...Inputs>
SPECIFIER inline void unwrapOperation(OutputT& res, Inputs&&... inputs)
{
    using OutputValueTypeT = typename VectorTraits<typename std::decay<OutputT>::type>::ValueType;
    res.data[I] = applyOperation<OperationT, OutputValueTypeT, I>(std::forward<Inputs>(inputs)...);
    unwrapOperation<OperationT, I + 1, SizeT>(res, std::forward<Inputs>(inputs)...);
}

template<template<typename> class OperationT, typename OutputT, typename ...Inputs>
SPECIFIER inline OutputT setupOperation(Inputs&&... inputs)
{
    enum{ SizeT = VectorTraits<typename std::decay<NthTypeOf<0, Inputs...>>::type>::Size };
    OutputT res;
    unwrapOperation<OperationT, 0, SizeT>(res, std::forward<Inputs>(inputs)...);
    return res;
}

template<template<typename> class OperationT, typename FirstT, typename ...Inputs>
SPECIFIER inline void setupAssignment(FirstT& first, Inputs&&... inputs)
{
    enum{ SizeT = VectorTraits<typename std::decay<FirstT>::type>::Size };
    unwrapOperation<OperationT, 0, SizeT>(first, first, std::forward<Inputs>(inputs)...);
}
}

#define BINARY_OPERATOR(operator, operation)\
template<typename FirstT, typename SecondT,\
    typename ElementT = typename detail::VectorTraits<typename std::decay<FirstT>::type>::ValueType,\
    size_t InputSize = detail::VectorTraits<typename std::decay<FirstT>::type>::Size,\
    typename ReturnT = typename std::result_of<operation<ElementT>(ElementT, ElementT)>::type,\
    typename OutputT = typename detail::VectorTraits<typename std::decay<FirstT>::type>::\
        template ContainerT<InputSize, ReturnT>>\
SPECIFIER inline OutputT operator(FirstT&& first, SecondT&& second)\
{\
    return detail::setupOperation<operation, OutputT, FirstT, SecondT>(\
        std::forward<FirstT>(first), std::forward<SecondT>(second));\
}

#define UNARY_OPERATOR(operator, operation)\
template<typename FirstT,\
    typename ElementT = typename detail::VectorTraits<typename std::decay<FirstT>::type>::ValueType,\
    size_t InputSize = detail::VectorTraits<typename std::decay<FirstT>::type>::Size,\
    typename ReturnT = typename std::result_of<operation<ElementT>(ElementT)>::type,\
    typename OutputT = typename detail::VectorTraits<typename std::decay<FirstT>::type>::\
        template ContainerT<InputSize, ReturnT>>\
SPECIFIER inline OutputT operator(FirstT&& first)\
{\
    return detail::setupOperation<operation, OutputT, FirstT>(std::forward<FirstT>(first));\
}

#define BINARY_ASSIGNMENT(operator, operation)\
template<typename FirstT, typename SecondT>\
SPECIFIER inline void operator(FirstT& first, SecondT&& second)\
{\
    return detail::setupAssignment<operation, FirstT, SecondT>(first, std::forward<SecondT>(second));\
}

namespace detail
{
BINARY_OPERATOR(operator+, std::plus);
BINARY_OPERATOR(operator-, std::minus);
BINARY_OPERATOR(operator*, std::multiplies);
BINARY_OPERATOR(operator/, std::divides);
BINARY_OPERATOR(operator==, std::equal_to);
BINARY_OPERATOR(operator!=, std::not_equal_to);
BINARY_OPERATOR(operator>, std::greater);
BINARY_OPERATOR(operator<, std::less);
BINARY_OPERATOR(operator>=, std::greater_equal);
BINARY_OPERATOR(operator<=, std::less_equal);
BINARY_ASSIGNMENT(operator+=, std::plus);
BINARY_ASSIGNMENT(operator-=, std::minus);
BINARY_ASSIGNMENT(operator*=, std::multiplies);
BINARY_ASSIGNMENT(operator/=, std::divides);
UNARY_OPERATOR(operator-, std::negate);
}

// Below could go in a separate header
#define GENERATE_FUNCTOR(name, operation)\
template<typename T>\
class name\
{\
public:\
    template<typename ...Args>\
    SPECIFIER inline T operator()(Args&&... args) const\
        {return operation<T>(std::forward<Args>(args)...);}\
};

#define GENERATE_FUNCTOR_NO_RETURN(name, operation)\
template<typename T>\
class name\
{\
public:\
    template<typename ...Args>\
    SPECIFIER inline T operator()(Args&&... args) const\
        {return operation(std::forward<Args>(args)...);}\
};

namespace detail
{
GENERATE_FUNCTOR(min, std::min);
GENERATE_FUNCTOR(max, std::max);
GENERATE_FUNCTOR_NO_RETURN(floor, std::floor);
GENERATE_FUNCTOR_NO_RETURN(ceil, std::ceil);
}

namespace math
{
BINARY_OPERATOR(min, detail::min);
BINARY_OPERATOR(max, detail::max);
UNARY_OPERATOR(floor, detail::floor);
UNARY_OPERATOR(ceil, detail::ceil);
}
}
