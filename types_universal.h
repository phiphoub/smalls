#pragma once

#include <type_traits>
#include <functional>
#include <algorithm>

#ifdef __CUDA_ARCH__
#define FOOBAR __device__ __host__
#else
#define FOOBAR 
#endif

namespace tmpFoo
{

namespace internal
{

template<typename> struct VectorTraits;

template<template<int, typename> class Vector, int SizeT, typename TypeT>
struct VectorTraits<Vector<SizeT, TypeT>>{
    static const int Size = SizeT;
    using ValueType = TypeT;
};

template<size_t I, typename ContainerT, typename ThisArg>
FOOBAR inline void setData(ContainerT& container, ThisArg&& arg)
{
    using TypeT = typename VectorTraits<ContainerT>::ValueType;
    container.data[I] = static_cast<TypeT>(std::forward<ThisArg>(arg));
}

template<size_t I, typename ContainerT, typename ThisArg, typename... Args>
FOOBAR inline void setData(ContainerT& container, ThisArg&& thisArg, Args&&... args)
{
    using TypeT = typename VectorTraits<ContainerT>::ValueType;
    container.data[I] = static_cast<TypeT>(std::forward<ThisArg>(thisArg));
    setData<I + 1>(container, std::forward<Args>(args)...);
}

template<bool B, typename T = void> using disable_if = std::enable_if<!B, T>;
}

template<int SizeT, typename ScalarT>
class Vector
{
public:
    typedef Vector<SizeT, ScalarT> Type;

    FOOBAR Vector() = default;

    template<typename Arg, typename ...Args, typename = typename
        internal::disable_if<
        sizeof...(Args) == 0 &&
        std::is_same<typename std::remove_reference<Arg>::type, Vector>::value>::type>
        FOOBAR explicit Vector(Arg&& arg, Args&&... args)
    {
        static_assert(sizeof...(Args) == SizeT - 1,
            "The number of arguments in the constructor must be equal to the size of the vector");
        internal::setData<0>(*this, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    ScalarT data[SizeT];
};

// Makers
template<typename ScalarT, typename ...Args>
FOOBAR Vector<sizeof...(Args), ScalarT> make_vector(Args&&... args)
{
    return Vector<sizeof...(Args), ScalarT>(std::forward<Args>(args)...);
}

namespace internal
{

template<
    template<typename> class BinaryOperationT,
    size_t I = 0, int SizeT, typename InputT1, typename InputT2, typename OutputT,
    typename = typename std::enable_if<I == SizeT-1, void>::type, int = 0>
    FOOBAR inline void binaryOperation(InputT1&& lhs, InputT2&& rhs,
    OutputT& res)
{
    using ValueType = typename internal::VectorTraits<typename std::decay<InputT1>::type>::ValueType;
    res.data[I] = BinaryOperationT<ValueType>() (lhs.data[I], rhs.data[I]);
}

template<
    template<typename> class BinaryOperationT,
    size_t I = 0, int SizeT, typename InputT1, typename InputT2, typename OutputT,
    typename = typename std::enable_if<I != SizeT-1, void>::type>
    FOOBAR inline void binaryOperation(InputT1&& lhs, InputT2&& rhs,
    OutputT& res)
{
    using ValueType = typename internal::VectorTraits<typename std::decay<InputT1>::type>::ValueType;
    res.data[I] = BinaryOperationT<ValueType>()(lhs.data[I], rhs.data[I]);
    binaryOperation<BinaryOperationT, I + 1, SizeT, InputT1, InputT2>(
        std::forward<InputT1>(lhs),
        std::forward<InputT2>(rhs),
        res);
}

template<template<typename> class BinaryOperationT, typename OutputT, typename InputT1, typename InputT2>
FOOBAR inline OutputT binaryOperation(InputT1&& lhs, InputT2&& rhs)
{
    using ValueType = typename internal::VectorTraits<typename std::decay<InputT1>::type>::ValueType;
    enum{ SizeT = internal::VectorTraits<typename std::decay<InputT1>::type>::Size };

    static_assert(std::is_same<Vector<SizeT, ValueType>, typename std::decay<InputT1>::type>::value,
        "The arthmetic operation must operate on a Vector type");

    OutputT res;
    binaryOperation<BinaryOperationT, 0, SizeT, InputT1, InputT2, OutputT>(
        std::forward<InputT1>(lhs), std::forward<InputT2>(rhs), res);
    return res;
}
}

#define GENERATE_BINARY_OPERATOR(operator, operation)\
template<typename InputT1, typename InputT2>\
FOOBAR inline typename std::decay<InputT1>::type operator(InputT1&& lhs, InputT2&& rhs)\
{\
    static_assert(std::is_same<typename std::decay<InputT1>::type, \
        typename std::decay<InputT2>::type>::value, \
    "Decayed types for arithmetic operation must be the same for lhs and rhs");\
    return internal::binaryOperation<operation, typename std::decay<InputT1>::type, InputT1, InputT2>(\
        std::forward<InputT1>(lhs), std::forward<InputT2>(rhs));\
}

#define GENERATE_COMPARISON_OPERATOR(operator, operation)\
template<typename InputT1, typename InputT2>\
FOOBAR inline Vector<internal::VectorTraits<typename std::decay<InputT1>::type>::Size, bool> \
operator(InputT1&& lhs, InputT2&& rhs)\
{\
    static_assert(std::is_same<typename std::decay<InputT1>::type, \
        typename std::decay<InputT2>::type>::value, \
    "Decayed types for arithmetic operation must be the same for lhs and rhs");\
    enum{ SizeT = internal::VectorTraits<typename std::decay<InputT1>::type>::Size };\
    return internal::binaryOperation<operation, Vector<SizeT, bool>, InputT1>(\
        std::forward<InputT1>(lhs), std::forward<InputT2>(rhs));\
}

#define GENERATE_FUNCTOR(name, operation)\
template<typename T>\
class name\
{\
public:\
    FOOBAR inline T operator()(const T& lhs, const T& rhs) const {return operation<T>(lhs, rhs);}\
};

namespace internal
{
GENERATE_FUNCTOR(min, std::min);
GENERATE_FUNCTOR(max, std::max);
}

GENERATE_BINARY_OPERATOR(operator+,             std::plus);
GENERATE_BINARY_OPERATOR(operator-,             std::minus);
GENERATE_BINARY_OPERATOR(operator*,             std::multiplies);
GENERATE_BINARY_OPERATOR(operator/,             std::divides);
GENERATE_BINARY_OPERATOR(min,                   internal::min);
GENERATE_BINARY_OPERATOR(max,                   internal::max);
GENERATE_COMPARISON_OPERATOR(operator==,        std::equal_to);
GENERATE_COMPARISON_OPERATOR(operator!=,        std::not_equal_to);
GENERATE_COMPARISON_OPERATOR(operator>,         std::greater);
GENERATE_COMPARISON_OPERATOR(operator<,         std::less);
GENERATE_COMPARISON_OPERATOR(operator>=,        std::greater_equal);
GENERATE_COMPARISON_OPERATOR(operator<=,        std::less_equal);

}
