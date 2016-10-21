#pragma once

#include "inc/operations.h"
#include <array>

namespace smalls
{
namespace detail
{
template<size_t I, typename ContainerT, typename ThisArg>
SPECIFIER inline void setData(ContainerT& container, ThisArg&& arg)
{
    using TypeT = typename VectorTraits<ContainerT>::ValueType;
    container.data[I] = static_cast<TypeT>(std::forward<ThisArg>(arg));
}

template<size_t I, typename ContainerT, typename ThisArg, typename... Args>
SPECIFIER inline void setData(ContainerT& container, ThisArg&& thisArg, Args&&... args)
{
    using TypeT = typename VectorTraits<ContainerT>::ValueType;
    container.data[I] = static_cast<TypeT>(std::forward<ThisArg>(thisArg));
    setData<I + 1>(container, std::forward<Args>(args)...);
}

template<bool B, typename T = void> using disable_if = std::enable_if<!B, T>;
}

#define VALIDATE_SIZE()\
{\
    enum{ SecondSize = detail::VectorTraits<typename std::decay<SecondT>::type>::Size };\
    static_assert((SecondSize == SizeT) || (SecondSize == 1),\
        "The second object must be a scalar or of the same size as the vector");\
}

#define MEMBER_BINARY_OPERATOR(operator, operation)\
template<typename SecondT,\
typename OperationReturnT = typename std::result_of<operation<ScalarT>(ScalarT, ScalarT)>::type>\
SPECIFIER inline Vector<SizeT, OperationReturnT> operator(SecondT&& second) const\
{\
    VALIDATE_SIZE();\
    return detail::operator(*this, std::forward<SecondT>(second));\
}

#define MEMBER_BINARY_ASSIGNMENT(operator)\
template<typename SecondT>\
SPECIFIER inline void operator(SecondT&& second)\
{\
    VALIDATE_SIZE();\
    return detail::operator(*this, std::forward<SecondT>(second));\
}

#define UNARY_MEMBER_OPERATOR(operator, operation)\
template<typename OperationReturnT = typename std::result_of<operation<ScalarT>(ScalarT)>::type>\
SPECIFIER inline Vector<SizeT, OperationReturnT> operator()\
{\
    return detail::operator(*this);\
}

template<int SizeT, typename ScalarT>
class Vector
{
public:

    SPECIFIER Vector() = default;

    template<typename Arg, typename ...Args,
        typename = typename detail::disable_if<
        sizeof...(Args) == 0 && std::is_same<typename std::decay<Arg>::type, Vector>::value>::type>
        SPECIFIER explicit Vector(Arg&& arg, Args&&... args)
    {
        static_assert(sizeof...(Args) == SizeT - 1,
            "The number of arguments in the constructor must be equal to the size of the vector");
        detail::setData<0>(*this, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    MEMBER_BINARY_OPERATOR(operator+, std::plus)
    MEMBER_BINARY_OPERATOR(operator-, std::minus)
    MEMBER_BINARY_OPERATOR(operator*, std::multiplies)
    MEMBER_BINARY_OPERATOR(operator/, std::divides)
    MEMBER_BINARY_ASSIGNMENT(operator+=)
    MEMBER_BINARY_ASSIGNMENT(operator-=)
    MEMBER_BINARY_ASSIGNMENT(operator*=)
    MEMBER_BINARY_ASSIGNMENT(operator/=)
    MEMBER_BINARY_OPERATOR(operator==, std::equal_to)
    MEMBER_BINARY_OPERATOR(operator!=, std::not_equal_to)
    MEMBER_BINARY_OPERATOR(operator>, std::greater)
    MEMBER_BINARY_OPERATOR(operator<, std::less)
    MEMBER_BINARY_OPERATOR(operator>=, std::greater_equal)
    MEMBER_BINARY_OPERATOR(operator<=, std::less_equal)
    UNARY_MEMBER_OPERATOR(operator-, std::negate)
    
public:
    ScalarT data[SizeT];
};

template<typename ScalarT, typename ...Args>
SPECIFIER Vector<sizeof...(Args), ScalarT> make_vector(Args&&... args)
{
    return Vector<sizeof...(Args), ScalarT>(std::forward<Args>(args)...);
}
}
