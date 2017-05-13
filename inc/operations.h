#ifndef SMALLS_OPERATIONS_H
#define SMALLS_OPERATIONS_H

#include <functional>
#include "inc/storage.h"

namespace smalls
{
namespace detail
{

///------------------------------------------------------------------------------------------------
/// @brief Helper structure for Matrix introspection.
///------------------------------------------------------------------------------------------------
template<typename> struct matrix_trait;
template<
    template<size_t, size_t, typename, template<typename, size_t> class> class MatrixT,
    size_t RowsT, size_t ColsT, typename ElementT, template<typename, size_t> class StorageT>
struct matrix_trait<MatrixT<RowsT, ColsT, ElementT, StorageT>>
{
    static const bool IsMatrix = true;
    static const size_t Size = RowsT*ColsT;
    static const size_t Rows = RowsT;
    static const size_t Cols = ColsT;
    using Element = ElementT;
    template<size_t Rows, size_t Cols, typename NewElementT>
    using ContainerT = MatrixT<Rows, Cols, NewElementT, StorageOwned>;

    template<typename Matrix>
    SPECIFIER static inline const Element& get_value(Matrix&& matrix, size_t I)
        { return matrix.at(I); }
    SPECIFIER static inline Element& get_value(
        MatrixT<RowsT, ColsT, Element, StorageT>& matrix, size_t I)
        { return matrix.at(I); }
};

template<typename ScalarT> struct matrix_trait
{
    static const bool IsMatrix = false;
    static const size_t Size = 1;
    static const size_t Rows = 1;
    static const size_t Cols = 1;
    using Element = ScalarT;
    template<typename Scalar>
    SPECIFIER static inline const Element& get_value(Scalar&& scalar, size_t) { return scalar; }
    SPECIFIER static inline Element& get_value(ScalarT& scalar, size_t) { return scalar; }
};

///------------------------------------------------------------------------------------------------
/// @brief SFINAE on the condition that either the first or the second template parameter
/// be a Matrix type.
///------------------------------------------------------------------------------------------------
template<typename FirstT, typename SecondT> using enable_if_has_a_matrix =
typename std::enable_if<detail::matrix_trait<typename std::decay<FirstT>::type>::IsMatrix ||
detail::matrix_trait<typename std::decay<SecondT>::type>::IsMatrix>::type;

///------------------------------------------------------------------------------------------------
/// @brief Helper structure to select the Matrix from 2 input types. Will fail if non are.
/// Will return the first, if both are of type Matrix.
///------------------------------------------------------------------------------------------------
template<typename FirstT, typename SecondT, bool useFirst, bool useSecond> struct selectInput;
template<typename FirstT, typename SecondT>
struct selectInput<FirstT, SecondT, false, true> { using value = SecondT; };
template<typename FirstT, typename SecondT, bool useSecond>
struct selectInput<FirstT, SecondT, true, useSecond> { using value = FirstT; };

///------------------------------------------------------------------------------------------------
/// @brief Helper structure to determine the type of an output Matrix given the input Matrix type
/// and the operation to be performed.
///------------------------------------------------------------------------------------------------
template<typename InputType, template<typename> class OperationFunctor> struct outputType
{
    using Element = typename detail::matrix_trait<typename std::decay<InputType>::type>::Element;
    static const size_t InputRows = detail::matrix_trait<typename std::decay<InputType>::type>::Rows;
    static const size_t InputCols = detail::matrix_trait<typename std::decay<InputType>::type>::Cols;
    using Return = typename std::result_of<OperationFunctor<Element>(Element, Element)>::type;
    using Type = typename detail::matrix_trait<typename std::decay<InputType>::type>::
        template ContainerT<InputRows, InputCols, Return>;
};

///------------------------------------------------------------------------------------------------
/// @brief Helper structure to find a Matrix in 2 template parameters (handles the case where only
/// 1 template parameter is passed.
///------------------------------------------------------------------------------------------------
template<typename FirstT, typename SecondT = void> struct findMatrixType
{
    using First = typename std::decay<FirstT>::type;
    using Second = typename std::decay<SecondT>::type;
    using Type = typename selectInput<First, Second,
        detail::matrix_trait<First>::IsMatrix, detail::matrix_trait<Second>::IsMatrix>::value;
};
template<typename FirstT> struct findMatrixType<FirstT, void>
{ using Type = typename std::decay<FirstT>::type; };

///------------------------------------------------------------------------------------------------
/// @brief Helper structure to apply an operation to a Matrix, element-wise.
///------------------------------------------------------------------------------------------------
template<template<typename> class OperationT,
    typename OutputElementT, size_t I, typename InputT1, typename InputT2>
SPECIFIER inline OutputElementT
apply_operation(InputT1&& input1, InputT2&& input2)
{
    using Input1Element = typename matrix_trait<typename std::decay<InputT1>::type>::Element;
    using Input2Element = typename matrix_trait<typename std::decay<InputT2>::type>::Element;

    static_assert(std::is_same<typename std::remove_cv<Input1Element>::type,
        typename std::remove_cv<Input2Element>::type>::value,
        "The value type for the operation are different - implicit conversion are not allowed "
        "(e.g. double->float), even narrowing conversions");

    return OperationT<Input1Element>()(
        matrix_trait<typename std::decay<InputT1>::type>::get_value(input1, I),
        matrix_trait<typename std::decay<InputT2>::type>::get_value(input2, I));
}
template<template<typename> class OperationT, typename OutputElementT, size_t I, typename InputT>
SPECIFIER inline OutputElementT
apply_operation(InputT&& input)
{
    using InputElement = typename matrix_trait<typename std::decay<InputT>::type>::Element;
    return OperationT<InputElement>()(
            matrix_trait<typename std::decay<InputT>::type>::get_value(input, I));
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to make a compound operator.
///------------------------------------------------------------------------------------------------
#define MAKE_COMPOUND(name, symbol)\
struct name\
{\
    template<typename T, typename U>\
    inline SPECIFIER void operator()(T& lhs, U&& rhs) const { lhs symbol rhs; }\
};

///------------------------------------------------------------------------------------------------
/// @brief Make standard Matrix compound operators.
///------------------------------------------------------------------------------------------------
MAKE_COMPOUND(assign, =)
MAKE_COMPOUND(assignPlus, +=)
MAKE_COMPOUND(assignMul, *=)
MAKE_COMPOUND(assignAnd, &=)
MAKE_COMPOUND(assignOr, |=)

///------------------------------------------------------------------------------------------------
/// @brief Helper method to unwrap a function parameter pack.
///------------------------------------------------------------------------------------------------
template<
    template<typename> class OperationT, size_t I = 0, size_t SizeT,
    typename AssignmentT = assign,
    typename = typename std::enable_if<I == SizeT-1, void>::type, 
    typename OutputT, typename ...Inputs, size_t = 0>
SPECIFIER inline void
unwrap_operation(OutputT& res, Inputs&&... inputs)
{
    using OutputElement = typename matrix_trait<typename std::decay<OutputT>::type>::Element;
    AssignmentT()(matrix_trait<typename std::decay<OutputT>::type>::get_value(res, I),
        apply_operation<OperationT, OutputElement, I>(std::forward<Inputs>(inputs)...));
}
template<
    template<typename> class OperationT, size_t I = 0, size_t SizeT,
    typename AssignmentT,
    typename = typename std::enable_if<I != SizeT-1, void>::type,
    typename OutputT, typename ...Inputs>
SPECIFIER inline void
unwrap_operation(OutputT& res, Inputs&&... inputs)
{
    using OutputElement = typename matrix_trait<typename std::decay<OutputT>::type>::Element;
    AssignmentT()(matrix_trait<typename std::decay<OutputT>::type>::get_value(res, I),
        apply_operation<OperationT, OutputElement, I>(std::forward<Inputs>(inputs)...));
    unwrap_operation<OperationT, I + 1, SizeT, AssignmentT>(res, std::forward<Inputs>(inputs)...);
}

///------------------------------------------------------------------------------------------------
/// @brief Helper method to initiate an operation that wil unwrap a function parameter pack.
///------------------------------------------------------------------------------------------------
template<template<typename> class OperationT, typename AssignmentT,
    typename OutputT, typename ...Inputs>
SPECIFIER inline void setup_operation(OutputT& res, Inputs&&... inputs)
{
    const auto Size = matrix_trait<typename detail::findMatrixType<Inputs...>::Type>::Size;
    unwrap_operation<OperationT, 0, Size, AssignmentT>(res, std::forward<Inputs>(inputs)...);
}

///------------------------------------------------------------------------------------------------
/// @brief Helper method to initiate an assignment that wil unwrap a function parameter pack.
///------------------------------------------------------------------------------------------------
template<template<typename> class OperationT, typename FirstT, typename ...Inputs>
SPECIFIER inline void setup_assignment(FirstT& first, Inputs&&... inputs)
{
    const auto Size = matrix_trait<typename std::decay<FirstT>::type>::Size;
    unwrap_operation<OperationT, 0, Size, assign>(first, first, std::forward<Inputs>(inputs)...);
}
} // end namespace detail

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to define a binary operator.
///------------------------------------------------------------------------------------------------
#define BINARY_OPERATOR(operator, operation)\
template<typename FirstT, typename SecondT,\
    typename = detail::enable_if_has_a_matrix<FirstT, SecondT>,\
    typename OutputT = typename detail::outputType<\
        typename detail::findMatrixType<FirstT, SecondT>::Type, operation>::Type>\
SPECIFIER inline OutputT operator(FirstT&& first, SecondT&& second)\
{\
    OutputT res{};\
    detail::setup_operation<operation, detail::assign>(res, \
        std::forward<FirstT>(first), std::forward<SecondT>(second)); \
    return res; \
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to define a unary operator.
///------------------------------------------------------------------------------------------------
#define UNARY_OPERATOR(operator, operation)\
template<typename FirstT,\
    typename ElementT = typename detail::matrix_trait<typename std::decay<FirstT>::type>::Element,\
    size_t InputRows = detail::matrix_trait<typename std::decay<FirstT>::type>::Rows,\
    size_t InputCols = detail::matrix_trait<typename std::decay<FirstT>::type>::Cols,\
    typename ReturnT = typename std::result_of<operation<ElementT>(ElementT)>::type,\
    typename OutputT = typename detail::matrix_trait<typename std::decay<FirstT>::type>::\
        template ContainerT<InputRows, InputCols, ReturnT>>\
SPECIFIER inline OutputT operator(FirstT&& first)\
{\
    OutputT res{};\
    detail::setup_operation<operation, detail::assign>(res, std::forward<FirstT>(first));\
    return res;\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to define a binary assignment.
///------------------------------------------------------------------------------------------------
#define BINARY_ASSIGNMENT(operator, operation)\
template<typename FirstT, typename SecondT>\
SPECIFIER inline void operator(FirstT& first, SecondT&& second)\
{\
    return detail::setup_assignment<operation>(first, std::forward<SecondT>(second));\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to define a reduction.
///------------------------------------------------------------------------------------------------
#define REDUCTION(operator, operation, assignment)\
template<typename FirstT, typename SecondT,\
    typename = detail::enable_if_has_a_matrix<FirstT, SecondT>,\
    typename OutputT = typename detail::outputType<\
        typename detail::findMatrixType<FirstT, SecondT>::Type, operation>::Return>\
SPECIFIER inline OutputT operator(FirstT&& first, SecondT&& second)\
{\
    OutputT res{};\
    detail::setup_operation<operation, assignment>(res, \
        std::forward<FirstT>(first), std::forward<SecondT>(second));\
    return res;\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro to define a unary reduction.
///------------------------------------------------------------------------------------------------
#define REDUCTION_UNARY(operator, operation, assignment, initialValue)\
template<typename FirstT,\
    typename ElementT = typename detail::matrix_trait<typename std::decay<FirstT>::type>::Element,\
    size_t InputRows = detail::matrix_trait<typename std::decay<FirstT>::type>::Rows,\
    size_t InputCols = detail::matrix_trait<typename std::decay<FirstT>::type>::Cols,\
    typename OutputT = typename std::result_of<operation<ElementT>(ElementT)>::type>\
SPECIFIER inline OutputT operator(FirstT&& first)\
{\
    OutputT res{initialValue};\
    detail::setup_operation<operation, assignment>(res, std::forward<FirstT>(first));\
    return res;\
}

namespace detail
{
///------------------------------------------------------------------------------------------------
/// @brief Define Matrix standard operations.
///------------------------------------------------------------------------------------------------
#define DEFINE_BINARY_OPERATION(operation, symbol)\
template<typename T> struct operation\
{\
    inline SPECIFIER T operator()(const T& lhs, const T& rhs) const { return (lhs symbol rhs); }\
};

#define DEFINE_COMPARISON_OPERATION(operation, symbol)\
template<typename T> struct operation\
{\
    inline SPECIFIER bool operator()(const T& lhs, const T& rhs) const { return (lhs symbol rhs); }\
};

template<typename T> struct min
{
    inline SPECIFIER T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};

template<typename T> struct max
{
    inline SPECIFIER T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

template<typename T> struct noop
{
    inline SPECIFIER T operator()(const T& a) const { return a; }
};

template<typename T> struct negate
{
    inline SPECIFIER T operator()(const T& a) const { return -a; }
};

DEFINE_BINARY_OPERATION(plus, +);
DEFINE_BINARY_OPERATION(minus, -);
DEFINE_BINARY_OPERATION(multiplies, *);
DEFINE_BINARY_OPERATION(divides, /);
DEFINE_COMPARISON_OPERATION(equal_to, ==);
DEFINE_COMPARISON_OPERATION(not_equal_to, != );
DEFINE_COMPARISON_OPERATION(greater, >);
DEFINE_COMPARISON_OPERATION(less, <);
DEFINE_COMPARISON_OPERATION(greater_equal, >=);
DEFINE_COMPARISON_OPERATION(less_equal, <=);

BINARY_ASSIGNMENT(operator+=, plus);
BINARY_ASSIGNMENT(operator-=, minus);
BINARY_ASSIGNMENT(operator*=, multiplies);
BINARY_ASSIGNMENT(operator/=, divides);
UNARY_OPERATOR(operator-, negate);

REDUCTION(dot, multiplies, detail::assignPlus);
REDUCTION_UNARY(all, noop, detail::assignAnd, true);
REDUCTION_UNARY(any, noop, detail::assignOr, false);
REDUCTION_UNARY(sum, noop, detail::assignPlus, 0);
REDUCTION_UNARY(prod, noop, detail::assignMul, 1);
}

BINARY_OPERATOR(operator+,  detail::plus);
BINARY_OPERATOR(operator-,  detail::minus);
BINARY_OPERATOR(operator*,  detail::multiplies);
BINARY_OPERATOR(operator/,  detail::divides);
BINARY_OPERATOR(operator==, detail::equal_to);
BINARY_OPERATOR(operator!=, detail::not_equal_to);
BINARY_OPERATOR(operator>,  detail::greater);
BINARY_OPERATOR(operator<,  detail::less);
BINARY_OPERATOR(operator>=, detail::greater_equal);
BINARY_OPERATOR(operator<=, detail::less_equal);
BINARY_OPERATOR(min, detail::min);
BINARY_OPERATOR(max, detail::max);
}

#endif //SMALLS_OPERATIONS_H
