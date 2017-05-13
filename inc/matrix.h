#ifndef SMALLS_MATRIX_H
#define SMALLS_MATRIX_H

#include "inc/operations.h"
#include "inc/storage.h"
#include <array>
#include <assert.h>
#include <cmath>
#include <cstring>

namespace smalls
{
namespace detail
{

///------------------------------------------------------------------------------------------------
/// @brief Helper method to setup data in a container (i.e. Matrix) from a function parameter pack.
///------------------------------------------------------------------------------------------------
template<size_t I, typename ContainerT, typename ThisArg>
SPECIFIER inline void set_data(ContainerT& container, ThisArg&& arg)
{
    using ElementT = typename matrix_trait<ContainerT>::Element;
    container.at(I) = static_cast<ElementT>(std::forward<ThisArg>(arg));
}

template<size_t I, typename ContainerT, typename ThisArg, typename... Args>
SPECIFIER inline void set_data(ContainerT& container, ThisArg&& thisArg, Args&&... args)
{
    using ElementT = typename matrix_trait<ContainerT>::Element;
    container.at(I) = static_cast<ElementT>(std::forward<ThisArg>(thisArg));
    set_data<I + 1>(container, std::forward<Args>(args)...);
}

template<bool B, typename T = void> using disable_if = std::enable_if<!B, T>;

///------------------------------------------------------------------------------------------------
/// @brief Helper structure validating a container size.
///------------------------------------------------------------------------------------------------
template<typename ContainerT, size_t expectedRows, size_t expectedCols>
struct validate_size
{
    ///--------------------------------------------------------------------------------------------
    /// @brief Will assert if the container is not of the expected rows and columns size or if it
    /// is a scalar
    ///--------------------------------------------------------------------------------------------
    SPECIFIER static void assert_if_not_valid() {
        const size_t NumRows = detail::matrix_trait<typename std::decay<ContainerT>::type>::Rows;\
        const size_t NumCols = detail::matrix_trait<typename std::decay<ContainerT>::type>::Cols;\
        static_assert(((NumRows == expectedRows) && (NumCols == expectedCols)) ||\
            ((NumRows == 1) && (NumCols == 1)), "The container is not of the expected "
                "size (see compiler instantiation trace) or is a scalar");\
    }
};
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro defining the supported reduction operators for the Matrix class.
///------------------------------------------------------------------------------------------------
#define MEMBER_REDUCTION(operator, operation)\
template<typename SecondT,\
typename ReturnT = typename std::result_of<operation<ScalarT>(ScalarT, ScalarT)>::type>\
SPECIFIER inline ReturnT operator(SecondT&& second) const\
{\
    detail::validate_size<SecondT, RowsT, ColsT>::assert_if_not_valid();\
    return detail::operator(*this, std::forward<SecondT>(second));\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro defining the supported binary assignment operators for the Matrix class.
///------------------------------------------------------------------------------------------------
#define MEMBER_BINARY_ASSIGNMENT(operator)\
template<typename SecondT>\
SPECIFIER inline void operator(SecondT&& second)\
{\
    detail::validate_size<SecondT, RowsT, ColsT>::assert_if_not_valid();\
    detail::operator(*this, std::forward<SecondT>(second));\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro defining the supported unary assignment operators for the Matrix class.
///------------------------------------------------------------------------------------------------
#define UNARY_MEMBER_OPERATOR(operator, operation)\
template<typename ReturnT = typename std::result_of<operation<ScalarT>(ScalarT)>::type>\
SPECIFIER inline Matrix<Rows, Cols, ReturnT> operator()\
{\
    return detail::operator(*this);\
}

///------------------------------------------------------------------------------------------------
/// @brief Helper macro defining the supported unary reduction operators for the Matrix class.
///------------------------------------------------------------------------------------------------
#define UNARY_MEMBER_REDUCTION(operator, operation)\
template<typename ReturnT = typename std::result_of<operation<ScalarT>(ScalarT)>::type>\
SPECIFIER inline ReturnT operator() const\
{\
    return detail::operator(*this);\
}

///------------------------------------------------------------------------------------------------
/// @brief The Matrix class provides encapsulation of 2D data and operators.
///------------------------------------------------------------------------------------------------
template<size_t RowsT, size_t ColsT, typename ScalarT,
    template <typename, size_t> class StorageT = StorageOwned>
class Matrix
{
public:

    using Element = ScalarT;

    static const size_t Size = RowsT * ColsT;
    static const size_t Rows = RowsT;
    static const size_t Cols = ColsT;

    SPECIFIER Matrix() = default;
    SPECIFIER Matrix(const StorageT<ScalarT, Size>& storage) : m_storage(storage) {};

    ///------------------------------------------------------------------------------------------------
    /// @brief Variadic template constructor. Some restrictions need to be applied to avoid binding
    /// to everything.
    ///------------------------------------------------------------------------------------------------
    template<typename Arg, typename ...Args,
        typename = typename detail::disable_if<
        sizeof...(Args) == 0 &&
        (std::is_same<typename std::decay<Arg>::type, Matrix>::value ||
        std::is_same<typename std::decay<Arg>::type, StorageT<ScalarT, Size>>::value)>::type>
        SPECIFIER explicit Matrix(Arg&& arg, Args&&... args)
    {
        static_assert(sizeof...(Args) == Size - 1,
            "The number of arguments in the constructor must be equal to the size of the matrix");
        detail::set_data<0>(*this, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Move assignment operator.
    ///------------------------------------------------------------------------------------------------
    template<typename OtherMatrixT,
        size_t OtherColsT = detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::Cols,
        size_t OtherRowsT = detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::Rows,
        typename OtherScalarT = 
            typename detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::Element,
        typename = typename std::enable_if<(OtherColsT == ColsT) && (OtherRowsT == RowsT) &&
        std::is_same<typename std::remove_cv<ScalarT>::type, typename std::remove_cv<OtherScalarT>::
            type>::value>::type>
    SPECIFIER Matrix& operator=(OtherMatrixT&& other)
    {
        const auto numBytes = Size*sizeof(ScalarT);
        memmove(static_cast<void*>(this->data()), static_cast<const void*>(other.data()), numBytes);
        return *this;
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Common member operators definition.
    ///------------------------------------------------------------------------------------------------
    MEMBER_BINARY_ASSIGNMENT(operator+=)
    MEMBER_BINARY_ASSIGNMENT(operator-=)
    MEMBER_BINARY_ASSIGNMENT(operator*=)
    MEMBER_BINARY_ASSIGNMENT(operator/=)
    UNARY_MEMBER_OPERATOR(operator-, detail::negate)
    MEMBER_REDUCTION(dot, detail::multiplies)
    UNARY_MEMBER_REDUCTION(all, detail::noop)
    UNARY_MEMBER_REDUCTION(any, detail::noop)
    UNARY_MEMBER_REDUCTION(sum, detail::noop)
    UNARY_MEMBER_REDUCTION(prod, detail::noop)

    ///------------------------------------------------------------------------------------------------
    /// @brief Calculates the Frobenius norm of a matrix (or L-2 norm of a vector).
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT norm() const
    {
        // CUDA does not provide functionality for sqrt other than for float and double types.
        static_assert(std::is_floating_point<ScalarT>::value,
            "Scalar must be of trait floating-point (e.g. float, double) to use norm()");
        return sqrt(this->dot(*this));
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Normalize the matrix with the Frobenius norm.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline void normalize()
    {
        *this /= norm();
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Computes the transpose of a matrix.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER Matrix<Cols, Rows, typename std::remove_cv<ScalarT>::type> transpose() const
    {
        auto transpose = Matrix<Cols, Rows, typename std::remove_cv<ScalarT>::type>{};
        for (auto rowIdx = 0; Rows > rowIdx; ++rowIdx)
        {
            for (auto colIdx = 0; Cols > colIdx; ++colIdx)
            {
                transpose.at(colIdx*Rows + rowIdx) = this->at(rowIdx*Cols + colIdx);
            }
        }
        return transpose;
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Get a pointer to the underlying data. The ownership remains with the Matrix object and
    /// the pointer to the data is invalidated when the Matrix object gets out of scope.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline const ScalarT* data() const
    {
        return m_storage.data();
    }

    SPECIFIER inline ScalarT* data()
    {
        return const_cast<ScalarT*>(static_cast<const Matrix*>(this)->data());
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Get a reference to a specific index in the Matrix object.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline const ScalarT& at(size_t index) const
    {
        assert((Size > index) && "Out of bound index");
        return this->data()[index];
    }

    SPECIFIER inline const ScalarT& at(size_t rowIdx, size_t colIdx) const
    {
        assert((Rows > rowIdx) && (Cols > colIdx) && "Out of bound indices");
        return this->at(rowIdx*Cols + colIdx);
    }

    SPECIFIER inline ScalarT& at(size_t index)
    {
        return const_cast<ScalarT&>(static_cast<const Matrix*>(this)->at(index));
    }

    SPECIFIER inline ScalarT& at(size_t rowIdx, size_t colIdx)
    {
        return const_cast<ScalarT&>(static_cast<const Matrix*>(this)->at(rowIdx, colIdx));
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Map a continuous chunk of data. The map will create a Matrix object around
    /// the underlying data, thus avoiding copying the data.
    ///------------------------------------------------------------------------------------------------
    template<size_t StartT = 0, size_t EndT = Size, size_t NewSizeT = EndT - StartT,
        typename NewMatrixT = Matrix<NewSizeT, 1, const ScalarT, StorageMapped>>
        SPECIFIER inline NewMatrixT mapContinuous() const
    {
        static_assert(StartT <= EndT, "The start index must not be greater than the end index");
        static_assert(EndT <= Size,
            "The end index must be smaller than the size of the original matrix");
        return NewMatrixT(StorageMapped<const ScalarT, NewSizeT>(this->data() + StartT));
    }

    template<size_t StartT = 0, size_t EndT = Size, size_t NewSizeT = EndT - StartT,
        typename NewMatrixT = Matrix<NewSizeT, 1, ScalarT, StorageMapped>>
        SPECIFIER inline NewMatrixT mapContinuous()
    {
        static_assert(StartT <= EndT, "The start index must not be greater than the end index");
        static_assert(EndT <= Size,
            "The end index must be smaller than the size of the original matrix");
        return NewMatrixT(StorageMapped<ScalarT, NewSizeT>(this->data() + StartT));
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Matrix multiplication.
    ///------------------------------------------------------------------------------------------------
    template<typename OtherMatrixT,
        typename ElementT = typename std::remove_cv<typename detail::matrix_trait<
            typename std::decay<OtherMatrixT>::type>::Element>::type,
        size_t OtherColsT = detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::Cols,
        size_t OtherRowsT = detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::Rows,
        typename ResultT = typename detail::matrix_trait<typename std::decay<OtherMatrixT>::type>::
        template ContainerT<Rows, OtherColsT, ElementT >>
        SPECIFIER ResultT mul(OtherMatrixT&& other) const
    {
        static_assert(std::is_same<typename std::remove_cv<ScalarT>::type,
                ElementT>::value,
            "The type of elements for matrix multiplication must be the same");
        static_assert(OtherRowsT == Cols, "Matrix sizes must agree");
        static_assert(Rows*OtherColsT > 1,
            "The matrix resulting of the product will be a scalar. Matrix of size 1x1 are not "
            " supported. However the Scalar type will handle the operator that you require. "
            " Use dot() to perform the vector multiplication");

        const auto& otherTranspose = other.transpose();

        ResultT product;
        for (size_t rowIdx = 0; Rows > rowIdx; ++rowIdx)
        {
            const auto startThis = this->data() + rowIdx*Cols;
            for (size_t colIdx = 0; OtherColsT > colIdx; ++colIdx)
            {
                const auto startOther = otherTranspose.data() + colIdx*OtherRowsT;

                ElementT dot(0);
                for (size_t index = 0; Cols > index; ++index)
                {
                    dot += startThis[index] * startOther[index];
                }
                product.at(rowIdx, colIdx) = dot;
            }
        }
        return product;
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Helper method generating a Matrix of zeros.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline static Matrix Zero()
    {
        return set_value(0, 0, 1);
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Helper method generating a Matrix of ones.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline static Matrix One()
    {
        return set_value(1, 0, 1);
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Helper method generating a Matrix with 1 on its diagonal.
    ///------------------------------------------------------------------------------------------------
    SPECIFIER inline static Matrix Identity()
    {
        return set_value(1, 0, Cols + 1);
    }

    ///------------------------------------------------------------------------------------------------
    /// @brief Cast operator. Casts a Matrix with element of a given type to another
    /// Matrix of a different type.
    ///------------------------------------------------------------------------------------------------
    template<typename OtherScalarT,
        typename NewMatrixT = Matrix<Rows, Cols, OtherScalarT, StorageOwned>>
    SPECIFIER inline NewMatrixT cast() const
    {
        NewMatrixT newMatrix;
        for(size_t i = 0; Size > i; ++i)
        {
            newMatrix.at(i) = static_cast<OtherScalarT>(this->at(i));
        }
        return newMatrix;
    }

private:
    static Matrix set_value(ScalarT value, size_t offset, size_t stride)
    {
        auto matrix = Matrix{};
        for (auto index = offset; Size > index; index += stride)
        {
            matrix.at(index) = value;
        }
        return matrix;
    }

private:
    StorageT<ScalarT, Size> m_storage;
};

///------------------------------------------------------------------------------------------------
/// @brief Helper method to create a matrix of specified size by passing a parameter pack.
///------------------------------------------------------------------------------------------------
template<size_t Rows, size_t Cols, typename ScalarT, typename ...Args>
SPECIFIER Matrix<Rows, Cols, ScalarT> make_matrix(Args&&... args)
{
    return Matrix<Rows, Cols, ScalarT>(std::forward<Args>(args)...);
}
}

#endif //SMALLS_MATRIX_H
