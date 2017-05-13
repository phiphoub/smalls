#ifndef SMALLS_VECTOR_H
#define SMALLS_VECTOR_H

#include "inc/operations.h"
#include "inc/matrix.h"
#include <array>
#include <math.h>

namespace smalls
{

///------------------------------------------------------------------------------------------------
/// @brief Creates a Vector alias as a column Matrix.
///------------------------------------------------------------------------------------------------
template<size_t SizeT, typename ScalarT, template <typename, size_t> class StorageT = StorageOwned>
    using Vector = Matrix<SizeT, 1, ScalarT, StorageT>;

///------------------------------------------------------------------------------------------------
/// @brief Helper function to create a Vector. The size is deduced at compile time, user needs
/// to specify the type.
///------------------------------------------------------------------------------------------------
template<typename ScalarT, typename ...Args>
SPECIFIER Vector<sizeof...(Args), ScalarT> make_vector(Args&&... args)
{
    return Vector<sizeof...(Args), ScalarT>(std::forward<Args>(args)...);
}

///------------------------------------------------------------------------------------------------
/// @brief Helper function to create a Vector that maps the input Vector.
/// This is useful to create "sub-views" of a Vector and avoid unnecessary copies.
///------------------------------------------------------------------------------------------------
template<size_t startIndex, size_t endIndex, size_t RowsT, size_t ColsT, typename ScalarT,
    template<size_t, size_t, typename, template <typename, size_t > class> class VectorT,
    template <typename, size_t> class StorageT,
    size_t mappedSize = endIndex - startIndex + 1,
    typename MappedVector = VectorT<mappedSize, ColsT, ScalarT, StorageMapped >>
SPECIFIER MappedVector make_map_vector(VectorT<RowsT, ColsT, ScalarT, StorageT>& vector)
{
    static_assert(
        endIndex <= VectorT<RowsT, ColsT, ScalarT, StorageT>::Size &&
        startIndex <= VectorT<RowsT, ColsT, ScalarT, StorageT>::Size &&
        startIndex <= endIndex,
        "The indices cannot be greater than the size of the underlying vector and "
        "the start index cannot be greater than the end index.");
    static_assert(1 == ColsT, "You are trying to map a non-vector type - mapping is only available "
        "for vector for now");
    StorageMapped<ScalarT, mappedSize> storageMapped(vector.data() + startIndex);
    return MappedVector(storageMapped);
}
}

#endif //SMALLS_VECTOR_H
