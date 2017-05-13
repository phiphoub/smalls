#ifndef SMALLS_MATRIX_STREAM_H
#define SMALLS_MATRIX_STREAM_H

#include "math/inc/operations.h"
#include <ostream>

namespace smalls
{

///------------------------------------------------------------------------------------------------
/// @brief Dump a matrix to an output stream.
///------------------------------------------------------------------------------------------------
template<typename MatrixT>
typename std::enable_if<detail::matrix_trait<
    typename std::decay<MatrixT>::type>::IsMatrix, std::ostream&>::type
operator<< (std::ostream& stream, const MatrixT& matrix)
{
    const size_t Rows = detail::matrix_trait<typename std::decay<MatrixT>::type>::Rows;
    const size_t Cols = detail::matrix_trait<typename std::decay<MatrixT>::type>::Cols;

    stream << std::endl;
    for (size_t rowIdx = 0; Rows> rowIdx; ++rowIdx)
    {
        for (size_t colIdx = 0; Cols> colIdx; ++colIdx)
        {
            stream << matrix.at(rowIdx, colIdx) << "\t";
        }
        stream << std::endl;
    }
    return stream;
}
}

#endif //SMALLS_MATRIX_STREAM_H
