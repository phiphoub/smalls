#ifndef SMALLS_STORAGE_H
#define SMALLS_STORAGE_H

#ifdef __CUDA_ARCH__
#define SPECIFIER __device__ __host__
#else
#define SPECIFIER 
#endif

namespace smalls
{

///------------------------------------------------------------------------------------------------
/// @brief StorageOwned defines a class that owns data storage for a Matrix.
///------------------------------------------------------------------------------------------------
template<typename ScalarT, size_t SizeT>
class StorageOwned
{
public:
    SPECIFIER StorageOwned() : m_data{} {};
    SPECIFIER inline ScalarT* data()
    {
        return m_data;
    }

    SPECIFIER inline const ScalarT* data() const
    {
        return m_data;
    }
private:
    ScalarT m_data[SizeT];
};

///------------------------------------------------------------------------------------------------
/// @brief StorageMapped defines a class that does not own data storage for a Matrix,
/// it simply proxies a pointer to the data that it was passed.
///------------------------------------------------------------------------------------------------
template<typename ScalarT, size_t SizeT>
class StorageMapped
{
public:
    SPECIFIER StorageMapped() : m_data{} {};
    SPECIFIER StorageMapped(ScalarT* data) : m_data{ data } {};
    SPECIFIER inline ScalarT* data()
    {
        return m_data;
    }

    SPECIFIER inline const ScalarT* data() const
    {
        return m_data;
    }
private:
    ScalarT* m_data;
};
}

#endif //SMALLS_STORAGE_H
