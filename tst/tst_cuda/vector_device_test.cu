#include "inc/vector.h"
#include "gtest/gtest.h"

using namespace smalls;

template<int SizeT, typename ScalarT>
__global__ void runAll(Vector<SizeT, ScalarT> vec1, Vector<SizeT, ScalarT> vec2)
{
    ScalarT scalar(3.0f);
    auto res1 = vec1 + vec2;
    res1 = vec2 - vec1;
    res1 = vec2 * vec1;
    res1 = vec2 / vec1;
    res1 = vec1 + scalar;
    res1 = vec2 - scalar;
    res1 = vec2 * scalar;
    res1 = vec2 / scalar;
    vec2 += vec1;
    vec2 -= vec1;
    vec2 *= vec1;
    vec2 /= vec1;
    vec1 += scalar;
    vec2 -= scalar;
    vec2 *= scalar;
    vec2 /= scalar;
    res1 = math::min(vec2, scalar);
    res1 = math::max(vec2,  scalar);
    res1 = math::min(vec2, vec1);
    res1 = math::max(vec2, vec1);
    res1 = math::floor(vec2);
    res1 = math::ceil(vec2);

    auto res2 = vec2 == vec1;
    res2 = vec2 != vec1;
    res2 = vec2 > vec1;
    res2 = vec2 < vec1;
    res2 = vec2 >= vec1;
    res2 = vec2 <= vec1;
    res2 = vec2 == scalar;
    res2 = vec2 != scalar;
    res2 = vec2 > scalar;
    res2 = vec2 < scalar;
    res2 = vec2 >= scalar;
    res2 = vec2 <= scalar;
}

TEST(VectorDevice, RunAllOperations)
{
    runAll << <1, 1 >> >(make_vector<float>(1, 2, 3, 4), make_vector<float>(5, 4, 3, 2));
    runAll << <1, 1 >> >(make_vector<double>(-2, 1), make_vector<double>(3, 2));
    runAll << <1, 1 >> >(
        make_vector<char>(1, 2, 3, 4, 5, 6, 7, 8, 9),
        make_vector<char>(9, 8, 7, 6, 5, 4, 3, 2, 1));
}


typedef Vector<4, float> VectorT;

__global__ void testGPU(const VectorT* dvec1, const VectorT* dvec2, VectorT* result)
{

    *result = (*dvec2) + (*dvec1);
}


TEST(VectorDevice, RunUploadDownloadOperation)
{
    VectorT vec1(1, 2, 3, 4);
    VectorT vec2(5, 4, 3, 2);
    VectorT result;

    VectorT *dvec1, *dvec2, *dres;
    cudaMalloc((void**)&dvec1, sizeof(VectorT));
    cudaMemcpy(dvec1, vec1.data, sizeof(VectorT), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dvec2, sizeof(VectorT));
    cudaMemcpy(dvec2, vec2.data, sizeof(VectorT), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dres, sizeof(VectorT));

    testGPU<<<1, 1>>>(dvec1, dvec2, dres);
    cudaMemcpy(result.data, dres->data, sizeof(VectorT), cudaMemcpyDeviceToHost);

    EXPECT_EQ(6, result.data[0]);
    EXPECT_EQ(6, result.data[1]);
    EXPECT_EQ(6, result.data[2]);
    EXPECT_EQ(6, result.data[3]);
}
