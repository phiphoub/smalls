#include "inc/vector.h"
#include "tst/cuda_types_utils.h"
#include "gtest/gtest.h"

using namespace smalls;

namespace test
{
class VectorTest
        : public ::testing::Test
{
public:
    typedef float TestType;
    using VectorT = Vector<3, TestType>;

    void SetUp() override
    {
        m_vector1 = make_vector<TestType>(2.5f, 1.0f, 3.0f);
        m_vector2 = make_vector<TestType>(5.0f, 1.0f, 6.0f);
        m_scalar = 5.0f;
    }

protected:
    VectorT m_vector1, m_vector2;
    TestType m_scalar;
    TestType m_eps = 1.0e-5f;
};

#define TEST_VECTOR_OPERATOR(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_vector1);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_SCALAR_OPERATOR(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_scalar);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_SCALAR_OPERATOR_SWP(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_scalar, m_vector2);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_REDUCTION_VECTOR(NAME, OPERATOR, RESULT)\
TEST_F(VectorTest, NAME)\
{\
    auto result = m_vector2.OPERATOR(m_vector1);\
    EXPECT_NEAR(RESULT, result, m_eps);\
}

#define TEST_REDUCTION_SCALAR(NAME, OPERATOR, RESULT)\
TEST_F(VectorTest, NAME)\
{\
    auto result = m_vector2.OPERATOR(m_scalar);\
    EXPECT_NEAR(RESULT, result, m_eps);\
}

#define TEST_VECTOR_ASSIGNMENT(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    m_vector2.OPERATOR(m_vector1);\
    EXPECT_NEAR(RESULT1, m_vector2.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, m_vector2.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, m_vector2.at(2), m_eps);\
}

#define TEST_SCALAR_ASSIGNMENT(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    m_vector2.OPERATOR(m_scalar);\
    EXPECT_NEAR(RESULT1, m_vector2.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, m_vector2.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, m_vector2.at(2), m_eps);\
}

#define TEST_VECTOR_BINARY_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_vector1);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_SCALAR_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_scalar);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_SCALAR_MATH_SWAPPED(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_scalar, m_vector2);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

#define TEST_VECTOR_UNARY_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTest, NAME)\
{\
    auto result = OPERATOR(m_vector2);\
    EXPECT_NEAR(RESULT1, result.at(0), m_eps);\
    EXPECT_NEAR(RESULT2, result.at(1), m_eps);\
    EXPECT_NEAR(RESULT3, result.at(2), m_eps);\
}

TEST_VECTOR_OPERATOR(BinaryOperator_plus_vector, operator+, 7.5f, 2.0f, 9.0f)

TEST_VECTOR_OPERATOR(BinaryOperator_minus_vector, operator-, 2.5f, 0.0f, 3.0f)

TEST_VECTOR_OPERATOR(BinaryOperator_multiplies_vector, operator*, 12.5f, 1.0f, 18.0f)

TEST_VECTOR_OPERATOR(BinaryOperator_divides_vector, operator/, 2.0f, 1.0f, 2.0f)

TEST_VECTOR_OPERATOR(BinaryOperator_equal_to_vector, operator==, false, true, false)

TEST_VECTOR_OPERATOR(BinaryOperator_not_equal_to_vector, operator!=, true, false, true)

TEST_VECTOR_OPERATOR(BinaryOperator_greater_vector, operator>, true, false, true)

TEST_VECTOR_OPERATOR(BinaryOperator_less_vector, operator<, false, false, false)

TEST_VECTOR_OPERATOR(BinaryOperator_greater_equal_vector, operator>=, true, true, true)

TEST_VECTOR_OPERATOR(BinaryOperator_less_equal_vector, operator<=, false, true, false)

TEST_VECTOR_ASSIGNMENT(BinaryAssignment_plus_vector, operator+=, 7.5f, 2.0f, 9.0f)

TEST_VECTOR_ASSIGNMENT(BinaryAssignment_minus_vector, operator-=, 2.5f, 0.0f, 3.0f)

TEST_VECTOR_ASSIGNMENT(BinaryAssignment_multiplies_vector, operator*=, 12.5f, 1.0f, 18.0f)

TEST_VECTOR_ASSIGNMENT(BinaryAssignment_divides_vector, operator/=, 2.0f, 1.0f, 2.0f)

TEST_SCALAR_OPERATOR(BinaryOperator_plus_scalar, operator+, 10.0f, 6.0f, 11.0f)

TEST_SCALAR_OPERATOR(BinaryOperator_minus_scalar, operator-, 0.0f, -4.0f, 1.0f)

TEST_SCALAR_OPERATOR(BinaryOperator_multiplies_scalar, operator*, 25.0f, 5.0f, 30.0f)

TEST_SCALAR_OPERATOR(BinaryOperator_divides_scalar, operator/, 1.0f, 0.2f, 1.2f)

TEST_SCALAR_OPERATOR(BinaryOperator_equal_to_scalar, operator==, true, false, false)

TEST_SCALAR_OPERATOR(BinaryOperator_not_equal_to_scalar, operator!=, false, true, true)

TEST_SCALAR_OPERATOR(BinaryOperator_greater_scalar, operator>, false, false, true)

TEST_SCALAR_OPERATOR(BinaryOperator_less_scalar, operator<, false, true, false)

TEST_SCALAR_OPERATOR(BinaryOperator_greater_equal_scalar, operator>=, true, false, true)

TEST_SCALAR_OPERATOR(BinaryOperator_less_equal_scalar, operator<=, true, true, false)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_plus_scalar_swapped, operator+, 10.0f, 6.0f, 11.0f)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_minus_scalar_swapped, operator-, 0.0f, 4.0f, -1.0f)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_multiplies_scalar_swapped, operator*, 25.0f, 5.0f, 30.0f)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_divides_scalar_swapped, operator/, 1.0f, 5.0f, 1.0f / 1.2f)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_equal_to_scalar_swapped, operator==, true, false, false)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_not_equal_to_scalar_swapped, operator!=, false, true, true)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_greater_scalar_swapped, operator>, false, true, false)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_less_scalar_swapped, operator<, false, false, true)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_greater_equal_scalar_swapped, operator>=, true, true, false)

TEST_SCALAR_OPERATOR_SWP(BinaryOperator_less_equal_scalar_swapped, operator<=, true, false, true)

TEST_SCALAR_ASSIGNMENT(BinaryAssignment_plus_scalar, operator+=, 10.0f, 6.0f, 11.0f)

TEST_SCALAR_ASSIGNMENT(BinaryAssignment_minus_scalar, operator-=, 0.0f, -4.0f, 1.0f)

TEST_SCALAR_ASSIGNMENT(BinaryAssignment_multiplies_scalar, operator*=, 25.0f, 5.0f, 30.0f)

TEST_SCALAR_ASSIGNMENT(BinaryAssignment_divides_scalar, operator/=, 1.0f, 0.2f, 1.2f)

TEST_VECTOR_BINARY_MATH(BinaryMath_min_vector, min, 2.5f, 1.0f, 3.0f)

TEST_VECTOR_BINARY_MATH(BinaryMath_max_vector, max, 5.0f, 1.0f, 6.0f)

TEST_SCALAR_MATH(BinaryMath_min_scalar, min, 5.0f, 1.0f, 5.0f)

TEST_SCALAR_MATH(BinaryMath_max_scalar, max, 5.0f, 5.0f, 6.0f)

TEST_SCALAR_MATH_SWAPPED(BinaryMath_min_scalar_swapped, min, 5.0f, 1.0f, 5.0f)

TEST_SCALAR_MATH_SWAPPED(BinaryMath_max_scalar_swapped, max, 5.0f, 5.0f, 6.0f)

TEST_REDUCTION_VECTOR(Reduction_dot_vector, dot, 31.5f)

TEST_REDUCTION_SCALAR(Reduction_dot_scalar, dot, 60.0f)

TEST_F(VectorTest, CopyConstructor)
{
    VectorT copy(m_vector1);
}

TEST_F(VectorTest, MoveConstructor)
{
    VectorT copy(m_vector1);
    VectorT move(std::move(copy));
}

TEST_F(VectorTest, CopyAssignment)
{
    VectorT copy = m_vector1;
    EXPECT_EQ(copy.at(0), m_vector1.at(0));
    EXPECT_EQ(copy.at(1), m_vector1.at(1));
    EXPECT_EQ(copy.at(2), m_vector1.at(2));
}

TEST_F(VectorTest, MoveAssignment)
{
    VectorT copy(m_vector1);
    VectorT move = std::move(copy);
    EXPECT_EQ(move.at(0), m_vector1.at(0));
    EXPECT_EQ(move.at(1), m_vector1.at(1));
    EXPECT_EQ(move.at(2), m_vector1.at(2));
}

TEST_F(VectorTest, UnaryOperator_negate)
{
    // Why does VC12 crash on -m_vector2?
    auto result = m_vector2.operator-();
    EXPECT_EQ(-5.0f, result.at(0));
    EXPECT_EQ(-1.0f, result.at(1));
    EXPECT_EQ(-6.0f, result.at(2));
}

TEST_F(VectorTest, Norm)
{
    auto result = m_vector2.norm();
    EXPECT_FLOAT_EQ(7.87400787f, result);
}

TEST_F(VectorTest, Normalize)
{
    m_vector2.normalize();
    EXPECT_FLOAT_EQ(0.635000635f, m_vector2.at(0));
    EXPECT_FLOAT_EQ(0.127000127f, m_vector2.at(1));
    EXPECT_FLOAT_EQ(0.762000762f, m_vector2.at(2));
}

TEST_F(VectorTest, Any_allFalse)
{
    auto vector = make_vector<bool>(false, false, false, false);
    EXPECT_FALSE(vector.any());
}

TEST_F(VectorTest, Any_atLeast1True)
{
    auto vector = make_vector<bool>(false, false, true, false);
    EXPECT_TRUE(vector.any());
}

TEST_F(VectorTest, Any_allTrue)
{
    auto vector = make_vector<bool>(true, true, true, true);
    EXPECT_TRUE(vector.all());
}

TEST_F(VectorTest, All_atLeast1False)
{
    auto vector = make_vector<bool>(true, true, true, false);
    EXPECT_FALSE(vector.all());
}

TEST_F(VectorTest, Sum)
{
    auto sum = m_vector1.sum();
    EXPECT_EQ(6.5f, sum);
}

TEST_F(VectorTest, Prod)
{
    auto prod = m_vector1.prod();
    EXPECT_EQ(7.5f, prod);
}

TEST_F(VectorTest, assignmentOperatorDifferentTypes)
{
    /// @note: The Matrix returned by mapContinuous is of a type StorageT = StorageMapped.
    m_vector1.template mapContinuous<>() = m_vector2;
    EXPECT_TRUE((m_vector1 == m_vector2).all());
}

TEST_F(VectorTest, make_map_vector)
{
    auto vector = make_vector<TestType>(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
    auto mappedVector = make_map_vector<1, 3>(vector);
    using MappedType = std::decay<decltype(mappedVector)>::type;
    EXPECT_EQ(vector.data() + 1, mappedVector.data());
    EXPECT_EQ(3, static_cast<int>(MappedType::Size));
    EXPECT_EQ(3, mappedVector.at(1));
}

class MatrixTest
        : public ::testing::Test
{
public:

    enum
    {
        RowsTest = 2
    };
    enum
    {
        ColsTest = 3
    };
    enum
    {
        ColsTest2 = 4
    };
    using TestType = float;
    using MatrixT1 = Matrix<RowsTest, ColsTest, TestType>;
    using MatrixT2 = Matrix<ColsTest, ColsTest2, TestType>;

    void SetUp() override
    {
        m_matrix1 = make_matrix<RowsTest, ColsTest, TestType>(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        m_matrix2 = make_matrix<ColsTest, ColsTest2, TestType>(
                12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    }

protected:
    MatrixT1 m_matrix1;
    MatrixT2 m_matrix2;
    Vector<3, float> m_saveRes;
    float3 m_saveRes2;
};

TEST_F(MatrixTest, transpose_size)
{
    using transpose = std::decay<decltype(m_matrix1.transpose())>::type;
    using matrix1 = std::decay<decltype(m_matrix1)>::type;
    EXPECT_EQ(static_cast<int>(matrix1::Rows), static_cast<int>(transpose::Cols));
    EXPECT_EQ(static_cast<int>(matrix1::Cols), static_cast<int>(transpose::Rows));
}

TEST_F(MatrixTest, transpose_elem01)
{
    auto transpose = m_matrix1.transpose();
    EXPECT_EQ(m_matrix1.at(0, 1), transpose.at(1, 0));
}

TEST_F(MatrixTest, transpose_onMappedElem01)
{
    auto transpose = static_cast<const MatrixT1>(m_matrix1).mapContinuous<>().transpose();
    EXPECT_EQ(m_matrix1.at(0, 1), transpose.at(1));
}

TEST_F(MatrixTest, matMul_size)
{
    using product = std::decay<decltype(m_matrix1.mul(m_matrix2))>::type;
    EXPECT_EQ(static_cast<int>(MatrixT1::Rows), static_cast<int>(product::Rows));
    EXPECT_EQ(static_cast<int>(MatrixT2::Cols), static_cast<int>(product::Cols));
}

TEST_F(MatrixTest, matMul_elemValue02)
{
    auto result = m_matrix1.mul(m_matrix2);
    auto val02 = TestType(28.0f);
    EXPECT_EQ(val02, result.at(0, 2));
}

TEST_F(MatrixTest, matMul_elemValue11)
{
    auto result = m_matrix1.mul(m_matrix2);
    auto val11 = TestType(97.0f);
    EXPECT_EQ(val11, result.at(1, 1));
}

TEST_F(MatrixTest, mapContinuous_verifyPointerToData)
{
    auto map = m_matrix1.mapContinuous<2, 5>();
    EXPECT_EQ(&m_matrix1.at(2), &map.at(0));
}

TEST_F(MatrixTest, mapContinuous_verifyValues)
{
    auto map = m_matrix1.mapContinuous<2, 5>();

    auto expectedElem2 = TestType(3.5);
    auto expectedElem4 = TestType(2.5);

    // We mapped starting at 2, so the map index is 2 "behind". 
    map.at(0) = expectedElem2;
    map.at(2) = expectedElem4;

    EXPECT_EQ(expectedElem2, m_matrix1.at(2));
    EXPECT_EQ(expectedElem4, m_matrix1.at(4));
}

TEST_F(MatrixTest, cast)
{
    using CastType = int;

    const auto matrixCast = m_matrix1.cast<CastType>();

    // Check that the first and last elements are actually equal.
    EXPECT_EQ(static_cast<CastType>(m_matrix1.at(0, 0)), matrixCast.at(0, 0));
    EXPECT_EQ(static_cast<CastType>(m_matrix1.at(1, 2)), matrixCast.at(1, 2));
}

size_t NUM_ITER = 1000000;

TEST_F(MatrixTest, CUtils_Plus)
{
    auto result = make_float3(0.0f, 0.0f, 0.0f);
    for(size_t i = 0; i < NUM_ITER; ++i)
    {
        auto add = make_float3(1.0f, 1.0f, 1.0f);
        result += add;
    }
    m_saveRes2 = result;
}

TEST_F(MatrixTest, Vector_plus)
{
    auto result = make_vector<float>(0.0f, 0.0f, 0.0f);
    for(size_t i = 0; i < NUM_ITER; ++i)
    {
        auto add = make_vector<float>(1.0f, 1.0f, 1.0f);
        result += add;
    }
    m_saveRes = result;
}
}
