#include "inc/vector.h"
#include "inc/cuda_types_utils.h"
#include "gtest/gtest.h"
#include <array>

using namespace smalls;

class VectorTTest : public ::testing::Test
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
TEST_F(VectorTTest, NAME)\
{\
    auto result = m_vector2.OPERATOR(m_vector1);\
    EXPECT_NEAR(RESULT1, result.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, result.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, result.data[2], m_eps);\
}

#define TEST_SCALAR_OPERATOR(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    auto result = m_vector2.OPERATOR(m_scalar);\
    EXPECT_NEAR(RESULT1, result.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, result.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, result.data[2], m_eps);\
}

#define TEST_VECTOR_ASSIGNMENT(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    m_vector2.OPERATOR(m_vector1);\
    EXPECT_NEAR(RESULT1, m_vector2.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, m_vector2.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, m_vector2.data[2], m_eps);\
}

#define TEST_SCALAR_ASSIGNMENT(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    m_vector2.OPERATOR(m_scalar);\
    EXPECT_NEAR(RESULT1, m_vector2.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, m_vector2.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, m_vector2.data[2], m_eps);\
}

#define TEST_VECTOR_BINARY_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_vector1);\
    EXPECT_NEAR(RESULT1, result.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, result.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, result.data[2], m_eps);\
}

#define TEST_SCALAR_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    auto result = OPERATOR(m_vector2, m_scalar);\
    EXPECT_NEAR(RESULT1, result.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, result.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, result.data[2], m_eps);\
}

#define TEST_VECTOR_UNARY_MATH(NAME, OPERATOR, RESULT1, RESULT2, RESULT3)\
TEST_F(VectorTTest, NAME)\
{\
    auto result = OPERATOR(m_vector2);\
    EXPECT_NEAR(RESULT1, result.data[0], m_eps);\
    EXPECT_NEAR(RESULT2, result.data[1], m_eps);\
    EXPECT_NEAR(RESULT3, result.data[2], m_eps);\
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
TEST_SCALAR_ASSIGNMENT(BinaryAssignment_plus_scalar, operator+=, 10.0f, 6.0f, 11.0f)
TEST_SCALAR_ASSIGNMENT(BinaryAssignment_minus_scalar, operator-=, 0.0f, -4.0f, 1.0f)
TEST_SCALAR_ASSIGNMENT(BinaryAssignment_multiplies_scalar, operator*=, 25.0f, 5.0f, 30.0f)
TEST_SCALAR_ASSIGNMENT(BinaryAssignment_divides_scalar, operator/=, 1.0f, 0.2f, 1.2f)

TEST_VECTOR_BINARY_MATH(BinaryMath_min_vector, math::min, 2.5f, 1.0f, 3.0f)
TEST_VECTOR_BINARY_MATH(BinaryMath_max_vector, math::max, 5.0f, 1.0f, 6.0f)
TEST_SCALAR_MATH(BinaryMath_min_scalar, math::min, 5.0f, 1.0f, 5.0f)
TEST_SCALAR_MATH(BinaryMath_max_scalar, math::max, 5.0f, 5.0f, 6.0f)

TEST_VECTOR_UNARY_MATH(UnaryMath_floor, math::floor, 5.0f, 1.0f, 6.0f) // Bad tests - should have decimals
TEST_VECTOR_UNARY_MATH(UnaryMath_ceil, math::ceil, 5.0f, 1.0f, 6.0f) // Bad tests - should have decimals

TEST_F(VectorTTest, CopyConstructor)
{
    VectorT copy(m_vector1);
}

TEST_F(VectorTTest, MoveConstructor)
{
    VectorT copy(m_vector1);
    VectorT move(std::move(copy));
}

TEST_F(VectorTTest, CopyAssignment)
{
    VectorT copy = m_vector1;
}

TEST_F(VectorTTest, MoveAssignment)
{
    VectorT copy(m_vector1);
    VectorT move = std::move(copy);
}

TEST_F(VectorTTest, UnaryOperator_negate)
{
    // Why does VC12 crash on -m_vector2?
    auto result = m_vector2.operator-();
    EXPECT_EQ(-5.0f, result.data[0]);
    EXPECT_EQ(-1.0f, result.data[1]);
    EXPECT_EQ(-6.0f, result.data[2]);
}

size_t NUM_ITER = 10000000;
TEST_F(VectorTTest, Perf_Vector_plus)
{
    auto result = make_vector<float>(0.0f, 0.0f, 0.0f);
    for (auto i = 0; i < NUM_ITER; ++i)
    {
        auto add = make_vector<float>(1.0f, 1.0f, 1.0f);
        result += add;
    }
}

TEST_F(VectorTTest, Perf_CUtils_PLus)
{
    auto result = make_float3(0.0f, 0.0f, 0.0f);
    for (auto i = 0; i < NUM_ITER; ++i)
    {
        auto add = make_float3(1.0f, 1.0f, 1.0f);
        result += add;
    }
}