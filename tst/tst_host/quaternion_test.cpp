#include <random>
#include "inc/quaternion.h"
#include "gtest/gtest.h"

using namespace smalls;

namespace test
{
class QuaternionTest : public ::testing::Test
{
public:

    using ScalarType = float;
    using QuaternionType = Quaternion<ScalarType>;
    using EulerType = EulerAngles<ScalarType>;

    void SetUp() override
    {
        m_quat2 = QuaternionType::fromEuler(MakeRandomVector3());
        m_quat1 = QuaternionType::fromEuler(MakeRandomVector3());
    }

    static EulerType MakeRandomVector3()
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_real_distribution<ScalarType> dist(0, 10);

        return EulerType(dist(engine), dist(engine), dist(engine));
    }

protected:
    QuaternionType m_quat1, m_quat2;
};

TEST_F(QuaternionTest, constructorDefault)
{
    QuaternionType quat{};
    EXPECT_TRUE(quat == QuaternionType(0.0f, 0.0f, 0.0f, 1.0f));
}

TEST_F(QuaternionTest, constructorAll)
{
    QuaternionType quat(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(ScalarType(1.0f), quat.x());
    EXPECT_EQ(ScalarType(2.0f), quat.y());
    EXPECT_EQ(ScalarType(3.0f), quat.z());
    EXPECT_EQ(ScalarType(4.0f), quat.w());
}

TEST_F(QuaternionTest, nonConstAccessors)
{
    QuaternionType quat(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(ScalarType(1.0f), quat.x());
    EXPECT_EQ(ScalarType(2.0f), quat.y());
    EXPECT_EQ(ScalarType(3.0f), quat.z());
    EXPECT_EQ(ScalarType(4.0f), quat.w());
}

TEST_F(QuaternionTest, constAccessors)
{
    const QuaternionType quat(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(ScalarType(1.0f), quat.x());
    EXPECT_EQ(ScalarType(2.0f), quat.y());
    EXPECT_EQ(ScalarType(3.0f), quat.z());
    EXPECT_EQ(ScalarType(4.0f), quat.w());
}

TEST_F(QuaternionTest, assignmentPlusScalar)
{
    auto original = m_quat1;
    auto scalar = ScalarType(3.0f);
    m_quat1 += scalar;
    EXPECT_EQ(original.x() + scalar, m_quat1.x());
    EXPECT_EQ(original.y() + scalar, m_quat1.y());
    EXPECT_EQ(original.z() + scalar, m_quat1.z());
    EXPECT_EQ(original.w() + scalar, m_quat1.w());
}

TEST_F(QuaternionTest, equality)
{
    m_quat1 = m_quat2;
    EXPECT_EQ(m_quat1, m_quat2);
}

TEST_F(QuaternionTest, inequality)
{
    m_quat1 = m_quat2 + 1.0f;
    EXPECT_NE(m_quat1, m_quat2);
}

TEST_F(QuaternionTest, multiplication)
{
    ///--------------------------------------------------------------------------------------------
    /// @note: Results is taken from here:
    /// http://www.math.wayne.edu/~isaksen/Teaching/Electronic/quaternion.pdf
    ///--------------------------------------------------------------------------------------------
    QuaternionType quaternion1(2.0f, 5.0f, 4.0f, 3.0f);
    QuaternionType quaternion2(5.0f, 3.0f, 1.0f, 4.0f);
    QuaternionType expected(16.0f, 47.0f, 0.f, -17.0f);

    QuaternionType actual = quaternion1*quaternion2;
    EXPECT_EQ(expected, actual);
}

TEST_F(QuaternionTest, fromToEulerAngles)
{
    /// @note: Hard-code a value because the Quaternion -> Euler transform is not unique.
    const auto expectedEulerAngles = EulerAngles<ScalarType>(-1.13926, -1.56806, -0.962517);
    const auto quat = QuaternionType::fromEuler(expectedEulerAngles);
    const auto actualEulerAngles = quat.toEuler();

    EXPECT_FLOAT_EQ(expectedEulerAngles.at(0), actualEulerAngles.at(0));
    EXPECT_FLOAT_EQ(expectedEulerAngles.at(1), actualEulerAngles.at(1));
    EXPECT_FLOAT_EQ(expectedEulerAngles.at(2), actualEulerAngles.at(2));
}

TEST_F(QuaternionTest, toFromEulerAngles)
{
    /// @note: Hard-code a value because the Quaternion -> Euler transform is not unique.
    const QuaternionType expectedQuat(0.626551f, -0.556052f, -0.300891f, 0.455746f);
    const auto eulerAngles = expectedQuat.toEuler();
    const auto actualQuat = QuaternionType::fromEuler(eulerAngles);

    const ScalarType eps(1.0e-5f);
    EXPECT_NEAR(expectedQuat.x(), actualQuat.x(), eps);
    EXPECT_NEAR(expectedQuat.y(), actualQuat.y(), eps);
    EXPECT_NEAR(expectedQuat.z(), actualQuat.z(), eps);
    EXPECT_NEAR(expectedQuat.w(), actualQuat.w(), eps);
}

TEST_F(QuaternionTest, point_transform)
{
    const auto rotationMatrix = m_quat1.toRotation();
    const auto point = MakeRandomVector3();

    const auto expectedTrans = rotationMatrix.mul(point);
    const auto actualTrans = m_quat1.mul(point);

    const ScalarType eps(1.0e-5f);
    EXPECT_NEAR(expectedTrans.at(0), actualTrans.at(0), eps);
    EXPECT_NEAR(expectedTrans.at(1), actualTrans.at(1), eps);
    EXPECT_NEAR(expectedTrans.at(2), actualTrans.at(2), eps);
}
}
