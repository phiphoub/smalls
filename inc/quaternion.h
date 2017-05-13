#ifndef SMALLS_QUATERNION_H
#define SMALLS_QUATERNION_H

#include "inc/vector.h"

namespace smalls
{
#define SCALAR_ASSIGNMENT(operator) \
SPECIFIER inline Quaternion operator(ScalarT scalar) \
{ static_cast<Base*>(this)->operator(scalar); return *this; }

#define OPERATOR_FROM_ASSIGNEMNT(operator) \
template<typename QuaternionT> \
SPECIFIER inline Quaternion operator(QuaternionT&& other) const \
{ Quaternion tmp(*this); tmp.operator##=(std::forward<QuaternionT>(other)); return tmp; }

///--------------------------------------------------------------------------------------------
/// @brief Define a EulerAngles type.
///--------------------------------------------------------------------------------------------
template<typename ScalarT> using EulerAngles = Vector<3, ScalarT>;

template<typename ScalarT>
class Quaternion : private Vector<4, ScalarT>
{
public:
    using Base = Vector<4, ScalarT>;

    ///--------------------------------------------------------------------------------------------
    /// @brief Default constructor.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER Quaternion() : Quaternion(Identity()) {};

    ///--------------------------------------------------------------------------------------------
    /// @brief Construct a quaternion from a Base vector.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER explicit Quaternion(const Base& base) : Base(base) {}

    ///--------------------------------------------------------------------------------------------
    /// @brief Construct a quaternion from its values.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER explicit Quaternion(ScalarT x, ScalarT y, ScalarT z, ScalarT w) : Base(x, y, z, w) {}

    ///--------------------------------------------------------------------------------------------
    /// @brief Non-const accessor to the quaternion x-value. x-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT& x() { return Base::at(0); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Non-const accessor to the quaternion y-value. y-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT& y() { return Base::at(1); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Non-const accessor to the quaternion z-value. z-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT& z() { return Base::at(2); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Non-const accessor to the quaternion w-value. Real part of the quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT& w() { return Base::at(3); } // real part

    ///--------------------------------------------------------------------------------------------
    /// @brief Const accessor to the quaternion x-value. x-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT x() const { return Base::at(0); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Const accessor to the quaternion y-value. y-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT y() const { return Base::at(1); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Const accessor to the quaternion z-value. z-component of the vectorial part.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT z() const { return Base::at(2); }

    ///--------------------------------------------------------------------------------------------
    /// @brief Const accessor to the quaternion w-value. Real part of the quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT w() const { return Base::at(3); } // real part

    ///--------------------------------------------------------------------------------------------
    /// @note Generates the scalar +, - , *, / operators.
    ///--------------------------------------------------------------------------------------------
    SCALAR_ASSIGNMENT(operator+=);
    SCALAR_ASSIGNMENT(operator-=);
    SCALAR_ASSIGNMENT(operator*=);
    SCALAR_ASSIGNMENT(operator/=);

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion to quaternion equality operator.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline bool operator==(const Quaternion& other) const
    {
        return smalls::operator==(
                *static_cast<const Base*>(this), *static_cast<const Base*>(&other)).all();
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion to quaternion inequality operator.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline bool operator!=(const Quaternion& other) const
    {
        return smalls::operator!=(
                *static_cast<const Base*>(this), *static_cast<const Base*>(&other)).all();
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion to quaternion addition assignment operator.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline Quaternion& operator+=(const Quaternion& other)
    {
        static_cast<Base*>(this)->operator+=(*static_cast<const Base*>(&other));
        return *this;
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion to quaternion subtraction assignment operator.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline Quaternion& operator-=(const Quaternion& other)
    {
        static_cast<Base*>(this)->operator-=(*static_cast<const Base*>(&other));
        return *this;
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion to quaternion multiplication assignment operator.
    ///--------------------------------------------------------------------------------------------
    template<typename QuaternionT>
    SPECIFIER inline Quaternion& operator*=(QuaternionT&& other)
    {
        Quaternion result{};
        result.w() = (w() * other.w()) - (x() * other.x()) - (y() * other.y()) - (z() * other.z());
        result.x() = (w() * other.x()) + (x() * other.w()) + (y() * other.z()) - (z() * other.y());
        result.y() = (w() * other.y()) + (y() * other.w()) + (z() * other.x()) - (x() * other.z());
        result.z() = (w() * other.z()) + (z() * other.w()) + (x() * other.y()) - (y() * other.x());
        std::swap(*this, result);
        return *this;
    }

    ///--------------------------------------------------------------------------------------------
    /// @note Generates the quaternion +, - , *, / operators.
    ///--------------------------------------------------------------------------------------------
    OPERATOR_FROM_ASSIGNEMNT(operator+);
    OPERATOR_FROM_ASSIGNEMNT(operator-);
    OPERATOR_FROM_ASSIGNEMNT(operator*);
    OPERATOR_FROM_ASSIGNEMNT(operator/);

    ///--------------------------------------------------------------------------------------------
    /// @brief Generates an Identity Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline static Quaternion Identity()
    {
        return Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Quaternion inverse.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline Quaternion inverse() const
    {
        return Quaternion(-x(), -y(), -z(), w());
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Calculates the norm of the Quaternions.
    /// @return The norm of the Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline ScalarT norm() const
    {
        return static_cast<const Base*>(this)->norm();
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Normalize a Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline void normalize()
    {
        return static_cast<Base*>(this)->normalize();
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Generates a Quaternion from Euler angles.
    /// @note https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
    /// @return A Quaternion representing a rotation for the given Euler angles.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline static Quaternion fromEuler(const EulerAngles<ScalarT>& euler)
    {
        const ScalarT& pitch = euler.at(0);
        const ScalarT& roll = euler.at(1);
        const ScalarT& yaw = euler.at(2);

        Quaternion quaternion;
        const ScalarT t0 = std::cos(yaw * ScalarT(0.5f));
        const ScalarT t1 = std::sin(yaw * ScalarT(0.5f));
        const ScalarT t2 = std::cos(roll * ScalarT(0.5f));
        const ScalarT t3 = std::sin(roll * ScalarT(0.5f));
        const ScalarT t4 = std::cos(pitch * ScalarT(0.5f));
        const ScalarT t5 = std::sin(pitch * ScalarT(0.5f));

        quaternion.w() = t0 * t2 * t4 + t1 * t3 * t5;
        quaternion.x() = t0 * t3 * t4 - t1 * t2 * t5;
        quaternion.y() = t0 * t2 * t5 + t1 * t3 * t4;
        quaternion.z() = t1 * t2 * t4 - t0 * t3 * t5;
        return quaternion;
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Generates EulerAngles from a Quaternion.
    /// @note https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
    /// @return EulerAngle representing a rotation for the Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline EulerAngles<ScalarT> toEuler() const
    {
        const ScalarT ysqr = this->y() * this->y();

        // roll (x-axis rotation)
        const ScalarT t0 = ScalarT(2.0f) * (this->w() * this->x() + this->y() * this->z());
        const ScalarT t1 = ScalarT(1.0f) - ScalarT(2.0f) * (this->x() * this->x() + ysqr);
        const ScalarT roll = atan2(t0, t1);

        // pitch (y-axis rotation)
        ScalarT t2 = ScalarT(2.0f) * (this->w() * this->y() - this->z() * this->x());
        t2 = t2 > ScalarT(1.0f) ? ScalarT(1.0f) : t2;
        t2 = t2 < ScalarT(-1.0f) ? ScalarT(-1.0f) : t2;
        const ScalarT pitch = asin(t2);

        // yaw (z-axis rotation)
        const ScalarT t3 = ScalarT(2.0f) * (this->w() * this->z() + this->x() * this->y());
        const ScalarT t4 = ScalarT(1.0f) - ScalarT(2.0f) * (ysqr + this->z() * this->z());
        const ScalarT yaw = atan2(t3, t4);

        return EulerAngles<ScalarT>(pitch, roll, yaw);
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Generates a Rotation matrix from a Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline Matrix<3, 3, ScalarT> toRotation()
    {
        Matrix<3, 3, ScalarT> rotation;

        rotation.at(0,0) = 1.0f - 2.0f*this->y()*this->y() - 2.0f*this->z()*this->z();
        rotation.at(0,1) = 2.0f*this->x()*this->y() - 2.0f*this->z()*this->w();
        rotation.at(0,2) = 2.0f*this->x()*this->z() + 2.0f*this->y()*this->w();
        rotation.at(1,0) = 2.0f*this->x()*this->y() + 2.0f*this->z()*this->w();
        rotation.at(1,1) = 1.0f - 2.0f*this->x()*this->x() - 2.0f*this->z()*this->z();
        rotation.at(1,2) = 2.0f*this->z()*this->y() - 2.0f*this->x()*this->w();
        rotation.at(2,0) = 2.0f*this->x()*this->z() - 2.0f*this->y()*this->w();
        rotation.at(2,1) = 2.0f*this->z()*this->y() + 2.0f*this->x()*this->w();
        rotation.at(2,2) = 1.0f - 2.0f*this->x()*this->x() - 2.0f*this->y()*this->y();

        return rotation;
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Transforms a point by the rotation represented by the Quaternion.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER inline Vector<3, ScalarT> mul(const Vector<3, ScalarT>& point) const
    {
        const Quaternion pointInQuat(point.at(0), point.at(1), point.at(2), ScalarT(0.0f));
        const auto transformedPointInQuat = (*this)*pointInQuat*(this->inverse());

        return make_vector<ScalarT>(transformedPointInQuat.x(), transformedPointInQuat.y(), transformedPointInQuat.z());
    }

    ///--------------------------------------------------------------------------------------------
    /// @brief Get the quaternion underlying data.
    /// @note We provide a copy to ensure that consumers do not abuse the data.
    ///--------------------------------------------------------------------------------------------
    SPECIFIER Base data() const
    {
        return Base(this->x(), this->y(), this->z(), this->w());
    }
};
}

#endif //SMALLS_QUATERNION_H
