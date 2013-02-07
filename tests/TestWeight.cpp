#define BOOST_TEST_DYN_LINK

#include "LogWeight.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>

BOOST_AUTO_TEST_CASE(testWeight)
{
  const LogWeight zero;
  const LogWeight one(1);
  const LogWeight two(2);
  const LogWeight three(3);
  
  // Check that the identities between log and real values hold.
  {
    BOOST_CHECK_EQUAL(exp(zero), 0);
    BOOST_CHECK_EQUAL(log(0), zero);
    BOOST_CHECK_EQUAL(exp(one), 1);
    BOOST_CHECK_EQUAL(log(1), one);
    BOOST_CHECK_EQUAL(zero, LogWeight(0));
  }
  
  // Test multiplication.
  {
    BOOST_CHECK_CLOSE((double)(two * three), 1.79175, 1e-3);
    BOOST_CHECK_CLOSE((double)(three * two), 1.79175, 1e-3);
    BOOST_CHECK_EQUAL((double)(zero * two), (double)zero);
    BOOST_CHECK_EQUAL((double)(two * zero), (double)zero);
    BOOST_CHECK_EQUAL((double)(zero * zero), (double)zero);
  }
  
  // Test division.
  {
    BOOST_CHECK_CLOSE((double)(two / three), -0.405465, 1e-3);
    BOOST_CHECK_CLOSE((double)(three / two), 0.405465, 1e-3);
    BOOST_CHECK_EQUAL((double)(zero / two), (double)zero);
    // TODO: Should division by zero return NaN, +Inf, or throw an exception?
  }
  
  // Test addition.
  {
    BOOST_CHECK_CLOSE((double)(two + three), 1.60943, 1e-3);
    BOOST_CHECK_CLOSE((double)(three + two), 1.60943, 1e-3);
    BOOST_CHECK_CLOSE((double)(zero + three), (double)three, 1e-3);
    BOOST_CHECK_CLOSE((double)(three + zero), (double)three, 1e-3);
  }
}
