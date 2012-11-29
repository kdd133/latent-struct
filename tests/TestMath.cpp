#define BOOST_TEST_DYN_LINK

#include "LogWeight.h"
#include "RealWeight.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>

BOOST_AUTO_TEST_CASE(testMath)
{
  BOOST_CHECK(exp(LogWeight(0)) == RealWeight(0));
  BOOST_CHECK(log(RealWeight(0)) == LogWeight(0));
  BOOST_CHECK(exp(LogWeight(1)) == RealWeight(1));
  BOOST_CHECK(log(RealWeight(1)) == LogWeight(1));
}
