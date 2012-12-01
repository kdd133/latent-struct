#define BOOST_TEST_DYN_LINK

#include "LogWeight.h"
#include "RealWeight.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>

BOOST_AUTO_TEST_CASE(testWeight)
{
  // Check that the relationships between log and real values hold.
  {
    BOOST_CHECK_EQUAL(exp(LogWeight(0)), RealWeight(0));
    BOOST_CHECK_EQUAL(log(RealWeight(0)), LogWeight(0));
    BOOST_CHECK_EQUAL(exp(LogWeight(1)), RealWeight(1));
    BOOST_CHECK_EQUAL(log(RealWeight(1)), LogWeight(1));
  }
  
  // Check that multiplication by a negated value equals division by the
  // original value.
  {
    LogWeight mult = LogWeight(2) * (-LogWeight(3));
    LogWeight div = LogWeight(2) / LogWeight(3);
    BOOST_CHECK_CLOSE((double)mult, (double)div, 1e-8);
  }
  
  // Check that multiplication by a negated value equals division by the
  // original value, this time using shorthand arithmetic operator assignments. 
  {
    LogWeight mult = LogWeight(2);
    mult *= (-LogWeight(3));
    LogWeight div = LogWeight(2);
    div /= LogWeight(3);
    BOOST_CHECK_CLOSE((double)mult, (double)div, 1e-8);
  }
  
  // Check that addition by a negated value equals subtraction by the original
  // value.
  {
    LogWeight add = LogWeight(2) + (-LogWeight(3));
    LogWeight sub = LogWeight(2) - LogWeight(3);
    BOOST_CHECK_CLOSE((double)add, (double)sub, 1e-8);
  }
  
  // Check that addition by a negated value equals subtraction by the original
  // value, this time using shorthand arithmetic operator assignments.
  {
    LogWeight add = LogWeight(2);
    add += (-LogWeight(3));
    LogWeight sub = LogWeight(2);
    sub -= LogWeight(3);
    BOOST_CHECK_CLOSE((double)add, (double)sub, 1e-8);
  }
}
