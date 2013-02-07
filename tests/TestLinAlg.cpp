#define BOOST_TEST_DYN_LINK

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "LogWeight.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace boost;
using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE(testLinAlg)
{
  using namespace boost::numeric::ublas;
  
  LogWeight value;
  
  mapped_vector<LogWeight> v(3, 2);
  v(0) = LogWeight(3);
  v(2) = LogWeight(1);
  
  mapped_vector<LogWeight> r = v + v;
  value = r(0); BOOST_CHECK_CLOSE((double)value, 1.79175946922806, 1e-8);
  value = r(1); BOOST_CHECK_EQUAL((double)value, LogWeight());
  value = r(2); BOOST_CHECK_CLOSE((double)value, 0.69314718055994, 1e-8);
  
  mapped_vector<LogWeight> s = v * LogWeight(2);
  value = s(0); BOOST_CHECK_CLOSE((double)value, 1.79175946922806, 1e-8);
  value = s(1); BOOST_CHECK_EQUAL((double)value, LogWeight());
  value = s(2); BOOST_CHECK_CLOSE((double)value, 0.69314718055994, 1e-8);
}
