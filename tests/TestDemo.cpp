#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDemo
#include <boost/test/unit_test.hpp>

#include "WeightVector.h"
 
BOOST_AUTO_TEST_CASE(universeInOrder)
{
  WeightVector w;
  BOOST_CHECK(w.getDim() == 0);
}

