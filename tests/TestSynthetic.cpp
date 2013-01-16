#define BOOST_TEST_DYN_LINK

#include "Parameters.h"
#include "SyntheticData.h"
#include "Ublas.h"
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace synthetic;

BOOST_AUTO_TEST_CASE(testSynthetic)
{
  const size_t nx = 3, ny = 2, nz = 2;
  SparseRealVec x(nx);
  for (size_t i = 0; i < nx; ++i)
    x(i) = i;
    
  const size_t dim = nx*ny*nz;
  double weights[dim] = {7,8,11,1,9,2,6,3,12,10,4,5};
  for (size_t i = 0; i < dim; ++i)
    weights[i] /= (double)dim;
  Parameters theta(dim);
  theta.setWeights(weights, dim);

  // Test the log_sum_exp function.
  BOOST_CHECK_CLOSE(log_sum_exp(x), 2.4076059644443801, 1e-8);
  
  // Test the prob_x function.
  const SparseRealVec probs_yz = prob_x(theta, x, ny, nz);
  BOOST_CHECK_CLOSE(probs_yz[0], 0.4376522764387575, 1e-8);
  BOOST_CHECK_CLOSE(probs_yz[1], 0.1061399003986655, 1e-8);
  BOOST_CHECK_CLOSE(probs_yz[2], 0.3408439356034873, 1e-8);
  BOOST_CHECK_CLOSE(probs_yz[3], 0.1153638875590894, 1e-8);
  
  // Test the cumsum function.
  const SparseRealVec cumsum_x = cumsum(x);
  BOOST_CHECK_CLOSE(cumsum_x[0], 0, 1e-8);
  BOOST_CHECK_CLOSE(cumsum_x[1], 1, 1e-8);
  BOOST_CHECK_CLOSE(cumsum_x[2], 3, 1e-8);
  const SparseRealVec cumsum_probs = cumsum(probs_yz);
  BOOST_CHECK_CLOSE(cumsum_probs[0], 0.4376522764387575, 1e-8);
  BOOST_CHECK_CLOSE(cumsum_probs[1], 0.5437921768374229, 1e-8);
  BOOST_CHECK_CLOSE(cumsum_probs[2], 0.8846361124409103, 1e-8);
  BOOST_CHECK_CLOSE(cumsum_probs[3], 1.0, 1e-8);
  
  // Test the ind2sub function.
  int i = -1, j = -1;
  ind2sub(2, 3, 0, i, j);
  BOOST_CHECK_EQUAL(i, 0);
  BOOST_CHECK_EQUAL(j, 0);
  i = -1, j = -1;
  ind2sub(2, 3, 1, i, j);
  BOOST_CHECK_EQUAL(i, 1);
  BOOST_CHECK_EQUAL(j, 0);
  i = -1, j = -1;
  ind2sub(2, 3, 2, i, j);
  BOOST_CHECK_EQUAL(i, 0);
  BOOST_CHECK_EQUAL(j, 1);
}
