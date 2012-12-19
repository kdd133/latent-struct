#define BOOST_TEST_DYN_LINK

#include "Parameters.h"
#include "Ublas.h"
#include "Utility.h"

#include <boost/shared_array.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;

BOOST_AUTO_TEST_CASE(testParameters)
{
  const int dw = 4, du = 5, len = dw + du; 
  Parameters theta(dw, du);
  BOOST_CHECK(theta.hasU());
  BOOST_CHECK_EQUAL(theta.w.getDim(), dw);
  BOOST_CHECK_EQUAL(theta.u.getDim(), du);
  BOOST_CHECK_EQUAL(theta.getTotalDim(), len);
  
  // Test the setWeights method.
  shared_array<double> values = Utility::generateGaussianSamples(len, 0, 3, 0);
  theta.setWeights(values.get(), len);
  for (int i = 0; i < len; ++i) {
    BOOST_CHECK_EQUAL(theta.getWeight(i), values[i]);
  }
  
  // Test the add method.
  shared_array<double> toAdd = Utility::generateGaussianSamples(len, 0, 3, 1);
  for (int i = 0; i < len; ++i) {
    theta.add(i, toAdd[i]);
    BOOST_CHECK_CLOSE(theta.getWeight(i), values[i] + toAdd[i], 1e-8);
  }

  // Test the squaredL2Norm method.
  double sqNormL2 = 0;
  for (int i = 0; i < len; ++i) {
    const double v = values[i] + toAdd[i];
    sqNormL2 += v * v;
  }
  BOOST_CHECK_CLOSE(theta.squaredL2Norm(), sqNormL2, 1e-8);
  
  // Test the innerProd method.
  RealVec fv(len);
  RealVec thetaFv(len);
  shared_array<double> fvVals = Utility::generateGaussianSamples(len, 0, 1, 2);
  for (int i = 0; i < len; ++i) {
    fv(i) = fvVals[i];
    thetaFv(i) = values[i] + toAdd[i];
  }  
  BOOST_CHECK_CLOSE(theta.innerProd(fv), inner_prod(fv, thetaFv), 1e-8);

  // Test the setParams method.
  Parameters theta2(dw, du);
  shared_array<double> values2 = Utility::generateGaussianSamples(len, 0, 1, 3);
  theta2.setWeights(values2.get(), len);
  theta.setParams(theta2);
  for (int i = 0; i < len; ++i) {
    BOOST_CHECK_EQUAL(theta2.getWeight(i), values2[i]);
  }

  // Test the zero method.
  theta.zero();
  BOOST_CHECK_EQUAL(theta.innerProd(fv), 0);
}
