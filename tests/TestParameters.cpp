#define BOOST_TEST_DYN_LINK

#include "Parameters.h"
#include "Ublas.h"
#include "Utility.h"

#include <boost/shared_array.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost;

BOOST_AUTO_TEST_CASE(testParameters)
{
  // Run some tests with a w-u model (no shared parameters).
  {
    const int dw = 4, du = 5, len = dw + du; 
    Parameters theta(dw, du);
    
    BOOST_CHECK(theta.hasU());
    BOOST_CHECK(!theta.hasSharedW());
    BOOST_CHECK(!theta.hasSharedU());
    BOOST_CHECK_EQUAL(theta.w.getDim(), dw);
    BOOST_CHECK_EQUAL(theta.u.getDim(), du);
    BOOST_CHECK_EQUAL(theta.getDimWU(), len);
    BOOST_CHECK_EQUAL(theta.getDimTotal(), len);    
    BOOST_CHECK_EQUAL(theta.indexW(), 0);
    BOOST_CHECK_EQUAL(theta.indexU(), dw);
    BOOST_CHECK_EQUAL(theta.indexSharedW(), -1);
    BOOST_CHECK_EQUAL(theta.indexSharedU(), -1);
    
    // Test the setWeights method.
    shared_array<double> values = Utility::generateGaussianSamples(len, 0, 3,
        0);
    theta.setWeights(values.get(), len);
    for (int i = 0; i < len; ++i) {
      BOOST_CHECK_EQUAL(theta[i], values[i]);
    }
    
    // Test the add method.
    shared_array<double> toAdd = Utility::generateGaussianSamples(len, 0, 3, 1);
    for (int i = 0; i < len; ++i) {
      theta.add(i, toAdd[i]);
      BOOST_CHECK_CLOSE(theta[i], values[i] + toAdd[i], 1e-8);
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
    shared_array<double> fvVals = Utility::generateGaussianSamples(len, 0, 1,
        2);
    for (int i = 0; i < len; ++i) {
      fv(i) = fvVals[i];
      thetaFv(i) = values[i] + toAdd[i];
    }  
    BOOST_CHECK_CLOSE(theta.innerProd(fv), inner_prod(fv, thetaFv), 1e-8);
  
    // Test the setParams method.
    Parameters theta2(dw, du);
    shared_array<double> values2 = Utility::generateGaussianSamples(len, 0, 1,
        3);
    theta2.setWeights(values2.get(), len);
    theta.setParams(theta2);
    for (int i = 0; i < len; ++i)
      BOOST_CHECK_EQUAL(theta2[i], values2[i]);
  
    // Test the zero method.
    theta.zero();
    BOOST_CHECK_EQUAL(theta.innerProd(fv), 0);
  }
  
  // Run some tests with a w-u model, this time with shared parameters.
  {
    const int dw = 4, du = 5, len = dw*2 + du*2; 
    Parameters theta(dw, du);
    theta.shared_w.reAlloc(dw);
    theta.shared_u.reAlloc(du);
    
    BOOST_CHECK(theta.hasU());
    BOOST_CHECK(theta.hasSharedW());
    BOOST_CHECK(theta.hasSharedU());
    BOOST_CHECK_EQUAL(theta.w.getDim(), dw);
    BOOST_CHECK_EQUAL(theta.u.getDim(), du);
    BOOST_CHECK_EQUAL(theta.getDimWU(), dw + du);
    BOOST_CHECK_EQUAL(theta.getDimTotal(), len);    
    BOOST_CHECK_EQUAL(theta.indexW(), 0);
    BOOST_CHECK_EQUAL(theta.indexU(), dw);
    BOOST_CHECK_EQUAL(theta.indexSharedW(), dw + du);
    BOOST_CHECK_EQUAL(theta.indexSharedU(), dw + du + dw);
    
    // Test the setWeights method.
    shared_array<double> values = Utility::generateGaussianSamples(len, 0, 3,
        0);
    theta.setWeights(values.get(), len);
    for (int i = 0; i < len; ++i) {
      BOOST_CHECK_EQUAL(theta[i], values[i]);
    }
    
    // Test the add method.
    shared_array<double> toAdd = Utility::generateGaussianSamples(len, 0, 3, 1);
    for (int i = 0; i < len; ++i) {
      theta.add(i, toAdd[i]);
      BOOST_CHECK_CLOSE(theta[i], values[i] + toAdd[i], 1e-8);
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
    shared_array<double> fvVals = Utility::generateGaussianSamples(len, 0, 1,
        2);
    for (int i = 0; i < len; ++i) {
      fv(i) = fvVals[i];
      thetaFv(i) = values[i] + toAdd[i];
    }  
    BOOST_CHECK_CLOSE(theta.innerProd(fv), inner_prod(fv, thetaFv), 1e-8);
  
    // Test the setParams method.
    Parameters theta2(dw, du);
    theta2.shared_w.reAlloc(dw);
    theta2.shared_u.reAlloc(du);
    shared_array<double> values2 = Utility::generateGaussianSamples(len, 0, 1,
        3);
    theta2.setWeights(values2.get(), len);
    theta.setParams(theta2);
    for (int i = 0; i < len; ++i)
      BOOST_CHECK_EQUAL(theta2[i], values2[i]);
  
    // Test the zero method.
    theta.zero();
    BOOST_CHECK_EQUAL(theta.innerProd(fv), 0);
  }
}
