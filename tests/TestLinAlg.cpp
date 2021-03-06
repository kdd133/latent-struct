#define BOOST_TEST_DYN_LINK

// Some of these checks fail when using, e.g., LogWeight as the element type
// in ublas vector and matrix classes.
#define BOOST_UBLAS_TYPE_CHECK 0

#include "LogWeight.h"
#include "Ublas.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace boost;
using namespace boost::numeric::ublas;

BOOST_AUTO_TEST_CASE(testLinAlg)
{
  {
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
  
  {
    AccumRealMat A(3, 3);
    A(0,0) = -1.09263; A(0,1) = -0.43284; A(0,2) = 1.01652;
    A(1,0) = -0.43284; A(1,1) = -0.25514; A(1,2) = 1.86776;
    A(2,0) =  1.01652; A(2,1) =  1.86776; A(2,2) = 0.50671;
    
    RealVec x(3);
    x(0) =  0.529738;
    x(1) =  0.080727;
    x(2) = -2.553975;
    
    SparseRealVec b(3);
    
    axpy_prod(x, A, b, true);
    
    RealVec bVerified(3);
    bVerified(0) = -3.20991617262;
    bVerified(1) = -5.02010082870;
    bVerified(2) = -0.60485673896;
    
    for (size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE((double)b[i], bVerified[i], 1e-8);
    
    // Pass only the lower portion of A to matrixVectorMultLowerSymmetric, and
    // we should get the same b that was output by axpy_prod.
    A.clear();
    A(0,0) = -1.09263;
    A(1,0) = -0.43284; A(1,1) = -0.25514;
    A(2,0) =  1.01652; A(2,1) =  1.86776; A(2,2) = 0.50671;
    
    ublas_util::matrixVectorMultLowerSymmetric(A, x, b);
    
    for (size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE((double)b[i], bVerified[i], 1e-8);
    

    // Construct a lower triangular matrix of feature co-occurrences.
    AccumLogMat Cooc(3, 3);
    Cooc(0,0) = LogWeight(-1.74343, true);
    Cooc(1,0) = LogWeight(-1.16411, true);
    Cooc(1,1) = LogWeight(-1.69952, true);
    Cooc(2,0) = LogWeight(-0.84798, true);
    Cooc(2,1) = LogWeight( 0.42003, true);
    Cooc(2,2) = LogWeight(-0.24885, true);
    
    SparseRealVec feats(3);
    feats(0) =  1.12585;
    feats(1) =  0.66175;
    feats(2) = -0.52248;
    
    AccumRealMat Cov(3, 3);
    ublas_util::computeLowerCovarianceMatrix(Cooc, feats, Cov);
    
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        BOOST_CHECK_CLOSE((double)Cov(i,j), (double)A(i,j), 1e-2);
  }
  
  {
    SparseLogVec logFeats(3);
    SparseRealVec feats(3);
    logFeats(0) = LogWeight(-0.5, true);
    logFeats(1) = LogWeight( 0.3, true);
    //logFeats(2) implicitly set to LogWeight(0) (i.e., -Inf)
    
    ublas_util::exponentiate(logFeats, feats);
    BOOST_CHECK_CLOSE((double)feats[0], 0.606530659712633, 1e-8);
    BOOST_CHECK_CLOSE((double)feats[1], 1.349858807576003, 1e-8);
    BOOST_CHECK_SMALL((double)feats[2], 1e-8);
    
    RealVec sigmaFeats(3);
    ublas_util::sigmoid(feats, sigmaFeats);
    BOOST_CHECK_CLOSE((double)sigmaFeats[0], 0.647148993051504, 1e-8);
    BOOST_CHECK_CLOSE((double)sigmaFeats[1], 0.794106544007045, 1e-8);
    BOOST_CHECK_CLOSE((double)sigmaFeats[2], 0.5, 1e-8);
    
    AccumRealMat M(3, 3);
    M.clear();
    M(0,0) = 1; M(0,1) = 2; M(0,2) = 3;
    M(1,0) = -1; M(1,2) = -3;
    M(2,0) = 0; M(2,1) = 0.5;
    ublas_util::scaleMatrixRowsByVecTimesOneMinusVec(M, sigmaFeats);
    BOOST_CHECK_CLOSE((double)M(0,0), 0.22834717384392825, 1e-8);
    BOOST_CHECK_CLOSE((double)M(0,1), 0.45669434768785649, 1e-8);
    BOOST_CHECK_CLOSE((double)M(0,2), 0.68504152153178477, 1e-8);
    BOOST_CHECK_CLOSE((double)M(1,0), -0.16350134077223155, 1e-8);
    BOOST_CHECK_CLOSE((double)M(1,1), 0, 1e-8);
    BOOST_CHECK_CLOSE((double)M(1,2), -0.49050402231669465, 1e-8);
    BOOST_CHECK_CLOSE((double)M(2,0), 0, 1e-8);
    BOOST_CHECK_CLOSE((double)M(2,1), 0.125, 1e-8);
    BOOST_CHECK_CLOSE((double)M(2,2), 0, 1e-8);
  }
}
