/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include <boost/shared_array.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>

using namespace boost;

namespace testing_util {

  void checkGradientFiniteDifferences(TrainingObjective& objective,
      Parameters& theta, double tol, int nWeightVectors) {
    const int d = theta.getDimWU();
    SparseRealVec gradFv(d);
    double fval;

    for (int wi = 0; wi < nWeightVectors; wi++) {
      // try several different (random) weight vectors
      boost::shared_array<double> weights = Utility::generateGaussianSamples(d,
          (wi%2 ? 1 : -1)*wi, 0.5, wi); // alternate the sign of the prior mean
      theta.setWeights(weights.get(), d);
      objective.valueAndGradient(theta, fval, gradFv, 0, true);
      for (int i = 0; i < d; i++) {
        const double numGrad_i = Utility::getNumericalGradientForCoordinate(
            objective, theta, i);
        if (std::abs(numGrad_i) > tol && std::abs(gradFv[i]) > tol) {
          BOOST_CHECK_CLOSE(numGrad_i, (double)gradFv[i], tol);
        }
        else {
          // BOOST_CHECK_CLOSE causes false positives when, e.g., one of the
          // values is zero. So, we verify that both values are small here.
          BOOST_CHECK_SMALL((double)gradFv[i], tol);
          BOOST_CHECK_SMALL(numGrad_i, tol);
        }
      }
    }
  }
}
