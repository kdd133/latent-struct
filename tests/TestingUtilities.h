/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _TESTINGUTILITIES_H
#define _TESTINGUTILITIES_H

class Parameters;
class TrainingObjective;

namespace testing_util {
  
  // Note: Modifies theta.
  void checkGradientFiniteDifferences(TrainingObjective& objective,
      Parameters& theta, double tolerance, int numRandomWeightVectors);
}

#endif
