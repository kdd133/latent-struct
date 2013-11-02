/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Example.h"
#include "FeatureGenConstants.h"
#include "Label.h"
#include "MaxMarginBinaryPipelineUW.h"
#include "Model.h"
#include "Parameters.h"
#include "StringPairAligned.h"
#include "Ublas.h"
#include "Utility.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

void MaxMarginBinaryPipelineUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  assert(0); // not yet implemented
}

void MaxMarginBinaryPipelineUW::predictPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  const Label ypos = TrainingObjective::kPositive;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
      
    // Impute the max-scoring alignment based using u.
    stringstream align_ss;
    // false --> exclude observed (global) features
    model.printAlignment(align_ss, theta.u, x, ypos, false);
    
    // Create a StringPairAligned object by parsing the alignment string.
    StringPairAligned xPrime = Utility::toStringPairAligned(align_ss.str());
    
    // Compute the "global" features and then classify using w.
    bool own = false;
    SparseRealVec* phi = model.observedFeatures(xPrime, ypos, own);
    assert(phi);
    const double z = theta.w.innerProd(*phi);
    if (own) delete phi;
    
    scores.setScore(id, ypos, z);
    scores.setScore(id, !ypos, (-z));
  }
}
