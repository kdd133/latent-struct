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
#include "KBestViterbiSemiring.h"
#include "Label.h"
#include "MaxMarginMultiPipelineUW.h"
#include "Model.h"
#include "Parameters.h"
#include "StringPairAligned.h"
#include "Ublas.h"
#include "Utility.h"
#include <algorithm>
#include <assert.h>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

void MaxMarginMultiPipelineUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {    
  initializeKBestPart(theta, model, begin, end, k);
  
  assert(0); // not yet implemented
}

void MaxMarginMultiPipelineUW::predictPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {
  initializeKBestPart(theta, model, begin, end, k);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::unordered_map<pair<size_t, Label>,
        vector<StringPairAligned> >::iterator it;
      {
        boost::mutex::scoped_lock lock(_lock); // place a lock on _kBestMap
        it = _kBestMap.find(item); // retrieve the k-best alignments
      }        
      assert(it != _kBestMap.end());
      if (it == _kBestMap.end()) {
        cout << "Error: " << name() << " failed to retrieve alignments.\n";
        exit(1);
      }
      
      const vector<StringPairAligned>& alignments = it->second;
      const double z = bestAlignmentScore(alignments, theta.w, model, y);
      scores.setScore(id, y, z);
    }
  }
}

void MaxMarginMultiPipelineUW::initializeKBestPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k) {
  assert(theta.hasU());
  assert(KBestViterbiSemiring::k > 0);
  
  // If the first example in this partition of the dataset already has a k-best
  // list stored, it means that the k-best lists for all the examples in this
  // partition have been initialized; so, we simply return.
  pair<size_t, Label> first = make_pair(begin->x()->getId(), 0);
  {    
    boost::mutex::scoped_lock lock(_lock); // place a lock on _kBestMap
    boost::unordered_map<pair<size_t, Label>,
      vector<StringPairAligned> >::iterator firstFind = _kBestMap.find(first);
    if (firstFind != _kBestMap.end())
      return;
  }
  
  for (Label y = 0; y < k; y++) {
    for (Dataset::iterator it = begin; it != end; ++it) {
      const Pattern& x = *it->x();
        
      // Impute the max-scoring alignment based using u.
      stringstream align_ss;
      // false --> exclude observed (global) features
      model.printAlignment(align_ss, theta.u, x, y, false);
      
      // Parse the k-best alignments from the alignment string.
      vector<StringPairAligned> alignments = Utility::toStringPairAligned(
          align_ss.str());
      assert(alignments.size() == KBestViterbiSemiring::k);
      
      boost::mutex::scoped_lock lock(_lock); // place a lock on _kBestMap
      
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::unordered_map<pair<size_t, Label>,
        vector<StringPairAligned> >::iterator itFind = _kBestMap.find(item);

      // The initialization method should only be called once, and we should
      // never try to insert a duplicate (Example,Label) pair.
      assert(itFind == _kBestMap.end());
      _kBestMap.insert(make_pair(item, alignments));
    }
  }
}

double MaxMarginMultiPipelineUW::bestAlignmentScore(
    const vector<StringPairAligned>& alignments, const WeightVector& weights,
    Model& model, const Label y) {
  int bestIndex = -1;
  double bestScore = -1;
  assert(alignments.size() > 0);
  for (int i = 0; i < alignments.size(); i++) {
    bool own = false;
    SparseRealVec* phi = model.observedFeatures(alignments[i], y, own);
    assert(phi);
    const double score = weights.innerProd(*phi);
    if (own) delete phi;
    if (bestIndex == -1 || score > bestScore) {
      bestIndex = i;
      bestScore = score;
    }
  }
  return bestScore;
}
