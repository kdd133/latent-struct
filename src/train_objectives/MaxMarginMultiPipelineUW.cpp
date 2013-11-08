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
#include <boost/foreach.hpp>
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
  assert(0); // not yet implemented
}

void MaxMarginMultiPipelineUW::predictPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {

  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::unordered_map<pair<size_t, Label>,
        vector<StringPairAligned> >::iterator it;
      it = _kBestMap.find(item); // retrieve the k-best alignments
      if (it == _kBestMap.end()) {
        assert(0);
        cout << "Error: " << name() << " failed to retrieve alignments.\n";
        exit(1);
      }
      
      const vector<StringPairAligned>& alignments = it->second;
      const double z = bestAlignmentScore(alignments, theta.w, model, y);
      scores.setScore(id, y, z);
    }
  }
}

void MaxMarginMultiPipelineUW::initKBestPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k) {
  assert(theta.hasU());
  assert(KBestViterbiSemiring::k > 0);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < k; y++) {
        
      // Impute the max-scoring alignment based using u.
      stringstream align_ss;
      // false --> exclude observed (global) features
      model.printAlignment(align_ss, theta.u, x, y, false);
      
      // Parse the k-best alignments from the alignment string.
      vector<StringPairAligned> alignments = Utility::toStringPairAligned(
          align_ss.str());
      assert(alignments.size() <= KBestViterbiSemiring::k);
      
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

void MaxMarginMultiPipelineUW::clearKBest() {
  _kBestMap.clear();
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
    cout << i << " " << alignments[i] << " " << score << endl;
    if (own) delete phi;
    if (bestIndex == -1 || score > bestScore) {
      bestIndex = i;
      bestScore = score;
    }
  }
  return bestScore;
}

/* We override gatherFeaturesPart() because we only want to extract observed
 * features here. The latent features will have already been extracted (and
 * their graphs cached, if applicable) in initKBestPart().
 */
void MaxMarginMultiPipelineUW::gatherFeaturesPart(Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, size_t& maxFvs, size_t& totalFvs) {
  totalFvs = 0;
  maxFvs = 1; // there can only be one observed feature vector per instance
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < k; y++) {
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::unordered_map<pair<size_t, Label>,
        vector<StringPairAligned> >::iterator it;
      it = _kBestMap.find(item); // retrieve the k-best alignments
      if (it == _kBestMap.end()) {
        assert(0);
        cout << "Error: " << name() << " failed to retrieve alignments.\n";
        exit(1);
      }      
      const vector<StringPairAligned>& alignments = it->second;
      BOOST_FOREACH(const StringPairAligned& alignment, alignments) {
        bool own;
        SparseRealVec* phi = model.observedFeatures(alignment, y, own);
        assert(phi);
        if (own)
          delete phi;
        totalFvs++;
      }
    }
  }
}
