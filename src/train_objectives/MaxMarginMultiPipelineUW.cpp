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
#include <boost/shared_ptr.hpp>
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

  const int n = theta.w.getDim(); // i.e., the number of features (all classes)
  const int d = theta.getDimWU(); // i.e., the length of the [w u] vector
  assert(theta.hasU());
  assert(n > 0 && n == theta.u.getDim());
  
  vector<double> score(k);
  vector<SparseRealVec> feats(k, SparseRealVec(d));
  vector<KBestInfo> kBest(k);
  
  funcVal = 0;
  
  // It is faster to accumulate using a dense vector.
  RealVec gradDense(d);
  gradDense.clear();
  
  // This if statement should never be executed while the objective is being
  // optimized, since the (EM) optimizer will perform the initialization.
  // However, if the gradient is requested outside of the optimization routine,
  // we need to initialize the latent FVs using the given weight vector.
  if (!_imputedFv)
    initLatentFeatureVectors(theta);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    for (Label y = 0; y < k; y++) {
      kBest[y] = fetchKBestInfo(xi, y);
      assert(kBest[y].alignments->size() == kBest[y].maxFvs->size()); 
    }
    
    // The block below looks like what we should be doing to compute the
    // max_{y,z}w'phi (2nd) term. To compute the first term, we need to check
    // all k alignments explicitly here, i.e., not call bestAlignment().
    
    double scoreMax = -numeric_limits<double>::infinity();
    Label yMax = 0;
    int indexMax = -1;
    shared_ptr<const SparseRealVec> phiMax_w;
    
    // Compute max_{z} (u-v)'*phi(xi,yi,z)
    for (int j = 0; j < kBest[yi].alignments->size(); j++) {
      shared_ptr<const SparseRealVec> phi_w = model.observedFeatures(
          kBest[yi].alignments->at(j), yi);
      assert(phi_w);
      const double score_u = theta.u.innerProd(*kBest[yi].maxFvs->at(j));
      const double score_w = theta.w.innerProd(*phi_w);
      const double score = score_u - score_w;
      if (score > scoreMax) {
        scoreMax = score;
        indexMax = j;
        phiMax_w = phi_w;
      }
    }
    assert(indexMax >= 0);
    assert(phiMax_w);
    
    // Update the gradient and function value.
    noalias(gradDense) += *kBest[yi].maxFvs->at(indexMax);
    noalias(gradDense) -= *phiMax_w;
    funcVal += scoreMax;
    
    // Compute max_{y,z} [delta(yi,y) + w'*phi(xi,y,z)]
    scoreMax = -numeric_limits<double>::infinity();  
    yMax = 0;
    indexMax = -1;
    for (Label y = 0; y < k; y++) {
      int kBestIndex = -1;
      score[y] = Utility::delta(yi,y) + bestAlignment(*kBest[y].alignments,
          theta.w, model, y, &kBestIndex);
      assert(kBestIndex >= 0);
      if (score[y] > scoreMax) {
        scoreMax = score[y];
        yMax = y;
        indexMax = kBestIndex;
      }
    }
    assert(indexMax >= 0);
    
    // Update the gradient and function value.
    noalias(gradDense) += *kBest[yMax].maxFvs->at(indexMax);
    funcVal += score[yMax];
    
    // Subtract the observed features and score for the correct label yi.
    shared_ptr<const SparseRealVec> phi_yi = model.observedFeatures(xi, yi);
    assert(phi_yi);
    noalias(gradDense) -= (*phi_yi);
    funcVal -= theta.w.innerProd(*phi_yi);
  }
  noalias(gradFv) = gradDense;
  assert(0); // this method is incomplete
}

void MaxMarginMultiPipelineUW::predictPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, LabelScoreTable& scores) {

  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < k; y++) {
      const KBestInfo& kBest = fetchKBestInfo(x, y);
      const double z = bestAlignment(*kBest.alignments, theta.w, model, y);
      scores.setScore(id, y, z);
    }
  }
}

const MaxMarginMultiPipelineUW::KBestInfo&
MaxMarginMultiPipelineUW::fetchKBestInfo(const Pattern& x, Label y) {
  pair<size_t, Label> item = make_pair(x.getId(), y);
  boost::unordered_map<pair<size_t, Label>, KBestInfo>::iterator it;
  it = _kBestMap.find(item); // retrieve the k-best alignments
  if (it == _kBestMap.end()) {
    assert(0);
    cout << "Error: " << name() << " failed to retrieve alignments.\n";
    exit(1);
  }  
  return it->second;
}

void MaxMarginMultiPipelineUW::initKBestPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k) {
  assert(theta.hasU());
  assert(KBestViterbiSemiring::k > 0);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < k; y++) {
      // Impute the k max-scoring alignments using u.
      KBestInfo kBest;
      stringstream align_ss;
      // false --> exclude observed (global) features
      model.getBestAlignments(align_ss, kBest.maxFvs, theta.u, x, y, false);
      
      // Parse the k-best alignments from the alignment string.
      kBest.alignments = Utility::toStringPairAligned(align_ss.str());
      assert(kBest.alignments->size() <= KBestViterbiSemiring::k);
      
      boost::mutex::scoped_lock lock(_lock); // place a lock on _kBestMap
      
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::unordered_map<pair<size_t, Label>, KBestInfo>::iterator itFind =
        _kBestMap.find(item);

      // The initialization method should only be called once, and we should
      // never try to insert a duplicate (Example,Label) pair.
      assert(itFind == _kBestMap.end());
      _kBestMap.insert(make_pair(item, kBest));
    }
  }
}

void MaxMarginMultiPipelineUW::clearKBest() {
  _kBestMap.clear();
}

double MaxMarginMultiPipelineUW::bestAlignment(
    const vector<StringPairAligned>& alignments, const WeightVector& weights,
    Model& model, const Label y, int* indexBest) {
  int bestIndex = -1;
  double bestScore = -1;
  assert(alignments.size() > 0);
  for (int i = 0; i < alignments.size(); i++) {
    shared_ptr<const SparseRealVec> phi = model.observedFeatures(alignments[i], y);
    assert(phi);
    const double score = weights.innerProd(*phi);
#if 0
    SparseRealVec::const_iterator pos = phi->begin();
    for (; pos != phi->end(); ++pos) {
      cout << pos.index() << " " << *pos << endl;
    }
    cout << endl;
    cout << i << " " << alignments[i] << " " << score << endl;
#endif
    if (bestIndex == -1 || score > bestScore) {
      bestIndex = i;
      bestScore = score;
    }
  }
  if (indexBest)
    *indexBest = bestIndex;
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
      const KBestInfo& kBest = fetchKBestInfo(x, y);
      BOOST_FOREACH(const StringPairAligned& alignment, *kBest.alignments) {
        shared_ptr<const SparseRealVec> phi = model.observedFeatures(
            alignment, y);
        assert(phi);
        totalFvs++;
      }
    }
  }
}

void MaxMarginMultiPipelineUW::valueAndGradientFinalize(const Parameters& theta,
    double& funcVal, SparseRealVec& gradFv) {    
  // Subtract the sum of the imputed vectors from the gradient.
  noalias(gradFv) -= (*_imputedFv);
  // Subtract the scores of the imputed vectors from the function value.
  funcVal = Utility::hinge(funcVal - theta.u.innerProd(*_imputedFv)); 
}

void MaxMarginMultiPipelineUW::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  SparseRealVec fv(theta.w.getDim());
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    
    // Note: The last argument is set to false because we wish to exclude the
    // observed features (if any) being included in the max feature vector.
    // Those do not need to be fixed in order to exploit the semi-convexity
    // property.
    model.maxFeatures(theta.u, &fv, xi, yi, false);
    boost::mutex::scoped_lock lock(_flag); // place a lock on _imputedFv
    noalias(*_imputedFv) += fv;
  }
}

void MaxMarginMultiPipelineUW::initLatentFeatureVectors(const Parameters& theta) {
  assert(_dataset.numExamples() > 0);
  _imputedFv.reset(new RealVec(theta.u.getDim()));
}

void MaxMarginMultiPipelineUW::clearLatentFeatureVectors() {
  assert(_imputedFv);
  _imputedFv->clear();
}
