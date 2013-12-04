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
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

void MaxMarginMultiPipelineUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label nc, double& funcVal, SparseRealVec& gradFv) {    

  // This training objective computes a function of parameters u and w. The
  // parameters u correspond to "local" features that decompose over a given
  // sequence, while the w parameters correspond to "global" features that are
  // computed given a complete latent structure.
  // The training objective expects a parameter vector of the form
  // [w1 u1 w2 u2 ... wk uk], where wj is the portion of w that corresponds to
  // the jth class, and similarly for u. It is further assumed that w and u are
  // of the same length, namely, the total length of the parameter vector, but
  // u will be have zero weights for all the w coorindates. On the other hand,
  // w may have non-zero weights for the u coordinates, since the global
  // feature vector may include local features from, e.g., the max alignment.
  //
  // Below, we will refer to the three terms of the objective as follows:  
  // (1)   max_{z} (u-w)'*phi(xi,yi,z)
  // (2) + max_{y,z2} [delta(yi,y) + w'*phi(xi,y,z2)]
  // (3) - max_{z3} u'*phi(xi,yi,z3)

  const int n = theta.w.getDim(); // i.e., the number of features (all classes)
  const int d = theta.getDimWU(); // i.e., the length of the [w u] vector
  assert(theta.hasU());
  assert(n > 0 && n == theta.u.getDim());
  
  funcVal = 0;
  
  // It is faster to accumulate using a dense vector.
  RealVec gradDense(d);
  gradDense.clear();
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    vector<KBestInfo*> kBest(nc);
    for (Label y = 0; y < nc; y++) {
      kBest[y] = fetchKBestInfo(xi, y);
      assert(kBest[y]->maxFvs);
      const int k = kBest[y]->maxFvs->size();
      assert(k > 0);
      
      // If the observed feature vectors haven't been stored yet, we create
      // them here by extracting features based on the string representation
      // of the k-best alignments that was stored in initKBestPart.
      if (!kBest[y]->observedFvs) {
        assert(kBest[y]->alignStrings.size() > 0);
        kBest[y]->observedFvs.reset(
            new vector<shared_ptr<const SparseRealVec> >(k));
        shared_ptr<vector<StringPairAligned> > alignments =
            Utility::toStringPairAligned(kBest[y]->alignStrings);
        for (int j = 0; j < k; j++) {
          shared_ptr<const SparseRealVec> phi = model.observedFeatures(
              (*alignments)[j], y);
          assert(phi);
          (*kBest[y]->observedFvs)[j] = phi;
        }
        // Now that we have the k-best feature vectors, we don't need the
        // string representation of the alignments any more, so free the memory.
        kBest[y]->alignStrings.clear();
      }
      assert(kBest[y]->observedFvs->size() == k);
      assert(fetchKBestInfo(xi, y)->observedFvs->size() == k);
    }
    
    // Compute terms (1) and (3).
    {
      int indexMaxUW, indexMaxU;
      double scoreMaxUW, scoreMaxU;     
      maxZ(*kBest[yi], yi, theta, model, scoreMaxUW, indexMaxUW, scoreMaxU,
          indexMaxU);
      assert((*kBest[yi]->maxFvs)[indexMaxUW]->size() == n);
      
      shared_ptr<const SparseRealVec> phiMax_w =
          (*kBest[yi]->observedFvs)[indexMaxUW];
      assert(phiMax_w);
      assert(phiMax_w->size() == n);
      
      // Update the gradient and function value contribution of term (1).
      noalias(subrange(gradDense, 0, n)) -= *phiMax_w;
      noalias(subrange(gradDense, n, d)) += *(*kBest[yi]->maxFvs)[indexMaxUW];
      funcVal += scoreMaxUW;
      
      // Update the gradient and function value contribution of term (3).
      noalias(subrange(gradDense, n, d)) -= *(*kBest[yi]->maxFvs)[indexMaxU];
      funcVal -= scoreMaxU;
    }

    // Compute term (2).
    {
      vector<double> score(nc);
      vector<SparseRealVec> feats(nc, SparseRealVec(d));
      Label yMax = 0;
      double scoreMax = -numeric_limits<double>::infinity();    
      int indexMax = -1;
      for (Label y = 0; y < nc; y++) {
        int kBestIndex = -1;
        score[y] = Utility::delta(yi,y) + bestAlignmentScore(*kBest[y], theta.w,
            model, y, &kBestIndex);
        assert(kBestIndex >= 0);
        if (score[y] > scoreMax) {
          scoreMax = score[y];
          yMax = y;
          indexMax = kBestIndex;
        }
      }
      assert(indexMax >= 0);
    
      // Update the gradient and function value.
      noalias(subrange(gradDense, 0, n)) +=
          *(*kBest[yMax]->observedFvs)[indexMax];
      funcVal += score[yMax];
    }
  }
  noalias(gradFv) = gradDense;
}

void MaxMarginMultiPipelineUW::predictPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label numClasses, LabelScoreTable& scores) {

#if 0 // 0 --> pipeline scoring; 1 --> objective function scoring
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < numClasses; y++) {
      const KBestInfo& kBest = fetchKBestInfo(x, y);
      assert(kBest.alignments->size() == kBest.maxFvs->size());
      
      int indexMaxUW, indexMaxU;
      double term1, term3;
      maxZ(kBest, y, theta, model, term1, indexMaxUW, term3, indexMaxU);

      double term2 = 0; //bestAlignmentScore(*kBest.alignments, theta.w, model, y);
      double z = term1 + term2 - term3;
      scores.setScore(id, y, z);
    }
  }
#else
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    const size_t id = x.getId();
    for (Label y = 0; y < numClasses; y++) {      
      int index = -1;
      double z;
      
      bool useKBest = true;
      if (!useKBest) {
        // Impute the max-scoring alignment based using u.
        stringstream align_ss;
        // false --> exclude observed (global) features
        shared_ptr<vector<shared_ptr<const SparseRealVec> > > maxFvs;
        model.getBestAlignments(align_ss, maxFvs, theta.u, x, y, false);
        
        // Parse the best alignments from the alignment string.
        shared_ptr<vector<StringPairAligned> > alignments =
            Utility::toStringPairAligned(align_ss.str());
        const int k = alignments->size();
        assert(k > 0);
        
        // Get the observed feature vector for each alignment.
        KBestInfo kBest;
        kBest.observedFvs->resize(k);
        for (int j = 0; j < k; j++)
          (*kBest.observedFvs)[j] = model.observedFeatures((*alignments)[j], y);
        
        z = bestAlignmentScore(kBest, theta.w, model, y, &index);
      }
      else {
        const KBestInfo* kBest = fetchKBestInfo(x, y);
        assert(kBest->observedFvs->size() > 0);
        z = bestAlignmentScore(*kBest, theta.w, model, y, &index);
      }
      
      assert(index >= 0); // index not used here, but we check validity anyway      
      scores.setScore(id, y, z);
    }
  }
#endif
}

MaxMarginMultiPipelineUW::KBestInfo*
MaxMarginMultiPipelineUW::fetchKBestInfo(const Pattern& x, Label y) {
  pair<size_t, Label> item = make_pair(x.getId(), y);
  boost::ptr_map<pair<size_t, Label>, KBestInfo>::iterator it;
  it = _kBestMap.find(item); // retrieve the k-best alignments
  if (it == _kBestMap.end()) {
    assert(0);
    cout << "Error: " << name() << " failed to retrieve alignments.\n";
    exit(1);
  }
  assert(it->second);
  return it->second;
}

void MaxMarginMultiPipelineUW::initKBestPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label nc) {
  assert(theta.hasU());
  assert(KBestViterbiSemiring::k > 0);
  
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < nc; y++) {
      // Impute the k max-scoring alignments using u.
      KBestInfo* kBest = new KBestInfo();
      stringstream align_ss;
      // false --> exclude observed (global) features
      model.getBestAlignments(align_ss, kBest->maxFvs, theta.u, x, y, false);
      assert(kBest->maxFvs->size() > 0);
      assert(kBest->maxFvs->size() <= KBestViterbiSemiring::k);
      kBest->alignStrings = align_ss.str();
      
      boost::mutex::scoped_lock lock(_lock); // place a lock on _kBestMap
      
      pair<size_t, Label> item = make_pair(x.getId(), y);
      boost::ptr_map<pair<size_t, Label>, KBestInfo>::iterator itFind =
        _kBestMap.find(item);

      // The initialization method should only be called once, and we should
      // never try to insert a duplicate (Example,Label) pair.
      assert(itFind == _kBestMap.end());
      _kBestMap.insert(item, kBest);
    }
  }
}

void MaxMarginMultiPipelineUW::clearKBest() {
  _kBestMap.clear();
}

double MaxMarginMultiPipelineUW::bestAlignmentScore(
    const KBestInfo& kBest, const WeightVector& weights,
    Model& model, const Label y, int* indexBest) {
  int bestIndex = -1;
  double bestScore = -1;
  assert(kBest.observedFvs->size() > 0);
  for (int i = 0; i < kBest.observedFvs->size(); i++) {
    shared_ptr<const SparseRealVec> phi = (*kBest.observedFvs)[i];
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
    const Label numClasses, size_t& maxFvs, size_t& totalFvs) {
  totalFvs = 0;
  maxFvs = 1; // there can only be one observed feature vector per instance
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < numClasses; y++) {
      KBestInfo* kBest = fetchKBestInfo(x, y);
      const int k = kBest->maxFvs->size();
      assert(k > 0);
      shared_ptr<vector<StringPairAligned> > alignments =
          Utility::toStringPairAligned(kBest->alignStrings);
      assert(alignments->size() == k);
      for (int j = 0; j < k; j++) {
        model.observedFeatures((*alignments)[j], y);
        totalFvs++;
      }
    }
  }
}

void MaxMarginMultiPipelineUW::valueAndGradientFinalize(const Parameters& theta,
    double& funcVal, SparseRealVec& gradFv) {
  assert(_imputedFv);
  // Subtract the sum of the imputed vectors from the gradient.
  const int n = theta.w.getDim(); // i.e., the number of features (all classes)
  const int d = theta.getDimWU(); // i.e., the length of the [w u] vector
  noalias(subrange(gradFv, n, d)) -= (*_imputedFv);
  // Subtract the scores of the imputed vectors from the function value.
  funcVal -= theta.u.innerProd(*_imputedFv);
}

void MaxMarginMultiPipelineUW::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  assert(_imputedFv);
  SparseRealVec fv(theta.u.getDim());
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& xi = *it->x();
    const Label yi = it->y();
    // Note: The last argument is set to false because the u parameters do not
    // overlap with the observed features in this model, so there is no need to
    // compute those features.
    model.maxFeatures(theta.u, &fv, xi, yi, false);
    boost::mutex::scoped_lock lock(_flag); // place a lock on _imputedFv
    noalias(*_imputedFv) += fv;
  }
}

void MaxMarginMultiPipelineUW::initLatentFeatureVectors(const Parameters& theta) {
  assert(theta.u.getDim() == theta.w.getDim());
  assert(_dataset.numExamples() > 0);
  _imputedFv.reset(new RealVec(theta.u.getDim()));
}

void MaxMarginMultiPipelineUW::clearLatentFeatureVectors() {
  assert(_imputedFv);
  _imputedFv->clear();
}

void MaxMarginMultiPipelineUW::maxZ(const KBestInfo& kBest, const Label y,
    const Parameters& theta, Model& model, double& scoreMaxUW, int& indexMaxUW,
    double& scoreMaxU, int& indexMaxU) {
  scoreMaxUW = -numeric_limits<double>::infinity();
  indexMaxUW = -1;
  scoreMaxU = -numeric_limits<double>::infinity();
  indexMaxU = -1;
  // Compute max_{z} (u-w)'*phi(x,y,z)
  assert(kBest.observedFvs->size() > 0);
  for (int j = 0; j < kBest.observedFvs->size(); j++) {
    shared_ptr<const SparseRealVec> phi_w = (*kBest.observedFvs)[j];
    assert(phi_w);
    const double scoreU = theta.u.innerProd(*(*kBest.maxFvs)[j]);
    const double scoreW = theta.w.innerProd(*phi_w);
    const double scoreUW = scoreU - scoreW;
    if (scoreUW > scoreMaxUW) {
      scoreMaxUW = scoreUW;
      indexMaxUW = j;
    }
    if (scoreU > scoreMaxU) {
      scoreMaxU = scoreU;
      indexMaxU = j;
    }
  }
  assert(indexMaxUW >= 0);
  assert(indexMaxU >= 0);
}
