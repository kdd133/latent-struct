/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "FeatureVector.h"
#include "Label.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "WeightVector.h"
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
using namespace boost;
using namespace std;

const Label TrainingObjective::kPositive = 1;

TrainingObjective::TrainingObjective(const Dataset& dataset,
    const vector<Model*>& models) :
    _dataset(dataset), _computeAverageLoss(true) {
  BOOST_FOREACH(Model* model, models) {
    _models.push_back(model);
  }
  assert(_dataset.numExamples() == 0 || // i.e., data not loaded yet
      _dataset.numPartitions() == _models.size());
}

void TrainingObjective::valueAndGradient(const WeightVector& w, double& fval,
    FeatureVector<RealWeight>& gradFv) {
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)_dataset.getLabelSet().size();
  vector<FeatureVector<RealWeight> > grads(numParts, w.getDim());
  vector<double> fvals(numParts, 0);
  
  // Compute the function values and gradients.
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(boost::bind(
        &TrainingObjective::valueAndGradientPart, this,
        cref(w), ref(_models[i]), begin, end, k, ref(fvals[i]), ref(grads[i])
    )));
  }
    
  // Combine the results.
  gradFv.zero();
  fval = 0;
  for (size_t i = 0; i < numParts; i++) {
    threads[i].join(); // Wait for the thread to finish.
    grads[i].addTo(gradFv);
    fval += fvals[i];
  }
  
  valueAndGradientFinalize(w, fval, gradFv);
  
  if (_computeAverageLoss) {
    // Scale the value and gradient by 1/t
    const double scaleFactor = 1.0 / _dataset.numExamples();
    fval *= scaleFactor;
    gradFv.timesEquals(scaleFactor); 
  }
}

void TrainingObjective::predict(const WeightVector& w, const Dataset& evalData,
    LabelScoreTable& scores) {
  const size_t numParts = evalData.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)evalData.getLabelSet().size();
  assert(k > 1);
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = evalData.partitionBegin(i);
    const Dataset::iterator end = evalData.partitionEnd(i);
    threads.push_back(new thread(boost::bind(
        &TrainingObjective::predictPart, this,
        cref(w), ref(_models[i]), begin, end, k, ref(scores)
    )));
  }  
  // Wait for the threads to finish.
  for (size_t i = 0; i < numParts; i++)
    threads[i].join();
}

void TrainingObjective::setLatentFeatureVectors(const WeightVector& w) {
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  clearLatentFeatureVectors();
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(boost::bind(
        &TrainingObjective::setLatentFeatureVectorsPart, this,
        cref(w), ref(_models[i]), begin, end
    )));
  }
  // Wait for the threads to finish.
  for (size_t i = 0; i < numParts; i++)
    threads[i].join();
}

void TrainingObjective::setLatentFeatureVectorsPart(const WeightVector& w,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  assert(0); // Not implemented by the given TrainingObjective subclass.
             // Should not be called for an objective that doesn't use it.
}

void TrainingObjective::clearLatentFeatureVectors() {
  assert(0); // Not implemented by the given TrainingObjective subclass.
             // Should not be called for an objective that doesn't use it.
}

void TrainingObjective::initLatentFeatureVectors(const WeightVector& w) {
  // Not implemented by the given TrainingObjective subclass. Do nothing.
}

void TrainingObjective::valueAndGradientFinalize(const WeightVector& w,
    double& f, FeatureVector<RealWeight>& g) {
  // Not implemented by the given TrainingObjective subclass. Do nothing.
}

void TrainingObjective::gatherFeatures(size_t& maxFvs, size_t& totalFvs) {
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)_dataset.getLabelSet().size();
  vector<size_t> maxFvsPart(numParts);
  vector<size_t> totalFvsPart(numParts);
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(boost::bind(
        &TrainingObjective::gatherFeaturesPart, this,
        ref(_models[i]), begin, end, k, ref(maxFvsPart[i]), ref(totalFvsPart[i])
    )));
  }
  // Wait for the threads to finish, and combine the results.
  maxFvs = 0;
  totalFvs = 0;
  for (size_t i = 0; i < numParts; i++) {
    threads[i].join();
    totalFvs += totalFvsPart[i];
    if (maxFvsPart[i] > maxFvs)
      maxFvs = maxFvsPart[i];
  }
}

void TrainingObjective::gatherFeaturesPart(Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, size_t& maxFvs, size_t& totalFvs) {
  totalFvs = 0;
  maxFvs = 0;
  for (Dataset::iterator it = begin; it != end; ++it) {
    const Pattern& x = *it->x();
    for (Label y = 0; y < k; y++) {
      if (!(isBinary() && y != 1)) { 
        const size_t numFvs = model.gatherFeatures(x, y);
        totalFvs += numFvs;
        if (numFvs > maxFvs)
          maxFvs = numFvs;
      }
    }
  }
}

void TrainingObjective::combineAndLockAlphabets() {
  shared_ptr<Alphabet> combined(new Alphabet(false, false));
  Alphabet::DictType::const_iterator it;
  for (size_t i = 0; i < getNumModels(); i++) {
    vector<shared_ptr<Alphabet> > alphabets;
    alphabets.push_back(_models[i].getFgenLatent()->getAlphabet());
    alphabets.push_back(_models[i].getFgenObserved()->getAlphabet());
    BOOST_FOREACH(shared_ptr<Alphabet> a, alphabets) {
      const size_t d = a->size();
      for (size_t j = 0; j < d; j++)
        combined->lookup(a->reverseLookup(j), true);
    }
  }
  combined->lock();
  for (size_t i = 0; i < getNumModels(); i++) {
    _models[i].getFgenLatent()->setAlphabet(combined);
    _models[i].getFgenObserved()->setAlphabet(combined);
  }
}
