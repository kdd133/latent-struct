/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Label.h"
#include "TrainingObjective.h"
#include "Parameters.h"
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <list>
#include <vector>

using boost::bind;
using boost::ptr_vector;
using boost::shared_ptr;
using boost::thread;
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

void TrainingObjective::valueAndGradientOne(const Parameters& theta,
    double& fval, RealVec& gradFv, size_t i) {
  assert(gradFv.size() == theta.getDimTotal());
  assert(i < _dataset.getExamples().size());
  const Label k = (Label)_dataset.getLabelSet().size();
  
  // In order to invoke the valueAndGradientPart function, we must pass it
  // list<Example> iterators. So, we create a list containing one example.
  const Example& ex = _dataset.getExamples()[i];
  list<Example> dataOne;
  dataOne.push_back(ex);
  const Dataset::iterator begin = dataOne.begin();
  const Dataset::iterator end = dataOne.end();
  
  fval = 0;
  gradFv.clear();
  valueAndGradientPart(theta, _models[0], begin, end, k, fval, gradFv);
}

void TrainingObjective::valueAndGradient(const Parameters& theta,
    const Dataset::iterator& begin, const Dataset::iterator& end, double& fval,
    RealVec& gradFv) {
  assert(gradFv.size() == theta.getDimTotal());
  const Label k = (Label)_dataset.getLabelSet().size();
  
  fval = 0;
  gradFv.clear();
  valueAndGradientPart(theta, _models[0], begin, end, k, fval, gradFv);
}

void TrainingObjective::valueAndGradient(const Parameters& theta, double& fval,
    RealVec& gradFv) {
  assert(gradFv.size() == theta.getDimTotal());
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)_dataset.getLabelSet().size();
  vector<RealVec> grads(numParts, RealVec(theta.getDimTotal()));
  vector<double> fvals(numParts, 0);
  
  // Compute the function values and gradients.
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(bind(
        &TrainingObjective::valueAndGradientPart, this, boost::cref(theta),
        boost::ref(_models[i]), begin, end, k, boost::ref(fvals[i]),
        boost::ref(grads[i])
    )));
  }
    
  // Combine the results.
  gradFv.clear();
  fval = 0;
  for (size_t i = 0; i < numParts; i++) {
    threads[i].join(); // Wait for the thread to finish.
    gradFv += grads[i];
    fval += fvals[i];
  }
  
  valueAndGradientFinalize(theta, fval, gradFv);
  
  if (_computeAverageLoss) {
    // Scale the value and gradient by 1/t
    const double scaleFactor = 1.0 / _dataset.numExamples();
    fval *= scaleFactor;
    gradFv *= scaleFactor; 
  }
}

void TrainingObjective::predict(const Parameters& theta, const Dataset& evalData,
    LabelScoreTable& scores) {
  const size_t numParts = evalData.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)evalData.getLabelSet().size();
  assert(k > 1);
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = evalData.partitionBegin(i);
    const Dataset::iterator end = evalData.partitionEnd(i);
    threads.push_back(new thread(bind(
        &TrainingObjective::predictPart, this, boost::cref(theta),
        boost::ref(_models[i]), begin, end, k, boost::ref(scores)
    )));
  }  
  // Wait for the threads to finish.
  for (size_t i = 0; i < numParts; i++)
    threads[i].join();
}

void TrainingObjective::setLatentFeatureVectors(const Parameters& theta) {
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  clearLatentFeatureVectors();
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(bind(
        &TrainingObjective::setLatentFeatureVectorsPart, this,
        boost::cref(theta), boost::ref(_models[i]), begin, end
    )));
  }
  // Wait for the threads to finish.
  for (size_t i = 0; i < numParts; i++)
    threads[i].join();
}

void TrainingObjective::setLatentFeatureVectorsPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end) {
  assert(0); // Not implemented by the given TrainingObjective subclass.
             // Should not be called for an objective that doesn't use it.
}

void TrainingObjective::clearLatentFeatureVectors() {
  assert(0); // Not implemented by the given TrainingObjective subclass.
             // Should not be called for an objective that doesn't use it.
}

void TrainingObjective::initLatentFeatureVectors(const Parameters& theta) {
  // Not implemented by the given TrainingObjective subclass. Do nothing.
}

void TrainingObjective::valueAndGradientFinalize(const Parameters& theta,
    double& f, RealVec& g) {
  // Not implemented by the given TrainingObjective subclass. Do nothing.
}

void TrainingObjective::gatherFeatures(size_t& maxFvs, size_t& totalFvs) {
  const size_t numParts = _dataset.numPartitions();
  assert(numParts == getNumModels());
  const Label k = (Label)_dataset.getLabelSet().size();
  // If there's only one unique label, it has to be 0 in order for the loop
  // over labels in gatherFeaturesPart() to make sense.
  assert(k > 1 || _dataset.getExamples().at(0).y() == 0);
  vector<size_t> maxFvsPart(numParts);
  vector<size_t> totalFvsPart(numParts);
  ptr_vector<thread> threads;
  for (size_t i = 0; i < numParts; i++) {
    const Dataset::iterator begin = _dataset.partitionBegin(i);
    const Dataset::iterator end = _dataset.partitionEnd(i);
    threads.push_back(new thread(bind(
        &TrainingObjective::gatherFeaturesPart, this, boost::ref(_models[i]),
        begin, end, k, boost::ref(maxFvsPart[i]), boost::ref(totalFvsPart[i])
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

shared_ptr<Alphabet> TrainingObjective::combineAlphabets(
    const set<Label>& labels) {
  shared_ptr<Alphabet> combined(new Alphabet(false, false));
  Alphabet::DictType::const_iterator it;
  for (size_t i = 0; i < getNumModels(); i++) {
    vector<shared_ptr<Alphabet> > alphabets;
    alphabets.push_back(_models[i].getFgenLatent()->getAlphabet());
    alphabets.push_back(_models[i].getFgenObserved()->getAlphabet());
    BOOST_FOREACH(shared_ptr<Alphabet> a, alphabets) {
      // Note: We don't use a->size() here because that would give us the
      // dimensionality of the space (across all classes), whereas we want the
      // number of (base) features in this case.
      const size_t n = a->numFeaturesPerClass();
      for (size_t j = 0; j < n; j++)
        combined->lookup(a->reverseLookup(j), 0, true);
    }
  }
  // We performed lookups using label 0 above, simply to populate the feature
  // dictionary. We must also tell the alphabet which class labels exist, so
  // that it knows how many "copies" of each feature are valid.
  set<Label>::const_iterator lbl;
  for (lbl = labels.begin(); lbl != labels.end(); ++lbl)
    combined->addLabel(*lbl);
  for (size_t i = 0; i < getNumModels(); i++) {
    _models[i].getFgenLatent()->setAlphabet(combined);
    _models[i].getFgenObserved()->setAlphabet(combined);
  }
  return combined;
}

Parameters TrainingObjective::getDefaultParameters(size_t numFeatures) const {
  if (isUW())
    return Parameters(numFeatures, numFeatures);
  else
    return Parameters(numFeatures);
}