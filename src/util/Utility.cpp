/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "Dataset.h"
#include "FeatureVector.h"
#include "InputReader.h"
#include "LabelScoreTable.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "RealWeight.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
using namespace std;

const double Utility::log1PlusTiny = 1e-4;

bool Utility::loadDataset(const InputReader& reader, string fileName,
    Dataset& dataset) {
  ifstream fin(fileName.c_str());
  if (!fin.good())
    return true;
  Pattern* pattern = 0;
  Label label;
  size_t i = 0;
  string line;
  while (getline(fin, line)) {
    reader.readExample(line, pattern, label);
    assert(pattern != 0);
    pattern->setId(i++);
    Example ex(pattern, label);
    dataset.addExample(ex);
    pattern = 0;
  }
  fin.close();
  return false;
}

// Add L2 regularization.
void Utility::addRegularizationL2(const WeightVector& W, const double beta,
    double& fval, FeatureVector<RealWeight>& grad) {
  assert(beta > 0.0);
  fval += beta/2 * W.squaredL2Norm();
  grad.plusEquals(W.getWeights(), W.getDim(), beta); // add beta*W to gradient
}

void Utility::evaluate(const vector<WeightVector>& weightVectors,
    TrainingObjective& obj, const Dataset& evalData,
    const vector<string>& identifiers, const vector<string>& fnames,
    bool caching) {
    
  assert(obj.getModel(0).getFgenLatent()->getAlphabet()->isLocked());
  assert(obj.getModel(0).getFgenObserved()->getAlphabet()->isLocked());
  assert(evalData.getLabelSet().size() > 1);
  
  const size_t numWeightVectors = weightVectors.size();
  assert(numWeightVectors > 0);
  assert(numWeightVectors == identifiers.size());
  
  // Determine the maximum pattern id in the eval data.
  size_t maxId = 0;
  BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
    // TODO: Sizing the matrix based on the max id in the dataset rather than
    // on the number of examples is convenient, but can be wasteful of memory
    // if the ids are not contiguous.
    const size_t id = ex.x()->getId();
    if (id > maxId)
      maxId = id;
  }
  
  // Create a table, one for each weight vector, that will store the predictions
  // for that weight vector.
  boost::ptr_vector<LabelScoreTable> labelScores;
  for (size_t i = 0; i < numWeightVectors; i++) {
    labelScores.push_back(new LabelScoreTable(maxId + 1,
        evalData.getLabelSet().size()));
  }
  
  // Ensure that caching is enabled (if requested), and that the cache is empty.
  for (size_t mi = 0; mi < obj.getNumModels(); mi++) {
    obj.getModel(mi).setCacheEnabled(caching);
    obj.getModel(mi).emptyCache();
  }
  
  // The dataset has already been partitioned into numThreads, but in order to
  // parallelize computations with caching enabled, we would end up storing fsts
  // for the entire eval set in memory. Instead, we'll sub-partition each part
  // and parallelize across each sub-partition, clearing the cache after making
  // all the predictions for each.
  const size_t numParts = evalData.numPartitions();
  for (size_t i = 0; i < numParts; i++) {
    Dataset partData(numParts);
    const Dataset::iterator begin = evalData.partitionBegin(i);
    const Dataset::iterator end = evalData.partitionEnd(i);
    for (Dataset::iterator it = begin; it != end; ++it) {
      partData.addExample(*it);
    }
    partData.addLabels(evalData.getLabelSet());
    assert(partData.getLabelSet().size() == evalData.getLabelSet().size());
    obj.predict(weightVectors[0], partData, labelScores[0]);
    
    // We have now cached the fsts for the first (original) partition of the
    // data, which can be reused for predicting with the other weight vectors.
    for (size_t wvIndex = 1; wvIndex < numWeightVectors; wvIndex++)
      obj.predict(weightVectors[wvIndex], partData, labelScores[wvIndex]);
    
    // Clear the cache (we don't need the fsts for this partition any more).
    if (caching) {
      for (size_t mi = 0; mi < obj.getNumModels(); mi++)
        obj.getModel(mi).emptyCache();
    }
  }

  // Either we're not writing files, or we're writing all of them.
  assert(fnames.size() == 0 || numWeightVectors == fnames.size());
  
  for (size_t wvIndex = 0; wvIndex < numWeightVectors; wvIndex++) {
    printResults(evalData, labelScores[wvIndex], identifiers[wvIndex],
        fnames[wvIndex]);
  }
}

void Utility::evaluate(const WeightVector& w, TrainingObjective& obj,
    const Dataset& evalData, const string& identifier, const string& fname) {
    
  assert(obj.getModel(0).getFgenLatent()->getAlphabet()->isLocked());
  assert(obj.getModel(0).getFgenObserved()->getAlphabet()->isLocked());
  
  size_t maxId = 0;
  BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
    // TODO: Sizing the matrix based on the max id in the dataset rather than
    // on the number of examples is convenient, but can be wasteful of memory
    // if the ids are not contiguous.
    const size_t id = ex.x()->getId();
    if (id > maxId)
      maxId = id;
  }
  
  LabelScoreTable labelScores(maxId + 1, evalData.getLabelSet().size());
  obj.predict(w, evalData, labelScores);
  printResults(evalData, labelScores, identifier, fname);
}

void Utility::printResults(const Dataset& evalData,
    LabelScoreTable& labelScores, const string& id, const string& fname) {
    
  const bool writeFiles = (fname.size() > 0);
  ofstream fout;
  if (writeFiles) {
    fout.open(fname.c_str());
    assert(fout.good());
  }
  
  const Label yPos = TrainingObjective::kPositive;
  int numErrors = 0;
  int pp = 0;
  int ppCorrect = 0;
  int tp = 0;  
  BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
    const size_t id = ex.x()->getId();
    const Label yi = ex.y();    
    
    Label yHat = 0;
    double score_yHat = -numeric_limits<double>::infinity();
    BOOST_FOREACH(const Label y, evalData.getLabelSet()) {
      const double score_y = labelScores.getScore(id, y);
      if (score_y > score_yHat) {
        yHat = y;
        score_yHat = score_y;
      }
    }
    
    if (yi != yHat)
      numErrors++;      
    if (yHat == yPos) {
      pp++;
      if (yi == yPos)
        ppCorrect++;
    }
    if (yi == yPos)
      tp++;
      
    if (writeFiles) {
      fout << ex.x()->getId() << " y " << ex.y() << " yHat " << yHat;
      fout << " scores";
      BOOST_FOREACH(const Label y, evalData.getLabelSet()) {
        fout << " " << y << " " << labelScores.getScore(id, y);
      }
      fout << endl;
    }
  }
  if (writeFiles)
    fout.close();

  const char* id_ = id.c_str();
  const int t = evalData.numExamples();
  const int correct = t - numErrors;
  const double accuracy = (double) correct / t;
  printf("%s-Accuracy: %.4f ( %d of %d )\n", id_, accuracy, correct, t);

  double precision = (pp == 0) ? 0 : (double) ppCorrect / pp;
  printf("%s-Precision: %.4f ( %d of %d )\n", id_, precision, ppCorrect, pp);

  float recall = (tp == 0) ? 0 : (double) ppCorrect / tp;
  printf("%s-Recall: %.4f ( %d of %d )\n", id_, recall, ppCorrect, tp);

  double fscore = (precision + recall == 0) ? 0 : 2 * ((precision * recall)
      / (precision + recall));
  printf("%s-Fscore: %.4f\n", id_, fscore);
}
