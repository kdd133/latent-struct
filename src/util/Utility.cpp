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
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ref.hpp>
#include <boost/thread/thread.hpp>
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

void Utility::evaluate(const WeightVector& w, TrainingObjective& obj,
    const Dataset& evalData, const char* id, const string fname) {
  assert(obj.getModel(0).getFgenLatent()->getAlphabet()->isLocked());
  assert(obj.getModel(0).getFgenObserved()->getAlphabet()->isLocked());
  
  const bool writeFiles = (fname.size() > 0);
  ofstream fout;
  if (writeFiles) {
    fout.open(fname.c_str());
    assert(fout.good());
  }
  
  size_t maxId = 0;
  BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
    const size_t id = ex.x()->getId();
    if (id > maxId)
      maxId = id;
  }
  
  // TODO: Sizing the matrix based on the maximum id in the dataset rather than
  // on the number of examples is convenient, but can be wasteful of memory.
  LabelScoreTable labelScores(maxId + 1, evalData.getLabelSet().size());
  obj.predict(w, evalData, labelScores);
  
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

  const int t = evalData.numExamples();
  const int correct = t - numErrors;
  const double accuracy = (double) correct / t;
  printf("%s-Accuracy: %.4f ( %d of %d )\n", id, accuracy, correct, t);

  double precision = (pp == 0) ? 0 : (double) ppCorrect / pp;
  printf("%s-Precision: %.4f ( %d of %d )\n", id, precision, ppCorrect, pp);

  float recall = (tp == 0) ? 0 : (double) ppCorrect / tp;
  printf("%s-Recall: %.4f ( %d of %d )\n", id, recall, ppCorrect, tp);

  double fscore = (precision + recall == 0) ? 0 : 2 * ((precision * recall)
      / (precision + recall));
  printf("%s-Fscore: %.4f\n", id, fscore);
}
