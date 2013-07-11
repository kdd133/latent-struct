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
#include "InputReader.h"
#include "LabelScoreTable.h"
#include "Model.h"
#include "ObservedFeatureGen.h"
#include "Parameters.h"
#include "TrainingObjective.h"
#include "Ublas.h"
#include "Utility.h"
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/shared_array.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <stack>
#include <string>

using namespace boost;
using namespace std;

const double Utility::log1PlusTiny = 1e-4;

bool Utility::loadDataset(const InputReader& reader, string fileName,
    Dataset& dataset, size_t firstId) {
  ifstream fin(fileName.c_str());
  if (!fin.good())
    return true;
  Pattern* pattern = 0;
  Label label;
  size_t i = firstId;
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

void Utility::evaluate(const vector<Parameters>& weightVectors,
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
    double accuracy, precision, recall, fscore, avg11ptPrec;
    calcPerformanceMeasures(evalData, labelScores[wvIndex], true,
        identifiers[wvIndex], fnames[wvIndex],
        accuracy, precision, recall, fscore, avg11ptPrec);
  }
}

void Utility::evaluate(const Parameters& w, TrainingObjective& obj,
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
  
  double accuracy, precision, recall, fscore, avg11ptPrec;
  calcPerformanceMeasures(evalData, labelScores, true, identifier, fname,
      accuracy, precision, recall, fscore, avg11ptPrec);
}

void Utility::calcPerformanceMeasures(const Dataset& evalData,
    LabelScoreTable& labelScores, bool toStdout,
    const string& id, const string& fname,
    double& accuracy, double& precision, double& recall, double& fscore,
    double& avg11ptPrec) {
    
  bool writeFiles = (fname.size() > 0);
  ofstream fout;
  if (writeFiles) {
    fout.open(fname.c_str());
    if (!fout.good()) {
      cout << "Unable to write " << fname << endl;
      writeFiles = false;
    }
  }
  
  std::vector<prediction> predictions(evalData.numExamples());
  size_t p_i = 0;
  
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
    
    prediction pred = {yi, labelScores.getScore(id, yPos)};
    predictions[p_i++] = pred;
    
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
  
  accuracy = (double) correct / t;
  precision = (pp == 0) ? 0 : (double) ppCorrect / pp;
  recall = (tp == 0) ? 0 : (double) ppCorrect / tp;
  fscore = (precision + recall == 0) ? 0 : 2 * ((precision * recall)
      / (precision + recall));
  avg11ptPrec = avg11ptPrecision(predictions);
      
  if (toStdout) {
    const char* id_ = id.c_str();
    printf("%s-Accuracy: %.4f ( %d of %d )\n", id_, accuracy, correct, t);
    printf("%s-Precision: %.4f ( %d of %d )\n", id_, precision, ppCorrect, pp);
    printf("%s-Recall: %.4f ( %d of %d )\n", id_, recall, ppCorrect, tp);
    printf("%s-Fscore: %.4f\n", id_, fscore);
    printf("%s-11ptAvgPrec: %.4f\n", id_, avg11ptPrec);
  }
}

shared_array<double> Utility::generateGaussianSamples(size_t n, double mean,
    double stdev, int seed) {
  shared_array<double> samples(new double[n]);
  mt19937 mt(seed);
  normal_distribution<> gaussian(mean, stdev);
  variate_generator<mt19937, normal_distribution<> > rgen(mt, gaussian);
  for (size_t i = 0; i < n; ++i)
    samples[i] = rgen();
  return samples;
}

shared_array<int> Utility::randPerm(int n, int seed) {
  mt19937 mt(seed);
  uniform_01<> uniform01;
  variate_generator<mt19937, uniform_01<> > rgen(mt, uniform01);
  shared_array<int> numbers(new int[n]);
  for (int i = 0; i < n; ++i)
    numbers[i] = i;
  for (int i = n - 1; i >= 0; --i) {
    int j = floor(rgen()*(i+1)); // randomly choose j in [0,i]
    int temp = numbers[j];
    numbers[j] = numbers[i];
    numbers[i] = temp;
  }
  return numbers;
}

double Utility::getNumericalGradientForCoordinate(TrainingObjective& obj,
    const Parameters& theta, int i, double epsilon) {
  Parameters thetaPlus(theta.getDimW(), theta.getDimU());
  thetaPlus.setParams(theta);
  thetaPlus.add(i, epsilon);
  
  Parameters thetaMinus(theta.getDimW(), theta.getDimU());
  thetaMinus.setParams(theta);
  thetaMinus.add(i, -epsilon);
  
  SparseRealVec grad(theta.getDimWU()); // this won't be used
  
  double fvalPlus, fvalMinus;
  obj.valueAndGradient(thetaPlus, fvalPlus, grad);
  obj.valueAndGradient(thetaMinus, fvalMinus, grad);
  
  return (fvalPlus - fvalMinus) / (2 * epsilon);
}

int Utility::levenshtein(const vector<string>& s, const vector<string>& t,
    vector<string>& sEps, vector<string>& tEps, int subCost) {
  multi_array<int, 2> cost(extents[s.size()+1][t.size()+1]);
  multi_array<int, 2> lastOp(extents[s.size()+1][t.size()+1]);
  enum {INS, DEL, SUB};
  cost[0][0] = 0;
  for (size_t i = 1; i <= s.size(); ++i) {
    cost[i][0] = i;
    lastOp[i][0] = INS;
  }
  for (size_t j = 1; j <= t.size(); ++j) {
    cost[0][j] = j;
    lastOp[0][j] = DEL;
  }
  
  for (size_t i = 1; i <= s.size(); ++i) {
    for (size_t j = 1; j <= t.size(); ++j) {
      if (s[i-1] == t[j-1]) { // the strings are indexed from zero 
        cost[i][j] = cost[i-1][j-1];
        lastOp[i][j] = SUB;
      }
      else {
        const int ins = cost[i-1][j]+1;
        const int del = cost[i][j-1]+1;
        const int sub = cost[i-1][j-1]+subCost;
        cost[i][j] = ins;
        lastOp[i][j] = INS;
        if (del < cost[i][j]) {
          cost[i][j] = del;
          lastOp[i][j] = DEL;
        }
        if (sub < cost[i][j]) {
          cost[i][j] = sub;
          lastOp[i][j] = SUB;
        }
      }
    }
  }

#if 0
  for (size_t i = 0; i <= s.size(); ++i) {
    for (size_t j = 0; j <= t.size(); ++j) {
      cout << cost[i][j] << " " << lastOp[i][j] << "|";
    }
    cout << endl;
  }
#endif
  
  stack<int> ops;
  size_t i = s.size();
  size_t j = t.size();
  while (i > 0 || j > 0) {
    switch (lastOp[i][j]) {
      case INS:
        ops.push(INS);
        i--;
        break;
      case DEL:
        ops.push(DEL);
        j--;
        break;
      case SUB:
        ops.push(SUB);
        i--;
        j--;
        break;
    }
  }
  
  sEps.clear();
  tEps.clear();
  i = 0;
  j = 0;
  while (!ops.empty()) {
    switch (ops.top()) {
      case INS:
        sEps.push_back(s[i++]);
        tEps.push_back("-");
        break;
      case DEL:
        sEps.push_back("-");
        tEps.push_back(t[j++]);
        break;
      case SUB:
        sEps.push_back(s[i++]);
        tEps.push_back(t[j++]);
        break;
    }
    ops.pop();
  } 

  return cost[s.size()][t.size()];
}

double Utility::avg11ptPrecision(std::vector<prediction>& predictions) {
  // Sort the predictions in descending order by score.
  sort(predictions.rbegin(), predictions.rend());
  
  // Record the number of positive predictions we've made at each point in the
  // list of predictions.
  int posTotal = 0;
  std::vector<Label> positiveCount(predictions.size());
  for (size_t i = 0; i < predictions.size(); i++) {
    if (predictions[i].y == TrainingObjective::kPositive)
      posTotal++;
    positiveCount[i] = posTotal; 
  }
  
  // Compute our precision at each of the 11 recall points.
  std::vector<double> precisionAtRecall(11, 0);
  for (int i = predictions.size()-1; i >= 0; i--) {
    double currentRecall = positiveCount[i] / (double)posTotal;
    double currentPrecision = positiveCount[i] / (double)(i+1);
    for (int ri = 0; ri <= currentRecall * 10; ri++) {
      if (currentPrecision > precisionAtRecall[ri])
        precisionAtRecall[ri] = currentPrecision;
    }
  }
  precisionAtRecall[0] = 1; // by definition
  
  // Return the 11-point average precision.
  double prSum = 0;
  for (int ri = 0; ri < 11; ri++)
    prSum += precisionAtRecall[ri];
  return prSum / 11;
}
