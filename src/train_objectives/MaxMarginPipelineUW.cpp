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
#include "MaxMarginPipelineUW.h"
#include "Model.h"
#include "Parameters.h"
#include "StringPairAligned.h"
#include "Ublas.h"
#include <algorithm>
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace boost;
using namespace std;

void MaxMarginPipelineUW::valueAndGradientPart(const Parameters& theta,
    Model& model, const Dataset::iterator& begin, const Dataset::iterator& end,
    const Label k, double& funcVal, SparseRealVec& gradFv) {
  assert(0); // not yet implemented
}

// Parses a string of the form:
//
// Mat11 Del1 Ins1 Mat11 Mat11 Mat11 Del1 Mat11 
// |^|j| |a|k|e|n|$
// |^| |s|a|k|e| |$
//
// and returns a StringPairAligned object.
StringPairAligned toStringPairAligned(const string& alignmentString) {
  typedef tokenizer<char_separator<char> > Tokenizer;
  char_separator<char> newlineSep("\n");
  char_separator<char> pipeSep("|");
  Tokenizer lines(alignmentString, newlineSep);
  Tokenizer::const_iterator line = lines.begin();
  ++line; // skip the sequence of edit operations
  
  // We define the edit distance to be the total number of epsilon symbols
  // that appear in the source and target strings. Note that this is not
  // necessarily the same as the edit distance computed by
  // Utility::levenshtein, which returns the total cost of the edits. But
  // since INS and DEL each have unit cost, we will get the same result.
  int numEpsilons = 0;
  
  Tokenizer tokens(*line, pipeSep);
  Tokenizer::const_iterator t;
  vector<string> source;
  int lenSource = 0;
  for (t = tokens.begin(); t != tokens.end(); ++t) {
    string s = *t;
    trim(s);
    if (s.size() == 0) {
      source.push_back(FeatureGenConstants::EPSILON);
      numEpsilons++;
    }
    else {
      source.push_back(s);
      lenSource++;
    }
  }
  ++line;
  
  tokens = Tokenizer(*line, pipeSep);
  vector<string> target;
  int lenTarget = 0;
  for (t = tokens.begin(); t != tokens.end(); ++t) {
    string s = *t;
    trim(s);
    if (s.size() == 0) {
      target.push_back(FeatureGenConstants::EPSILON);
      numEpsilons++;
    }
    else {
      target.push_back(s);
      lenTarget++;
    }
  }
  
  return StringPairAligned(source, target, max(lenSource, lenTarget),
      numEpsilons);
}

void MaxMarginPipelineUW::predictPart(const Parameters& theta, Model& model,
    const Dataset::iterator& begin, const Dataset::iterator& end,
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
    StringPairAligned xPrime = toStringPairAligned(align_ss.str());
    
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
