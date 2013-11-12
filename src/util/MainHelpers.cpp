/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "AlignmentFeatureGen.h"
#include "Alphabet.h"
#include "Dataset.h"
#include "MainHelpers.h"
#include "Model.h"
#include "Parameters.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "WeightVector.h"
#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include <set>
#include <vector>

using namespace boost;
using namespace std;

void initWeights(WeightVector& w, const string& initType, double noiseLevel,
    int seed, const shared_ptr<Alphabet> alphabet, const set<Label>& labels,
    const shared_ptr<const AlignmentFeatureGen> fgen) {
  w.zero();
  if (initType != "zero") {
    const int d = w.getDim();
    double* v = new double[d];
    for (int i = 0; i < d; i++)
      v[i] = 0;    
    if (istarts_with(initType, "heuristic")) {
      set<Label>::const_iterator labelIt;
      for (labelIt = labels.begin(); labelIt != labels.end(); ++labelIt) {
        const Alphabet::DictType& dict = alphabet->getDict();
        Alphabet::DictType::const_iterator it;
        for (it = dict.begin(); it != dict.end(); ++it) {
          const int index = alphabet->lookup(it->first, *labelIt, false);
          assert(index >= 0);
          v[index] += fgen->getDefaultFeatureWeight(it->first, *labelIt);
        }
      }
    }
    if (iends_with(initType, "noise")) {
      shared_array<double> samples = Utility::generateGaussianSamples(d, 0,
          noiseLevel, seed);
      for (int i = 0; i < d; i++)
        v[i] += samples[i];
    }
    w.setWeights(v, d);
    delete[] v;
  }
}

void evaluateMultipleWeightVectors(const vector<Parameters>& weightVectors,
    const Dataset& evalData, shared_ptr<TrainingObjective> objective,
    const string& path, int id, bool writeFiles, bool writeAlignments,
    bool cachingEnabled) {
  vector<string> identifiers;
  vector<string> fnames;
  for (size_t wvIndex = 0; wvIndex < weightVectors.size(); wvIndex++) {
    stringstream fname;
    if (writeFiles) {
      fname << path << wvIndex << "-eval" << id << "_predictions.txt";
      fnames.push_back(fname.str());
    }
    else
      fnames.push_back("");
    stringstream identifier;
    identifier << wvIndex << "-Eval";
    identifiers.push_back(identifier.str());
    
    if (writeAlignments) {
      // FIXME: This does not make use of multiple threads or of caching. It
      // should probably be performed alongside the eval predictions. 
      stringstream alignFname;
      alignFname << path << wvIndex << "-eval" << id << "_alignments_yi.txt";
      ofstream alignOut(alignFname.str().c_str());
      if (!alignOut.good()) {
        cout << "Warning: Unable to write " << alignFname.str() << endl;
        continue;
      }
      Model& model = objective->getModel(0);
      assert(!model.getCacheEnabled()); // this would waste memory
      const Parameters& params = weightVectors[wvIndex];
      cout << "Printing alignments to " << alignFname.str() << ".\n";
      BOOST_FOREACH(const Example& ex, evalData.getExamples()) {
        BOOST_FOREACH(const Label y, evalData.getLabelSet()) {
          if (objective->isBinary() && y != 1)
            continue;
          alignOut << ex.x()->getId() << " (yi = " << ex.y() << ")  y = "
              << y << endl;
          // FIXME: Will it always be the case that a U-W objective uses *only*
          // the u parameters to impute the latent variables?
          shared_ptr<vector<shared_ptr<SparseRealVec> > > maxFvs;
          if (params.hasU())
            model.getBestAlignments(alignOut, maxFvs, params.u, *ex.x(), y);
          else
            model.getBestAlignments(alignOut, maxFvs, params.w, *ex.x(), y);
          alignOut << endl;
        }
      }
      alignOut.close();
    }
  }
  Utility::evaluate(weightVectors, objective, evalData, identifiers, fnames,
      cachingEnabled);
}
