/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "BiasFeatureGen.h"
#include "FeatureGenConstants.h"
#include "Label.h"
#include "LogWeight.h"
#include "Pattern.h"
#include "Ublas.h"
#include <boost/program_options.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <set>
#include <sstream>
#include <string>

using namespace boost;
using namespace std;

const string BiasFeatureGen::kPrefix = "Bias";

BiasFeatureGen::BiasFeatureGen(shared_ptr<Alphabet> alphabet, bool normalize) :
    ObservedFeatureGen(alphabet), _normalize(normalize) {}

int BiasFeatureGen::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options(name() + " options");
  bool noNormalize = false;
  options.add_options()
    ("bias-no-normalize", opt::bool_switch(&noNormalize),
        "do not normalize by the length of the longer pattern")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  if (vm.count("help")) {
    cout << options << endl;
    return 0;
  }
  
  if (noNormalize) {
    _normalize = false;
  }
  
  return 0;
}

SparseRealVec* BiasFeatureGen::getFeatures(const Pattern& x, const Label y) {  
  stringstream ss;
  ss << y << FeatureGenConstants::PART_SEP << kPrefix;
  const int fId = _alphabet->lookup(ss.str(), true);
  
  const size_t d = _alphabet->size();
  SparseRealVec* fv = new SparseRealVec(d);
  
  if (fId == -1) {
    // This should only ever happen if there's a class in the test set that
    // didn't appear in the training set.
    return fv; // return the zero vector
  }
  
  if (fId >= d) {
    // Assume we are in feature gathering mode.
    return fv;
  }

  (*fv)(fId) = 1;

  if (_normalize) {
    const double normalization = x.getSize();
    assert(normalization > 0);
    (*fv) /= normalization;
  }
  
  return fv;
}
