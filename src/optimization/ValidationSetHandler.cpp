/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#include "Dataset.h"
#include "LabelScoreTable.h"
#include "TrainingObjective.h"
#include "Utility.h"
#include "ValidationSetHandler.h"
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdio>
#include <limits>

using namespace boost;
using namespace std;

ValidationSetHandler::ValidationSetHandler(shared_ptr<Dataset> dataset,
    shared_ptr<TrainingObjective> objective) : _validationSet(dataset),
    _objective(objective), _scoreBest(-numeric_limits<double>::infinity()),
    _perfMeasure("11pt_avg_prec"), _quiet(false) {
  
  // Initialize a data structure that will be used to store the predictions made
  // on the validation set. 
  _labelScores.reset(new LabelScoreTable(_validationSet->getMaxId() + 1,
      _objective->getDataset().getLabelSet().size()));
}

void ValidationSetHandler::clearBest() {
  _thetaBest = Parameters();
  _scoreBest = -numeric_limits<double>::infinity();
}

void ValidationSetHandler::evaluate(const Parameters& theta, int iter) {
  double accuracy, precision, recall, fscore, avg11ptPrec;
  timer::cpu_timer clock;
  if (!_quiet) {
    cout << "ValidationSetHandler: Predicting on validation set examples...  ";
    cout.flush();
  }
  _objective->predict(theta, *_validationSet, *_labelScores);
  Utility::calcPerformanceMeasures(*_validationSet, *_labelScores, false, "",
      "", accuracy, precision, recall, fscore, avg11ptPrec);
      
  double* score = &fscore;
  if (_perfMeasure == "accuracy")
    score = &accuracy;
  else if (_perfMeasure == "11pt_avg_prec")
    score = &avg11ptPrec;
  if (*score > _scoreBest) {
    _scoreBest = *score;
    _thetaBest.setParams(theta);
  }
  
  if (!_quiet) {
    printf("t = %d  acc = %.3f  prec = %.3f  rec = %.3f  ", iter, accuracy,
        precision, recall);
    printf("fscore = %.3f  11ptAvgPrec = %.3f  best = %.3f %s", fscore,
        avg11ptPrec, _scoreBest, clock.format().c_str());
  }
}

int ValidationSetHandler::processOptions(int argc, char** argv) {
  namespace opt = boost::program_options;
  opt::options_description options("ValidationSetHandler options");
  options.add_options()
    ("performance-measure", opt::value<string>(&_perfMeasure)->default_value(
        "11pt_avg_prec"), "the statistic that determines the 'best' set of \
parameters, determined on a validation set {accuracy, fscore, 11pt_avg_prec}")
    ("quiet", opt::bool_switch(&_quiet), "suppress output")
    ("help", "display a help message")
  ;
  opt::variables_map vm;
  opt::store(opt::command_line_parser(argc, argv).options(options)
      .allow_unregistered().run(), vm);
  opt::notify(vm);
  
  to_lower(_perfMeasure);
  if (_perfMeasure != "fscore" && _perfMeasure != "accuracy" &&
      _perfMeasure != "11pt_avg_prec") {
    cout << "Invalid arguments: Unrecognized performance measure\n";
    cout << options << endl;
    return 1;
  }
  
  if (vm.count("help"))
    cout << options << endl;
  return 0;
}
