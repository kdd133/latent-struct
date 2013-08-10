/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _VALIDATIONSETHANDLER_H
#define _VALIDATIONSETHANDLER_H

#include "Parameters.h"
#include <boost/shared_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <lbfgs.h>
#include <string>

class Dataset;
class LabelScoreTable;
class TrainingObjective;

class ValidationSetHandler {

  public:
  
    ValidationSetHandler(boost::shared_ptr<Dataset> dataset,
        boost::shared_ptr<TrainingObjective> objective);
        
    virtual int processOptions(int argc, char** argv);
    
    const Parameters& getBestParams() const {
      return _thetaBest;
    }
    
    double getBestScore() const {
      return _scoreBest;
    }
    
    void clearBest();
    
    std::string getPerfMeasure() const {
      return _perfMeasure;
    }
    
    void evaluate(const Parameters& theta, int iterationNum);
    
    bool wasEvaluated() const {
      return _wasEvaluated;
    }
    
  private:
  
    boost::shared_ptr<Dataset> _validationSet;
    
    boost::shared_ptr<TrainingObjective> _objective;
    
    boost::shared_ptr<LabelScoreTable> _labelScores;
    
    Parameters _thetaBest;
    
    double _scoreBest;
    
    std::string _perfMeasure;
    
    bool _quiet;
    
    // Value is true if the evaluate() method was called at least once. Gets
    // reset to false whenever clearBest() is called.
    bool _wasEvaluated;
};

#endif
