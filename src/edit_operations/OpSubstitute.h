/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPSUBSTITUTE_H
#define _OPSUBSTITUTE_H

#include "EditOperation.h"
#include <boost/regex.hpp>
#include <list>
#include <string>
#include <vector>
using namespace std;

// A Substitute operation is only applied when the two phrases differ.
// (Note: See OpReplace for an operation that is applied unconditionally.) 
class OpSubstitute : public EditOperation {

  public:
  
    OpSubstitute(int opId, const StateType* defaultDestinationState,
      string name = "Substitute", int phraseLengthSource = 1,
      int phraseLengthTarget = 1);
    
    virtual ~OpSubstitute() {}
    
    virtual const StateType* apply(const vector<string>& source,
                                   const vector<string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const;
              
    void setCondition(string tokenRegexStrSource, string tokenRegexStrTarget,
      bool acceptMatchingSource = true, bool acceptMatchingTarget = true);
        
  private:
    
    int _phraseLengthSource;
    
    int _phraseLengthTarget;
    
    boost::regex _tokenRegexSource;
    
    boost::regex _tokenRegexTarget;
    
    bool _conditionEnabledSource;
    
    bool _conditionEnabledTarget;
    
    bool _acceptMatchingSource;
    
    bool _acceptMatchingTarget;
};

#endif
