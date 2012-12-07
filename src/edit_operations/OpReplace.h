/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPREPLACE_H
#define _OPREPLACE_H

#include "EditOperation.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>

// Replace is an operation that generalizes Substitute and Match. That is,
// Replace is applied regardless of whether the source phrase and the target
// phrase are identical or distinct.
class OpReplace : public EditOperation {

  public:
  
    OpReplace(int opId, const StateType* defaultDestinationState,
      std::string name = "Replace", int phraseLengthSource = 1,
      int phraseLengthTarget = 1);
    
    virtual ~OpReplace() {}
    
    virtual const StateType* apply(const std::vector<std::string>& source,
                                   const std::vector<std::string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const;
              
    void setCondition(std::string tokenRegexStrSource,
      std::string tokenRegexStrTarget, bool acceptMatchingSource = true,
      bool acceptMatchingTarget = true);
      
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
