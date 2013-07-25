/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _OPDELETE_H
#define _OPDELETE_H

#include "EditOperation.h"
#include <boost/regex.hpp>
#include <string>
#include <vector>

class StateType;

class OpDelete : public EditOperation {
  public:
    OpDelete(int opId, const StateType* defaultDestinationState,
      std::string name = "Delete", int phraseLengthSource = 1);
      
    virtual ~OpDelete() {}
    
    virtual const StateType* apply(const std::vector<std::string>& source,
                                   const std::vector<std::string>& target,
                                   const StateType* prevStateType,
                                   const int i,
                                   const int j,
                                   int& iNew,
                                   int& jNew) const;
              
    void setCondition(std::string tokenRegexStr, bool acceptMatching = true);
              
  private:
  
    int _phraseLengthSource;
    
    // If _acceptMatching is true, then all tokens (i.e., in a phrase) must
    // match this regex in order for the the apply method to return true;
    // otherwise, all tokens must not match the regex.
    boost::regex _tokenRegex;
    
    // True if a non-empty std::string was passed to the constructor via the
    // tokensMatchRegex argument.
    bool _conditionEnabled;
    
    bool _acceptMatching;
};

#endif
