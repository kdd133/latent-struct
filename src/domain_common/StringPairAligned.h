/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Kenneth Dwyer
 */

#ifndef _STRINGPAIRALIGNED_H
#define _STRINGPAIRALIGNED_H

#include "FeatureGenConstants.h"
#include "StringPair.h"
#include "Utility.h"
#include <ostream>
#include <string>
#include <vector>

// A data structure that stores an aligned pair of strings, which have been
// augmented with "-" symbols that indicate insertions and deletions.
class StringPairAligned : public StringPair {
  public:
    StringPairAligned(std::vector<std::string> source,
        std::vector<std::string> target) : StringPair(source, target) {
      _longest = StringPair::getSize(); // record this before inserting epsilons
      _editDistance = Utility::levenshtein(source, target, _source, _target,
          _substitutionCost);
    }
    
    // This constructor is accepts an already-aligned pair of strings; i.e.,
    // epsilon symbols are already present.
    StringPairAligned(std::vector<std::string> source,
        std::vector<std::string> target, int longest, int editDistance) :
        StringPair(source, target), _longest(longest),
        _editDistance(editDistance) {
      if (_hasBeginEnd) // this gets set in the StringPair constructor
        _longest -= 2;
    }
    
    // Returns the length of the longer string (excluding epsilon symbols).
    virtual int getSize() const;
    
    virtual int getEditDistance() const;
    
  private:
    
    static const int _substitutionCost = 3;
    
    // The length of the longest string that was passed to the constructor.
    int _longest;
    
    int _editDistance;
};

inline int StringPairAligned::getSize() const {
  return _longest;
}

inline int StringPairAligned::getEditDistance() const {
  return _editDistance;
}

#endif
