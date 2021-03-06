/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _STRINGPAIR_H
#define _STRINGPAIR_H

#include "FeatureGenConstants.h"
#include "Pattern.h"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// A data structure that stores a pair of strings, and provides simple methods
// for accessing them.
class StringPair : public Pattern {
  public:
    StringPair(std::vector<std::string> source, std::vector<std::string> target) :
      _source(source), _target(target), _hasBeginEnd(false) {
      // We only need to check for begin/end markers in the source string, since
      // they are either present in both or in neither.
      setHasBeginEnd(_source);
      updateHashString();
    }
      
    // Assume the source and target strings are arrays of characters (i.e.,
    // there are no "phrase-like" characters that span more than one position).
    StringPair(std::string source, std::string target) : _hasBeginEnd(false) {
      for (std::size_t i = 0; i < source.size(); ++i)
        _source.push_back(source.substr(i, 1));
      for (std::size_t i = 0; i < target.size(); ++i)
        _target.push_back(target.substr(i, 1));
      setHasBeginEnd(_source);
      updateHashString();
    }
    
    virtual const std::vector<std::string>& getSource() const;

    virtual const std::vector<std::string>& getTarget() const;
    
    // Returns the length of the longer string.
    virtual int getSize() const;
    
    virtual std::string getHashString() const;
    
    void updateHashString();
    
    friend std::ostream& operator<<(std::ostream& out, const StringPair& sp);


  protected:
    std::vector<std::string> _source;

    std::vector<std::string> _target;
    
    bool _hasBeginEnd;
    
    std::string _hashString;
    
    void setHasBeginEnd(std::vector<std::string> source);
};

inline const std::vector<std::string>& StringPair::getSource() const {
  return _source;
}

inline const std::vector<std::string>& StringPair::getTarget() const {
  return _target;
}

inline int StringPair::getSize() const {
  int size = _source.size() > _target.size() ? _source.size() : _target.size();
  if (_hasBeginEnd)
    return size - 2; // we don't want to count the begin/end markers
  return size;
}

inline std::ostream& operator<<(std::ostream& out, const StringPair& sp) {
  for (std::size_t i = 0; i < sp._source.size(); ++i)
    out << sp._source[i] << " ";
  out << "--> ";
  for (std::size_t i = 0; i < sp._target.size(); ++i)
    out << sp._target[i] << " ";
  return out;
}

// Locate the first and last characters that are not epsilon symbols. If these
// are the BEGIN_CHAR and END_CHAR symbols, respectively, then we set the flag
// to true.
inline void StringPair::setHasBeginEnd(std::vector<std::string> vecStr) {
  int firstNonEpsilon = -1;
  int lastNonEpsilon = -1;
  for (int i = 0; i < vecStr.size(); i++)
    if (vecStr[i] != FeatureGenConstants::EPSILON) {
      firstNonEpsilon = i;
      break;
    }
  for (int i = vecStr.size()-1; i >= 0; i--)
    if (vecStr[i] != FeatureGenConstants::EPSILON) {
      lastNonEpsilon = i;
      break;
    }
  assert(firstNonEpsilon >= 0 && lastNonEpsilon >= 0);
  if (vecStr[firstNonEpsilon] == FeatureGenConstants::BEGIN_CHAR) {
    if (vecStr[lastNonEpsilon] != FeatureGenConstants::END_CHAR) {
      std::cout << "Warning: String has begin marker but no end marker.\n";
      _hasBeginEnd = false;
    }
    else
      _hasBeginEnd = true;
  }
  else
    _hasBeginEnd = false;
}

inline std::string StringPair::getHashString() const {
  return _hashString;
}

inline void StringPair::updateHashString() {
  std::stringstream ss;
  for (std::size_t i = 0; i < _source.size(); ++i)
    ss << _source[i] << FeatureGenConstants::PART_SEP;
  ss << FeatureGenConstants::WORDFEAT_SEP;
  for (std::size_t i = 0; i < _target.size(); ++i)
    ss << _target[i] << FeatureGenConstants::PART_SEP;
  _hashString = ss.str();
}

#endif
