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

#include <boost/foreach.hpp>
#include <string>
#include <vector>

#include "Pattern.h"

// A data structure that stores a pair of strings, and provides simple methods
// for accessing them.
class StringPair : public Pattern {
  public:
    StringPair(std::vector<std::string> source, std::vector<std::string> target) :
      _source(source), _target(target) {}
      
    // Assume the source and target strings are arrays of characters (i.e.,
    // there are no "phrase-like" characters that span more than one position).
    StringPair(std::string source, std::string target) {
      for (std::size_t i = 0; i < source.size(); ++i)
        _source.push_back(source.substr(i, 1));
      for (std::size_t i = 0; i < target.size(); ++i)
        _target.push_back(target.substr(i, 1));
    }
    
    virtual const std::vector<std::string>& getSource() const;

    virtual const std::vector<std::string>& getTarget() const;
    
    // Returns the length of the longer string.
    virtual int getSize() const;
    
    friend std::ostream& operator<<(std::ostream& out, const StringPair& sp);


  protected:
    std::vector<std::string> _source;

    std::vector<std::string> _target;

};

inline const std::vector<std::string>& StringPair::getSource() const {
  return _source;
}

inline const std::vector<std::string>& StringPair::getTarget() const {
  return _target;
}

inline int StringPair::getSize() const {
  return _source.size() > _target.size() ? _source.size() : _target.size();
}

inline std::ostream& operator<<(std::ostream& out, const StringPair& sp) {
  for (std::size_t i = 0; i < sp._source.size(); ++i)
    out << sp._source[i] << " ";
  out << "--> ";
  for (std::size_t i = 0; i < sp._target.size(); ++i)
    out << sp._target[i] << " ";
  return out;
}

#endif
