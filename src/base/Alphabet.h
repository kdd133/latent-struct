/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _ALPHABET_H
#define _ALPHABET_H

#include "Label.h"
#include <assert.h>
#include <boost/unordered_map.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

class Alphabet {
  
  public:
  
    typedef boost::unordered_map<std::string,int> DictType;
    typedef DictType::value_type PairType;
  
    Alphabet(bool locked = false, bool count = false) : _locked(locked),
      _count(count) { }
    
    // If the given string is contained in the alphabet, return its index.
    // Otherwise, if addIfAbsent is true, add the string to the alphabet and
    // return its index. If the string is absent and addIfAbsent=false or
    // locked=true, return -1.
    int lookup(std::string feat, Label label, bool addIfAbsent);
    
    // Return the string that is associated with the given index.
    std::string reverseLookup(std::size_t index) const;
    
    bool isLocked() const;
  
    // Disallows additions to this alphabet.
    void lock();
  
    size_t size() const;
    
    size_t numFeaturesPerClass() const;
    
    bool read(const std::string& fname);
    
    bool write(const std::string& fname) const;
    
    const DictType& getDict() const;
    
    void addLabel(Label label);
    
  private:
    
    DictType _dict;

    std::vector<std::string> _entries;

    bool _locked;
    
    bool _count;
    
    DictType _counts;
    
    std::vector<int> _labelIndices;
    
    std::set<Label> _uniqueLabels;
};

inline std::string Alphabet::reverseLookup(std::size_t index) const {
  assert(index < _entries.size());
  return _entries.at(index);
}

inline bool Alphabet::isLocked() const {
  return _locked;
}

inline const Alphabet::DictType& Alphabet::getDict() const {
  return _dict;
}

#endif
