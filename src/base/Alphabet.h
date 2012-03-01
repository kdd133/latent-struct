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

#include <assert.h>
#include <map>
#include <string>
#include <tr1/unordered_map>
#include <vector>
using namespace std;

class Alphabet {

  public:
  
    typedef tr1::unordered_map<string,int> DictType;
    typedef DictType::value_type PairType;
  
    Alphabet(bool locked = false, bool count = false) : _locked(locked),
      _count(count) { }
    
    // If the given string is contained in the alphabet, return its index.
    // Otherwise, if addIfAbsent is true, add the string to the alphabet and
    // return its index. If the string is absent and addIfAbsent=false or
    // locked=true, return -1.
    int lookup(string str, bool addIfAbsent = false);
    
    // Return the string that is associated with the given index.
    string reverseLookup(size_t index);
    
    bool isLocked() const;
  
    // Disallows additions to this alphabet.
    void lock();
  
    // Allows additions to this alphabet.
    void unlock();
  
    size_t size() const;
    
    bool read(const string& fname);
    
    bool write(const string& fname) const;
    
    const DictType& getDict() const;
    
  private:
    
    DictType _dict;

    vector<string> _entries;

    bool _locked;
    
    bool _count;
    
    DictType _counts;
};

inline string Alphabet::reverseLookup(size_t index) {
  assert(index < _entries.size());
  return _entries.at(index);
}

inline bool Alphabet::isLocked() const {
  return _locked;
}

inline void Alphabet::lock() {
  _locked = true;
}

inline void Alphabet::unlock() {
  _locked = false;
}

inline size_t Alphabet::size() const {
  assert(_entries.size() == _dict.size());
  return _entries.size();
}

#endif
