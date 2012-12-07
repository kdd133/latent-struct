/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Alphabet.h"
#include <assert.h>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>

using namespace std;

// If the given string is contained in the alphabet, return its index.
// Otherwise, if addIfAbsent is true, add the string to the alphabet and return
// its index. If the string is absent and addIfAbsent=false or _locked=true,
// return -1.
int Alphabet::lookup(string key, bool addIfAbsent) {
  int index = -1;
  DictType::const_iterator it = _dict.find(key);
  if (it != _dict.end()) {
    index = it->second;
    assert(index >= 0);
  }
  else if (addIfAbsent && !_locked) {
    index = _entries.size();
    pair<DictType::iterator, bool> ret = _dict.insert(PairType(key, index));
    assert(ret.second); // will be false if key already present in _dict
    _entries.push_back(key);
  }
  
  if (_count && !_locked) {
    int newCount;
    DictType::iterator iter = _counts.find(key);
    if (iter == _counts.end())
      newCount = 1;
    else {
      newCount = iter->second + 1;
      _counts.erase(iter);
    }
    _counts.insert(PairType(key, newCount));
  }
  
  return index;
}

bool Alphabet::read(const std::string& fname) {
  using namespace boost;
  using namespace std;
  assert(_entries.size() == 0);
  _counts.clear();
  _dict.clear();
  _entries.clear();
  ifstream fin(fname.c_str(), ifstream::in);
  if (!fin.good())
    return false;
  char_separator<char> spaceSep(" ");
  string line;
  while (getline(fin, line)) {
    tokenizer<char_separator<char> > fields(line, spaceSep);
    tokenizer<char_separator<char> >::const_iterator it = fields.begin();
    int index = lexical_cast<int>(*it++);
    string key = *it++;
    assert(it == fields.end());
    pair<DictType::iterator, bool> ret = _dict.insert(PairType(key, index));
    assert(ret.second); // will be false if key already present in _dict
    _entries.push_back(key);
  }
  fin.close();
  return true;
}

bool Alphabet::write(const string& fname) const {
  ofstream fout(fname.c_str());
  if (!fout.good())
    return false;
  BOOST_FOREACH(const PairType& entry, _dict) {
    fout << entry.second << " " << entry.first << endl;
  }
  fout.close();
  return true;
}

const Alphabet::DictType& Alphabet::getDict() const {
  return _dict;
}
