/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Alphabet.h"
#include "Label.h"
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
int Alphabet::lookup(string key, Label label, bool addIfAbsent) {
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
  
  // The dictionary only indexes the feature names, but in actuality there are
  // copies of each feature, one per class label. In order to give the caller
  // the illusion that we actually store all these features, we offset each
  // feature by a multiple of the given class label. The _labelIndices data
  // structure maps each label to an index, thus ensuring that the indices are
  // contiguous even though the labels may not be. For example, a binary
  // classifier might only use y=1, but since it is the only label we need not
  // offset its features by _entries.size() * 1. Instead, the label y=1 will be
  // mapped to index 0 in this case.
  if (addIfAbsent && !_locked) {
    addLabel(label);
  }
  else if (index >= 0) {
    // Note: This assert can fail if the Alphabet is used without being locked.
    assert(label < _labelIndices.size());
    index += _entries.size() * _labelIndices[label];
  }
  
  // Indices are generally not valid until the Alphabet is locked.
  if (!_locked)
    return -1;
  
  return index;
}

void Alphabet::addLabel(Label label) {
  assert(!_locked);
  _uniqueLabels.insert(label);
}

// When the Alphabet is locked, we map the class labels that we've seen to a
// contiguous set of indices (see long comment in lookup() method above).
void Alphabet::lock() {
  int nextIndex = 0;
  set<Label>::const_iterator it;
  for (it = _uniqueLabels.begin(); it != _uniqueLabels.end(); ++it) {
    const Label y = *it;
    while (y > _labelIndices.size())
      _labelIndices.push_back(-1);
    _labelIndices.push_back(nextIndex++);
  }
  _locked = true;
}

size_t Alphabet::size() const {
  assert(_entries.size() == _dict.size());
  assert(_uniqueLabels.size() > 0);
  return _entries.size() * _uniqueLabels.size();
}

size_t Alphabet::numFeaturesPerClass() const {
  assert(_entries.size() == _dict.size());
  assert(_uniqueLabels.size() > 0);
  return _entries.size();
}

bool Alphabet::read(const string& fname) {
  using namespace boost;
  assert(_entries.size() == 0);
  _counts.clear();
  _dict.clear();
  _entries.clear();
  ifstream fin(fname.c_str(), ifstream::in);
  if (!fin.good())
    return false;
  char_separator<char> spaceSep(" ");
  string line;
  
  // First, read a line of space-separated integers that are the entries of the
  // _labelIndices vector.
  {
    assert(_labelIndices.size() == 0);
    getline(fin, line);
    tokenizer<char_separator<char> > fields(line, spaceSep);
    tokenizer<char_separator<char> >::const_iterator it = fields.begin();
    while (it != fields.end())
      _labelIndices.push_back(lexical_cast<int>(*it++));

    // Populate _uniqueLabels with the entries that are not "gaps" (i.e., -1).
    for (size_t i = 0; i < _labelIndices.size(); ++i)
      if (_labelIndices[i] >= 0)
        _uniqueLabels.insert(_labelIndices[i]);
  }
  
  // Populate the feature dictionary.
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
  // The first line of output is the _labelIndices vector.
  assert(_labelIndices.size() > 0);
  fout << _labelIndices[0];
  for (size_t i = 1; i < _labelIndices.size(); ++i)
    fout << " " << _labelIndices[i];
  fout << endl;
  // The remaining output consists of the dictionary entries.
  BOOST_FOREACH(const PairType& entry, _dict)
    fout << entry.second << " " << entry.first << endl;
  fout.close();
  return true;
}
