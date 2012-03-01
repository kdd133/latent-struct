/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _DATASET_H
#define _DATASET_H

#include "Example.h"
#include <assert.h>
#include <list>
#include <set>
#include <vector>
using namespace std;


class Dataset {

  public:
  
    typedef list<Example>::const_iterator iterator;
  
    Dataset(size_t partitions = 1);
    
    void addExample(const Example& ex);
    
    size_t numExamples() const;
    
    size_t numPartitions() const;
    
    size_t partitionSize(size_t i) const;
    
    iterator partitionBegin(size_t i) const;
    
    iterator partitionEnd(size_t i) const;
    
    void clear();

    const vector<Example>& getExamples() const;
    
    const set<Label>& getLabelSet() const;
    
    void setLabelSet(set<Label> labels);

  private:
  
    size_t _numPartitions;
    
    vector<Example> _examples;
    
    vector<list<Example> > _partitions;
    
    set<Label> _labels;
};

inline size_t Dataset::numExamples() const {
  return _examples.size();
}

inline size_t Dataset::numPartitions() const {
  assert(_partitions.size() == _numPartitions);
  return _numPartitions;
}

inline Dataset::iterator Dataset::partitionBegin(size_t i) const {
  return _partitions[i].begin();
}

inline Dataset::iterator Dataset::partitionEnd(size_t i) const {
  return _partitions[i].end();
}

inline size_t Dataset::partitionSize(size_t i) const {
  return _partitions[i].size();
}

inline const vector<Example>& Dataset::getExamples() const {
  return _examples;
}

inline const set<Label>& Dataset::getLabelSet() const {
  return _labels;
}

inline void Dataset::setLabelSet(set<Label> labels) {
  _labels = labels;
}

#endif
