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


class Dataset {

  public:
  
    typedef std::list<Example>::const_iterator iterator;
  
    Dataset(std::size_t partitions = 1);
    
    void addExample(const Example& ex);
    
    std::size_t numExamples() const;
    
    std::size_t numPartitions() const;
    
    iterator partitionBegin(std::size_t i) const;
    
    iterator partitionEnd(std::size_t i) const;
    
    void clear();

    const std::vector<Example>& getExamples() const;
    
    const std::set<Label>& getLabelSet() const;
    
    void addLabels(const std::set<Label>& labels);

  private:
  
    std::size_t _numPartitions;
    
    std::vector<Example> _examples;
    
    std::vector<std::list<Example> > _partitions;
    
    std::set<Label> _labels;
};

inline std::size_t Dataset::numExamples() const {
  return _examples.size();
}

inline std::size_t Dataset::numPartitions() const {
  assert(_partitions.size() == _numPartitions);
  return _numPartitions;
}

inline Dataset::iterator Dataset::partitionBegin(std::size_t i) const {
  return _partitions[i].begin();
}

inline Dataset::iterator Dataset::partitionEnd(std::size_t i) const {
  return _partitions[i].end();
}

inline const std::vector<Example>& Dataset::getExamples() const {
  return _examples;
}

inline const std::set<Label>& Dataset::getLabelSet() const {
  return _labels;
}

#endif
