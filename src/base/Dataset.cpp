/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#include "Dataset.h"
#include "Example.h"
#include <assert.h>
#include <list>
#include <vector>

using namespace std;

Dataset::Dataset(size_t partitions) : _numPartitions(partitions), _maxId(0) {
  for (size_t i = 0; i < _numPartitions; i++)
    _partitions.push_back(list<Example>());
}

void Dataset::addExample(const Example& ex) {
  _examples.push_back(ex);
  if (ex.x()->getId() > _maxId)
    _maxId = ex.x()->getId(); 
  _labels.insert(ex.y());
  _partitions[_examples.size() % _numPartitions].push_back(ex);
}

void Dataset::clear() {
  _examples.clear();
  //_labels.clear(); // Keep the label set for e.g., copying from train to eval
  for (size_t i = 0; i < _numPartitions; i++)
    _partitions[i].clear();
  _maxId = 0;
}

void Dataset::addLabels(const set<Label>& labels) {
  for (set<Label>::const_iterator it = labels.begin(); it != labels.end(); ++it)
    _labels.insert(*it);
}
