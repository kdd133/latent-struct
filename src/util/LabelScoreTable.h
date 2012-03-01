/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LABELSCORETABLE_H
#define _LABELSCORETABLE_H

#include <boost/multi_array.hpp>
#include <boost/thread/mutex.hpp>

class LabelScoreTable {

  public:
  
    // Note: t is the number of examples, k is the number of classes.
    LabelScoreTable(size_t t, size_t k);
    
    void setScore(size_t i, size_t y, double score);
    
    double getScore(size_t i, size_t y);

  private:
    
    boost::multi_array<double, 2> _scores;
    
    boost::mutex _flag;

};

#endif
