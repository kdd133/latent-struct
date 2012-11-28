/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _LOGWEIGHT_H
#define _LOGWEIGHT_H

#include <assert.h>
#include <ostream>
using std::ostream;

class RealWeight;

class LogWeight {

  public:
    LogWeight(double value = 0);
    
    LogWeight(const LogWeight& from);
    
    LogWeight& operator=(const LogWeight& from);
    
    inline double toDouble() const { return _val; }
    
    RealWeight convert() const;
    
    const LogWeight operator+(const LogWeight& w) const;
    
    const LogWeight operator*(const LogWeight& w) const;
    
    LogWeight& operator+=(const LogWeight& w);
    
    LogWeight& operator*=(const LogWeight& w);
    
    friend LogWeight operator-(const LogWeight& w);
    
    friend ostream& operator<<(ostream& out, const LogWeight& w);

    friend bool operator==(const LogWeight& a, const LogWeight& b);
    
    friend bool operator!=(const LogWeight& a, const LogWeight& b);
    
  private:
    LogWeight plus(const LogWeight& d) const;
    
    LogWeight times(const LogWeight& d) const;

    void plusEquals(const LogWeight& d);
    
    void timesEquals(const LogWeight& d);
    
    double _val;
};

#endif
