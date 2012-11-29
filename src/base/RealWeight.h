/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012 Kenneth Dwyer
 */

#ifndef _REALWEIGHT_H
#define _REALWEIGHT_H

#include <ostream>
using std::ostream;

class LogWeight;

class RealWeight {

  public:
    RealWeight(double value = 0) : _val(value) {}
    
    operator double() const { return _val; }
    
    LogWeight convert() const;
    
    const RealWeight operator+(const RealWeight& w) const;
    
    const RealWeight operator*(const RealWeight& w) const;
    
    RealWeight& operator+=(const RealWeight& w);
    
    RealWeight& operator*=(const RealWeight& w);
    
    friend RealWeight operator-(const RealWeight& v);
    
    friend ostream& operator<<(ostream& out, const RealWeight& w);
    
  private:
    inline RealWeight plus(const RealWeight& d) const { return _val + d._val; }
    
    inline RealWeight times(const RealWeight& d) const { return _val * d._val; }
    
    inline void plusEquals(const RealWeight& d) { _val += d._val; }
    
    inline void timesEquals(const RealWeight& d) { _val *= d._val; }
    
    double _val;
};

#endif
