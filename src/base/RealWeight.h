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
    RealWeight(double value = kZero) : _val(value), _sign(0) {}
    
    inline double value() const { return _val; }
    
    inline int sign() const { return (int)_sign; }
    
    inline RealWeight plus(const double d) const { return _val + d; }
    
    inline RealWeight times(const double d) const { return _val * d; }
    
    inline void plusEquals(const double d) { _val += d; }
    
    inline void timesEquals(const double d) { _val *= d; }
    
    LogWeight convert() const;
    
    operator double() const { return _val; }
    
    friend RealWeight operator-(const RealWeight& v);
    
    friend ostream& operator<<(ostream& out, const RealWeight& w);
    
    
    static const double kZero;
    
    static const double kOne;
    
  private:
    double _val;
    
    // Not used in RealWeight, but needed so that we can cast between arrays
    // of RealWeight and LogWeight.
    char _sign;
    
};

#endif
