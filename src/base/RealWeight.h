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
    RealWeight(double value = kZero) : _val(value) {}
    
    inline double value() const { return _val; }
    
    LogWeight convert() const;
    
    operator double() const { return _val; }
    
    const RealWeight operator+(const RealWeight& w) const;
    
    const RealWeight operator*(const RealWeight& w) const;
    
    RealWeight& operator+=(const RealWeight& w);
    
    RealWeight& operator*=(const RealWeight& w);
    
    friend RealWeight operator-(const RealWeight& v);
    
    friend ostream& operator<<(ostream& out, const RealWeight& w);
    
    
    static const double kZero;
    
    static const double kOne;
    
  private:
    inline RealWeight plus(const double d) const { return _val + d; }
    
    inline RealWeight times(const double d) const { return _val * d; }
    
    inline void plusEquals(const double d) { _val += d; }
    
    inline void timesEquals(const double d) { _val *= d; }
    
    double _val;
};

#endif
