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

class LogWeight {

  public:
    explicit LogWeight();

    explicit LogWeight(double value, bool valueIsLog = false);

    operator double() const { return _val; }
    
    const LogWeight operator+(const LogWeight& w) const;
    
    const LogWeight operator*(const LogWeight& w) const;
    
    const LogWeight operator/(const LogWeight& w) const;
    
    LogWeight& operator+=(const LogWeight& w);
    
    LogWeight& operator*=(const LogWeight& w);
    
    LogWeight& operator/=(const LogWeight& w);
    
    friend std::ostream& operator<<(std::ostream& out, const LogWeight& w);
    
  private:
    double _val;
};

LogWeight sqrt(LogWeight w);

LogWeight abs(LogWeight w);

#endif
