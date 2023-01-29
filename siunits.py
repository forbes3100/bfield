#!/usr/bin/env python
# ============================================================================
#  siunits.py -- Physical Units Conversions
#
#  Copyright 2023 Scott Forbes
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

import re
import math as ma
import numpy as np

if 1:
    # Unicode-encoded special characters
    mu = u"\u03BC"
    Ohms = u"\u2126"
    Infin = u"\u221E"
else:
    mu = "u"
    Ohms = "Ohms"
    Infin = "inf"

SIPrefixes = ("f", "p", "n",  mu, "m", "", "k", "M", "G", "T")

# dictionary of the power-of-ten of each SI-unit prefix string
SIPowers = {}
i = -15
for pr in SIPrefixes:
    SIPowers[pr] = i
    i += 3

# useful units multipliers
ms = 1e-3
ns = 1e-9
us = 1e-6
mV = 1e-3
MHz = 1e6
GHz = 1e9
Mohm = 1e6

# flags for si():
SI_NO       = 1 # suppress SI units
SI_ASCII    = 2 # suppress Unicode
SI_SIGN     = 4 # always include +/-

def si(n, places=5, flags=0):
    """Represent a number in SI units.
    n:        given number.
    places:   significant digits.
    flags:    sum of any:
        SI_NO:    suppress SI units
        SI_ASCII: suppress Unicode
        SI_SIGN:  always include +/-

    Examples:
    >>> si(45)
    '45'
    >>> si(45, SI_NO)
    '4e+01'
    >>> si(10000)
    '10k'
    >>> si(.00002)
    u'20\u03bc'
    >>> si(.00002, flags=SI_ASCII)
    '20u'
    >>> si(1234.5, flags=SI_SIGN)
    '+1.2345k'
    >>> si(-1234.5, flags=SI_SIGN)
    '-1.2345k'
    >>> si(None)
    >>> si(0)
    '0'
    >>> si(12345678.90)
    '12.346M'
    >>> si(1.23456789e-9, 3)
    '1.23n'
    >>> si(1.23456789e-9, 8)
    '1.2345679n'
    >>> si(34e-12)
    '34p'
    >>> si(200e-15)
    '200f'
    >>> si(2200)
    '2.2k'
    >>> si(123456.7891234)
    '123.46k'
    >>> si(-12345678912.34)
    '-12.346G'
    >>> si(12345678.91234)
    '12.346M'
    >>> si(1e300)
    '1e+288T'
    >>> si(np.Inf)
    u'\u221e'
    >>> si(-np.Inf)
    u'-\u221e'
    >>> si(float('nan'))
    'NaN'
    >>>
    """
    if n is None:
        return None
    if ma.isnan(n):
        return "NaN"
    if n == 0 or (flags & SI_NO):
        return "%%0.%gg" % places % round(n, places)
    if abs(n) == np.Inf:
        return ("-", "")[n > 0] + Infin
    thou = min(max(int((ma.log10(abs(n)) + 15) / 3), 0), 9)
    p = SIPrefixes[thou]
    if (flags & SI_ASCII) and p == mu:
        p = "u"
    sign = "+" if (flags & SI_SIGN) else ""
    return "%%%s0.%gg%%s" % (sign, places) % (n * 10**(15-3*thou), p)

# Return the corresponding multiplier for an SI-unit prefix string.

def siScale(s):
    return 10**(SIPowers[s])

# Convert string s to floating point, with an empty string returning 0,
# String may be scaled by SI units.

numSIPat = r"([0-9.e\-+]+)([a-zA-Z]*)"

def floatSI(s):
    m = re.match(numSIPat, s)
    if m:
        try:
            sValue, units = m.groups()
            value = float(sValue)
            if len(units) > 0:
                p = units[0]
                if p in SIPrefixes:
                    value *= siScale(p)
                elif p == "u":
                    value *= siScale(mu)
                elif p == "K":
                    value *= siScale("k")
            return value
        except ValueError:
            ##print f"Bad float: '{s}'"
            pass
    return 0.
