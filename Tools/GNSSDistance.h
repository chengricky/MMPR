#ifndef _GNSSDISTANCE_H
#define _GNSSDISTANCE_H


long double toRadians(const long double degree);
long double GNSSdistance(long double lat1, long double long1,
	long double lat2, long double long2);
double ddmm2dd(double ddmm);

#endif