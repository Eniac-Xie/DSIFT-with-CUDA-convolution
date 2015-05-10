#ifndef FASTMATH_H
#define FASTMATH_H
#include <cmath>

#define DSIFT_EPSILON_F 1.19209290E-07F
#define DSIFT_PI 3.141592653589793

inline float fast_atan2_f(float y, float x)
{
	float angle, r;
	float const c3 = 0.1821F;
	float const c1 = 0.9675F;
	float abs_y = fabs(y) + DSIFT_EPSILON_F;

	if (x >= 0) {
		r = (x - abs_y) / (x + abs_y);
		angle = (float)(DSIFT_PI / 4);
	}
	else {
		r = (x + abs_y) / (abs_y - x);
		angle = (float)(3 * DSIFT_PI / 4);
	}
	angle += (c3*r*r - c1) * r;
	return (y < 0) ? -angle : angle;
}

inline float fast_resqrt_f(float x)
{
	/* 32-bit version */
	union {
		float x;
		int  i;
	} u;

	float xhalf = (float) 0.5 * x;

	/* convert floating point value in RAW integer */
	u.x = x;

	/* gives initial guess y0 */
	u.i = 0x5f3759df - (u.i >> 1);
	/*u.i = 0xdf59375f - (u.i>>1);*/

	/* two Newton steps */
	u.x = u.x * ((float) 1.5 - xhalf*u.x*u.x);
	u.x = u.x * ((float) 1.5 - xhalf*u.x*u.x);
	return u.x;
}

inline float fast_sqrt_f(float x)
{
	return (x < 1e-8) ? 0 : x * fast_resqrt_f(x);
}

inline float mod_2pi_f(float x)
{
	while (x >(float)(2 * DSIFT_PI)) x -= (float)(2 * DSIFT_PI);
	while (x < 0.0F) x += (float)(2 * DSIFT_PI);
	return x;
}

inline long int floor_f(float x)
{
	long int xi = (long int)x;
	if (x >= 0 || (float)xi == x) return xi;
	else return xi - 1;
}
#endif