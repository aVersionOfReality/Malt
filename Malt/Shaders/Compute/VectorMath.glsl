// Vector (vec3) math nodes for compute graphs.
// All dependencies are self-contained (no Common.glsl / render-pipeline variables).
// Rotation nodes (Rotate Euler, Rotate Axis Angle) are omitted — they require
// mat4_rotation_from_euler / quaternion helpers that live in Common.glsl.

#ifndef COMPUTE_VECTOR_MATH_GLSL
#define COMPUTE_VECTOR_MATH_GLSL

// ── Shared helpers (same guard as FloatMath.glsl) ───────────────────────────
#ifndef COMPUTE_MATH_HELPERS_GLSL
#define COMPUTE_MATH_HELPERS_GLSL

float safe_mix(float a, float b, float f)
    { return f == 0.0 ? a : f == 1.0 ? b : mix(a, b, f); }
vec3  safe_mix(vec3 a, vec3 b, vec3 f)
    { return f == vec3(0) ? a : f == vec3(1) ? b : mix(a, b, f); }
vec3  safe_mix(vec3 a, vec3 b, float f)
    { return f == 0.0 ? a : f == 1.0 ? b : mix(a, b, f); }

#define _mrange(v,a,b,c,d)   (safe_mix((c),(d),((v)-(a))/((b)-(a))))
#define _mrange_c(v,a,b,c,d) clamp(_mrange(v,a,b,c,d),min((c),(d)),max((d),(c)))
#define _snap(v,r)            (round((v)/(r))*(r))
#define _distort(base,d,fac)  ((base)+((d)-0.5)*2.0*(fac))
#define _vangle(a,b)          acos(dot((a),(b))/(length(a)*length(b)))

#endif // COMPUTE_MATH_HELPERS_GLSL
// ────────────────────────────────────────────────────────────────────────────

/* META GLOBAL
    @meta: category=Math; subcategory=Vector 3D;
*/

/*META @meta: label=Add; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_add(vec3 a, vec3 b) { return a + b; }
/*META @meta: label=Subtract; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_subtract(vec3 a, vec3 b) { return a - b; }
/*META @meta: label=Multiply; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_multiply(vec3 a, vec3 b) { return a * b; }
/*META @meta: label=Divide; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_divide(vec3 a, vec3 b) { return a / b; }
/*META @meta: label=Scale; @a: subtype=Vector;*/
vec3 Vec3_scale(vec3 a, float fac) { return a * fac; }

/*META
    @meta: label=Map Range;
    @clamped: default=true;
    @a: label=Vector; default='vec3(0.5)'; subtype=Vector;
    @from_min: subtype=Vector; default=vec3(0.0);
    @from_max: subtype=Vector; default=vec3(1.0);
    @to_min: subtype=Vector; default=vec3(0.0);
    @to_max: subtype=Vector; default=vec3(1.0);
*/
vec3 Vec3_map_range(bool clamped, vec3 a,
                    vec3 from_min, vec3 from_max,
                    vec3 to_min,   vec3 to_max)
{
    return clamped
        ? _mrange_c(a, from_min, from_max, to_min, to_max)
        :  _mrange(a,  from_min, from_max, to_min, to_max);
}

/*META @meta: label=Modulo; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_modulo(vec3 a, vec3 b) { return mod(a, b); }
/*META @meta: label=Power; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_pow(vec3 a, vec3 b) { return pow(a, b); }
/*META @meta: label=Square Root; @a: subtype=Vector;*/
vec3 Vec3_sqrt(vec3 a) { return sqrt(a); }

/*META @meta: label=Distort; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_distort(vec3 a, vec3 b, float fac) { return _distort(a, b, fac); }

/*META @meta: label=Round; @a: subtype=Vector;*/
vec3 Vec3_round(vec3 a) { return round(a); }
/*META @meta: label=Fraction; @a: subtype=Vector;*/
vec3 Vec3_fract(vec3 a) { return fract(a); }
/*META @meta: label=Floor; @a: subtype=Vector;*/
vec3 Vec3_floor(vec3 a) { return floor(a); }
/*META @meta: label=Ceil; @a: subtype=Vector;*/
vec3 Vec3_ceil(vec3 a) { return ceil(a); }

/*META @meta: label=Snap; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_snap(vec3 a, vec3 b) { return _snap(a, b); }

/*META @meta: label=Clamp; @a: subtype=Vector; @b: label=Min; subtype=Vector; @c: label=Max; subtype=Vector;*/
vec3 Vec3_clamp(vec3 a, vec3 b, vec3 c) { return clamp(a, b, c); }
/*META @meta: label=Sign; @a: subtype=Vector;*/
vec3 Vec3_sign(vec3 a) { return sign(a); }
/*META @meta: label=Absolute; @a: subtype=Vector;*/
vec3 Vec3_abs(vec3 a) { return abs(a); }
/*META @meta: label=Min; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_min(vec3 a, vec3 b) { return min(a, b); }
/*META @meta: label=Max; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_max(vec3 a, vec3 b) { return max(a, b); }

/*META @meta: label=Mix 3D; @a: subtype=Vector; @b: subtype=Vector; @c: label=Factor; subtype=Vector;*/
vec3 Vec3_mix(vec3 a, vec3 b, vec3 c) { return safe_mix(a, b, c); }
/*META @meta: label=Mix; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_mix_float(vec3 a, vec3 b, float fac) { return safe_mix(a, b, fac); }

/*META @meta: label=Normalize; @a: subtype=Vector;*/
vec3 Vec3_normalize(vec3 a) { return a != vec3(0) ? normalize(a) : vec3(0); }

/*META @meta: label=Length; @a: subtype=Vector;*/
float Vec3_length(vec3 a) { return a != vec3(0) ? length(a) : 0.0; }
/*META @meta: label=Distance; @a: subtype=Vector; @b: subtype=Vector;*/
float Vec3_distance(vec3 a, vec3 b) { return a != b ? distance(a, b) : 0.0; }
/*META @meta: label=Dot Product; @a: subtype=Vector; @b: subtype=Vector;*/
float Vec3_dot_product(vec3 a, vec3 b) { return dot(a, b); }
/*META @meta: label=Cross Product; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_cross_product(vec3 a, vec3 b) { return cross(a, b); }
/*META @meta: label=Reflect; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_reflect(vec3 a, vec3 b) { return reflect(a, b); }
/*META @meta: label=Refract; @a: subtype=Vector; @b: subtype=Vector;*/
vec3 Vec3_refract(vec3 a, vec3 b, float ior) { return refract(a, normalize(b), ior); }
/*META @meta: label=Faceforward; @a: subtype=Vector; @b: subtype=Vector; @c: subtype=Vector;*/
vec3 Vec3_faceforward(vec3 a, vec3 b, vec3 c) { return faceforward(a, b, c); }

/*META @meta: label=Sine; @a: subtype=Vector;*/
vec3 Vec3_sin(vec3 a) { return sin(a); }
/*META @meta: label=Cosine; @a: subtype=Vector;*/
vec3 Vec3_cos(vec3 a) { return cos(a); }
/*META @meta: label=Tangent; @a: subtype=Vector;*/
vec3 Vec3_tan(vec3 a) { return tan(a); }

/*META @meta: label=Angle; @a: subtype=Vector; @b: subtype=Vector;*/
float Vec3_angle(vec3 a, vec3 b) { return _vangle(a, b); }

/*META @meta: label=Equal; @a: subtype=Vector; @b: subtype=Vector;*/
bool Vec3_equal(vec3 a, vec3 b) { return a == b; }
/*META @meta: label=Not Equal; @a: subtype=Vector; @b: subtype=Vector;*/
bool Vec3_not_equal(vec3 a, vec3 b) { return a != b; }

/*META @meta: label=If Else; @a: label=If True; subtype=Vector; @b: label=If False; subtype=Vector;*/
vec3 Vec3_if_else(bool condition, vec3 a, vec3 b) { return condition ? a : b; }

/*META @meta: label=Combine;*/
vec3 Vec3_combine(float x, float y, float z) { return vec3(x, y, z); }
/*META @meta: label=Separate; @a: subtype=Vector;*/
void Vec3_separate(vec3 a, out float x, out float y, out float z)
    { x = a.x; y = a.y; z = a.z; }

#endif // COMPUTE_VECTOR_MATH_GLSL
