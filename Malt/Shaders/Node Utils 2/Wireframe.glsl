#ifndef WIREFRAME_GLSL
#define WIREFRAME_GLSL

#ifdef PIXEL_SHADER

/* META
    @meta: category=Input; subcategory=Wireframe;
    @Width: default=1.0; min=0.0; max=10.0;
    @result: label=Factor;
*/
float Wireframe(float Width)
{
#ifdef _HAS_BARY_EXT
    // Per-rasterized-triangle barycentric coordinates from hardware extension.
    // Shows tessellated sub-triangle edges and original geometry edges.
    vec3 bary = _BARY_COORDS;
#else
    // Fallback: per-vertex barycentrics passed via varying.
    // Only shows original triangle edges, not tessellated sub-triangles.
    vec3 bary = IO_BARYCENTRIC;
#endif
    float d = min(bary.x, min(bary.y, bary.z));
    float fw = fwidth(d);
    return 1.0 - smoothstep(0.0, fw * Width, d);
}

#endif //PIXEL_SHADER

#endif //WIREFRAME_GLSL
