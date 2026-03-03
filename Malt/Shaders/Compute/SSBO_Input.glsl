// Compute-graph node: read a face-corner attribute SSBO by index.
//
// Index 0–7 correspond to mesh attributes named malt_ssbo_0 through malt_ssbo_7.
// Any attribute domain (Point, Face, or Face Corner) is expanded to face-corner
// on the CPU side before upload, so all slots are indexed identically by loop_index.
//
// Outputs the stored vec4 color for the current loop, or vec4(0) if the slot is
// not active (no attribute assigned).
//
// Note: loop_index is NOT a parameter. The node uses gl_GlobalInvocationID.x
// internally, so it can be placed anywhere in the graph without needing to be
// in the execution chain.

#ifndef SSBO_INPUT_GLSL
#define SSBO_INPUT_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=SSBO Input;
    @Index: min=0; max=7; default=0;
*/
void SSBO_Input(int Index, out vec4 Color) {}

#else // COMPUTE_STAGE

void SSBO_Input(int Index, out vec4 Color)
{
    uint li = gl_GlobalInvocationID.x;
    Color = vec4(0.0);
    if      (Index == 0 && SSBO_ACTIVE[0])      Color = ssbo_data_0[li];
    else if (Index == 1 && SSBO_ACTIVE[1])      Color = ssbo_data_1[li];
    else if (Index == 2 && SSBO_ACTIVE[2])      Color = ssbo_data_2[li];
    else if (Index == 3 && SSBO_ACTIVE[3])      Color = ssbo_data_3[li];
#ifdef NEEDS_SSBO_DATA_HIGH
    else if (Index == 4 && SSBO_ACTIVE_HIGH[0]) Color = ssbo_data_4[li];
    else if (Index == 5 && SSBO_ACTIVE_HIGH[1]) Color = ssbo_data_5[li];
    else if (Index == 6 && SSBO_ACTIVE_HIGH[2]) Color = ssbo_data_6[li];
    else if (Index == 7 && SSBO_ACTIVE_HIGH[3]) Color = ssbo_data_7[li];
#endif
}

#endif // COMPUTE_STAGE

#endif // SSBO_INPUT_GLSL
