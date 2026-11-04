// The push constants of the DeepSeek-V4 hyper-connection ops. The offsets and strides are in floats, one set for
// each of up to four sources and the destination, which is the last.
layout (push_constant) uniform parameter
{
    uint ne0;
    uint hc;
    uint flag;
    float param;
    uint offs0; uint offs1; uint offs2; uint offs3; uint offs4;
    uint nb00; uint nb01; uint nb02;
    uint nb10; uint nb11; uint nb12;
    uint nb20; uint nb21; uint nb22;
    uint nb30; uint nb31; uint nb32;
    uint nb40; uint nb41; uint nb42;
} p;
