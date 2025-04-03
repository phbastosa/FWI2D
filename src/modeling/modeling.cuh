# ifndef MODELING_HPP
# define MODELING_HPP

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

class Modeling
{
private:

    float * d_b1d = nullptr;
    float * d_b2d = nullptr;

    float * d_wavelet = nullptr;
    float * d_seismogram = nullptr;

    void set_wavelet();
    void set_boundaries();
    void set_properties();
    void set_seismogram();
    void set_specifications();

public:

    float fmax, bd;
    float dx, dz, dt;

    int nTraces;
    int tlag, nThreads;
    int sBlocks, nBlocks;

    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int srcId, recId, sIdx, sIdz;

    float * Vp = nullptr;

    float * d_Pi = nullptr;
    float * d_Pf = nullptr;
    float * d_Vp = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdz = nullptr;

    int * current_xrec = nullptr;
    int * current_zrec = nullptr;

    float * seismogram = nullptr;
    float * output_data = nullptr;

    Geometry * geometry;

    std::string parameters;
    std::string data_folder;
    std::string modeling_name;

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);

    void set_parameters();
    void set_conditions();

    void initialization();
    void forward_solver();

    void show_information();    

    void export_output_data();
};

__global__ void compute_pressure(float * Vp, float * Pi, float * Pf, float * wavelet, float * d1D, float * d2D, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float dx, float dz, float dt);

__global__ void compute_seismogram(float * P, int * rIdx, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);

__device__ float get_boundary_damper(float * d1D, float * d2D, int i, int j, int nxx, int nzz, int nb);

# endif
