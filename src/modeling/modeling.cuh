# ifndef MODELING_HPP
# define MODELING_HPP

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

class Modeling
{
private:

    float * Pi = nullptr;
    float * Pf = nullptr;

    float * dtVp2 = nullptr;

    void set_wavelet();
    void set_boundaries();
    void set_properties();
    void set_seismogram();
    void set_specifications();

protected:

    float fmax, bd;

    int tlag, nThreads;
    int sBlocks, nBlocks;

    float * d1D = nullptr;
    float * d2D = nullptr;

    int * rIdx = nullptr;
    int * rIdz = nullptr;

    int * current_xrec = nullptr;
    int * current_zrec = nullptr;

    float * wavelet = nullptr;
    float * seismogram = nullptr;
    float * synthetic_data = nullptr;

public:

    float dx, dz, dt;
    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int srcId, recId, sIdx, sIdz;

    float * Vp = nullptr;

    Geometry * geometry;

    int nTraces;

    float * output_data = nullptr;

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

__global__ void compute_pressure(float * dtvp2, float * Pi, float * Pf, float * wavelet, float * d1D, float * d2D, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float dx, float dz);

__global__ void compute_seismogram(float * P, int * rIdx, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);

__device__ float get_boundary_damper(float * d1D, float * d2D, int i, int j, int nxx, int nzz, int nb);

# endif
