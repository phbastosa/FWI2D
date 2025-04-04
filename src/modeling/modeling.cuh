# ifndef MODELING_CUH
# define MODELING_CUH

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

class Modeling
{
protected:

    bool ABC;

    float fmax, bd;
    float dx, dz, dt;

    int tlag, nThreads;
    int sBlocks, nBlocks;

    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int sIdx, sIdz;

    int * rIdx = nullptr;
    int * rIdz = nullptr;

    float * Vp = nullptr;

    float * seismogram = nullptr;
    float * seismic_data = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdz = nullptr;

    float * d_P = nullptr;
    float * d_Vp = nullptr;
    float * d_Pold = nullptr;

    float * d_b1d = nullptr;
    float * d_b2d = nullptr;
    
    float * d_wavelet = nullptr;
    float * d_seismogram = nullptr;

    std::string data_folder;

    void set_wavelet();
    void set_geometry();
    void set_properties();
    void set_seismogram();
    void set_cerjan_dampers();
    void set_main_parameters();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);

public:

    int srcId;

    Geometry * geometry;

    std::string parameters;

    void set_parameters();

    void initialization();

    void forward_solver();

    void show_information();    

    void export_output_data();
};

__global__ void compute_pressure(float * Vp, float * Pi, float * Pf, float * wavelet, float * d1D, float * d2D, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float dx, float dz, float dt, bool ABC);

__global__ void compute_seismogram(float * P, int * rIdx, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);

__device__ float get_boundary_damper(float * d1D, float * d2D, int i, int j, int nxx, int nzz, int nb);

# endif
