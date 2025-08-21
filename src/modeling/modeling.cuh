# ifndef MODELING_CUH
# define MODELING_CUH

# include <cuda_runtime.h>
# include <curand_kernel.h>

# include "../geometry/geometry.hpp"

# define NW 8

# define WIDTH 80
# define NTHREADS 256

class Modeling
{
protected:

    bool ABC;

    float abc_length;
    float abc_factor;
    float vmax, vmin;
    float dh, dt, fmax;

    std::string title;
    
    int padding;

    int sBlocks, nBlocks;

    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int sIdx, sIdz, tlag;

    float * Vp = nullptr;

    float * seismogram = nullptr;

    float * d_X = nullptr;
    float * d_Z = nullptr;

    float * d_sw = nullptr;
    float * d_rw = nullptr;

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
    void set_coordinates();
    void set_seismograms();    
    void set_abc_dampers();
    void set_main_parameters();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);

    void set_random_boundary(float * vp, float ratio, float varVp);

public:

    int srcId;

    Geometry * geometry;

    std::string parameters;

    void set_parameters();
    void initialization();
    void forward_solver();
    void get_seismogram();
    void show_information();    
    void export_output_data();
};

__global__ void compute_pressure(float * Vp, float * P, float * Pold, float * d_wavelet, float * d_sw, float * d_b1d, float * d_b2d, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float dh, float dt, bool ABC);
__global__ void compute_seismogram(float * P, int * d_rIdx, int * d_rIdz, float * d_rw, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);
__device__ float get_boundary_damper(float * d1D, float * d2D, int i, int j, int nxx, int nzz, int nb);

__device__ float get_random_value(float velocity, float function, float parameter, int index, float varVp);
__global__ void random_boundary_bg(float * Vp, int nxx, int nzz, int nb, float varVp);
__global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, float xb, float zb, float factor, float A, float xc, float zc, float r, float vmax, float vmin, float varVp);

# endif
