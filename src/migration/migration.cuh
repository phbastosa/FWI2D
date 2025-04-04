# ifndef MIGRATION_CUH
# define MIGRATION_CUH

# include <curand_kernel.h>

# include "../modeling/modeling.cuh"

class Migration : public Modeling
{
protected:

    float vmax, vmin;
    float rbc_ratio;
    float rbc_varVp;

    float * d_X = nullptr;
    float * d_Z = nullptr;

    float * d_Pr = nullptr;
    float * d_Prold = nullptr;
    float * d_image = nullptr;
    float * d_sumPs = nullptr;
    float * d_sumPr = nullptr;

    float * image = nullptr;
    float * sumPs = nullptr;
    float * sumPr = nullptr;
    float * partial = nullptr;

    std::string stage;
    std::string input_file;
    std::string output_folder;
    
    void set_coordinates();
    void show_information();
    void set_seismic_source();
    void set_random_boundary();

public:

    void set_parameters();
    void forward_propagation();
    void backward_propagation();
    void image_enhancing();
    void export_seismic();
};

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, float * sumPr, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt);

__device__ float get_random_value(float velocity, float function, float parameter, int index, float varVp);
__global__ void random_boundary_bg(float * Vp, int nxx, int nzz, int nb, float varVp);
__global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, float xb, float zb, float A, float xc, float zc, float r, float vmax, float vmin, float varVp);

# endif