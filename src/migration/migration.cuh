# ifndef MIGRATION_CUH
# define MIGRATION_CUH

# include "../modeling/modeling.cuh"

class Migration : public Modeling
{
protected:

    float * d_Pr = nullptr;
    float * d_Prold = nullptr;
    float * d_image = nullptr;
    float * d_sumPs = nullptr;
    float * d_sumPr = nullptr;

    float * image = nullptr;
    float * sumPs = nullptr;
    float * sumPr = nullptr;
    float * partial = nullptr;

    std::string stage_info;
    std::string input_folder;
    std::string output_folder;
    
    void show_information();
    void set_seismic_source();

public:

    void set_parameters();
    void forward_propagation();
    void backward_propagation();
    void image_enhancing();
    void export_seismic();
};

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, float * sumPr, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt);

# endif