# ifndef INVERSION_CUH
# define INVERSION_CUH

# include "../modeling/modeling.cuh"

class Inversion : public Modeling
{
private:

    int iteration;
    int max_iteration;

    int nx_out;
    int nz_out;
    int nt_out;
    int nb_out;

    float dx_out;
    float dz_out;
    float dt_out;

    int nPoints_out;

    float * freqs = nullptr;
    float * obs_data = nullptr;

    float * d_Pr = nullptr;
    float * d_Prold = nullptr;
    float * d_sumPs = nullptr;
    float * d_gradient = nullptr;
    float * d_Vp_clean = nullptr;

    float * sumPs = nullptr;
    float * partial = nullptr;    
    float * gradient = nullptr;

    std::string stage_info;
    
    std::string model_file;
    std::string input_folder;
    std::string output_folder; 
    
    std::vector<float> residuo;

    void set_seismic_source();
    void forward_propagation();
    void backward_propagation();

public:

    int freqId;
    int nFreqs;

    bool converged;

    void set_parameters();
    void set_observed_data();
    void set_model_dimension();
    void show_information();
    void check_convergence();
    void set_calculated_data();
    void compute_gradient();
    void optimization();
    void update_model();

    void export_final_model();
    void export_convergence();
};

__global__ void FWI(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * gradient, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt);

# endif