# ifndef INVERSION_CUH
# define INVERSION_CUH

# include "../modeling/modeling.cuh"

class Inversion : public Modeling
{
private:

    float step;
    float zmask;
    float sum_res;

    int iteration;
    int max_iteration;

    float rbc_ratio;
    float rbc_varVp;
    float rbc_length; 

    int abc_nxx, abc_nzz, abc_nb, abc_matsize;
    int rbc_nxx, rbc_nzz, rbc_nb, rbc_matsize;

    float * model = nullptr;

    float * partial1 = nullptr;    
    float * partial2 = nullptr;    
    float * gradient = nullptr;
    float * obs_data = nullptr;

    float * d_Vp_rbc = nullptr;

    float * d_Ps_rbc = nullptr;
    float * d_Pr_rbc = nullptr;

    float * d_Ps_old_rbc = nullptr;
    float * d_Pr_old_rbc = nullptr;

    float * d_sumPs_rbc = nullptr;
    float * d_gradient_rbc = nullptr;

    std::string stage_info;
    std::string input_folder;
    std::string input_prefix;
    std::string output_folder; 
    std::string residuo_folder; 
    
    std::vector<float> residuo;
    
    void show_information();

    void set_ABC_dimension();
    void get_ABC_dimension();
    
    void set_RBC_dimension();
    void get_RBC_dimension();

    void set_initial_model();

    void update_RBC();
    void set_obs_data();
    void get_cal_data();

    void set_adjoint_source();
    void forward_propagation();
    void backward_propagation();

    void linesearch(float alpha);

public:

    bool converged;

    void set_parameters();

    void compute_gradient();
    void check_convergence();

    void optimization();
    void update_model();

    void export_final_model();
    void export_convergence();
};

__global__ void inject_adjoint(float * Pr, int * rIdx, int * rIdz, float * seismogram, int nr, int tId, int nt, int nzz, float idh2);
__global__ void build_gradient(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * gradient, float * sumPs, int nxx, int nzz, int nt, float dt, float idh2);

# endif