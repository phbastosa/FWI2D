# ifndef INVERSION_CUH
# define INVERSION_CUH

# include "../modeling/modeling.cuh"

class Inversion : public Modeling
{
private:

    std::string stage_info;
    
    std::string input_folder;
    std::string output_folder; 

    int iteration;
    int max_iteration;

    float * freqs = nullptr;
    float * obs_data = nullptr;





public:

    int nFreqs;

    bool converged;

    void set_parameters();
    void set_observed_data();
    void set_model_dimension();
    void show_information();
    void check_convergence();
    void forward_modeling();
    void compute_gradient();
    void optimization();
    void update_model();
};

# endif