# include "inversion/inversion.cuh"

int main(int argc, char **argv)
{
    auto inversion = new Inversion();

    inversion->parameters = std::string(argv[1]);

    inversion->set_parameters();

    auto ti = std::chrono::system_clock::now();

    for (int fId = 0; fId < inversion->nFreqs; fId++)
    {
        inversion->set_observed_data();
        inversion->set_model_dimension();

        while (true)
        {
            inversion->forward_modeling();

            inversion->check_convergence();

            if (inversion->converged) break;

            inversion->compute_gradient();
           
            inversion->optimization();

            inversion->update_model();
        }
    
        // modeling->export_output_data();
    }

    auto tf = std::chrono::system_clock::now();


    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}