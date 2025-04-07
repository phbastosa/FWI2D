# include "inversion.cuh"

void Inversion::set_parameters()
{
    set_main_parameters();

    set_wavelet();
    set_geometry();
    set_coordinates();
    set_cerjan_dampers();

    input_folder = catch_parameter("inversion_input_folder", parameters);
    output_folder = catch_parameter("inversion_output_folder", parameters);

    max_iteration = std::stoi(catch_parameter("max_iteration", parameters));

    auto frequencies = split(catch_parameter("multi_frequency", parameters),',');

    nFreqs = frequencies.size();

    freqs = new float[nFreqs]();
    for (int fId = 0; fId < nFreqs; fId++) 
        freqs[fId] = std::stof(frequencies[fId]);

    iteration = 0;

    sumPs = new float[nPoints]();
    partial = new float[matsize]();
    gradient = new float[nPoints]();

    cudaMalloc((void**)&(d_Pr), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), matsize*sizeof(float));
    cudaMalloc((void**)&(d_gradient), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vp_clean), matsize*sizeof(float));
}

void Inversion::set_observed_data()
{
    obs_data = new float[nt_out*geometry->nTraces]();

    std::string input_file = input_folder + "seismogram_nt" + std::to_string(nt_out) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string(int(freqs[freqId])) + "Hz_" + std::to_string(int(1e3f*dt_out)) + "ms.bin";

    import_binary_float(input_file, obs_data, nt_out*geometry->nTraces); 
}

void Inversion::set_model_dimension()
{    
    float N = 4.0f;
    float S = 0.5f;

    std::string initial_model_file;
    std::string current_model_file;
    
    if (freqId == 0)
    {
        nb_out = nb;
    
        nx_out = nx; nz_out = nz;
        dx_out = dx; dz_out = dz;
        nt_out = nt; dt_out = dt;
        
        nPoints_out = nx_out*nz_out;
    
        initial_model_file = catch_parameter("model_file", parameters);
        model_file = initial_model_file; 
    }
    else 
    {   
        current_model_file = output_folder + "final_model_" + std::to_string(int(freqs[freqId-1])) + "Hz_" + std::to_string(nz_out) + "x" + std::to_string(nx_out) + ".bin";
        model_file = current_model_file;
    }

    float * vp = new float[nPoints_out];

    import_binary_float(model_file, vp, nPoints_out);
    
    vmax = 0.0f;
    vmin = 1e9f;

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        vmax = vmax < vp[index] ? vp[index] : vmax; 
        vmin = vmin > vp[index] ? vp[index] : vmin; 
    }

    dx = vmin / (N*freqs[freqId]);
    dz = vmin / (N*freqs[freqId]);
    
    dt = S / vmax * (1.0f / sqrtf(1.0f / (dx*dx) + 1.0f / (dz*dz)));
    dt = 1e-3f*std::floor(1e3f*dt*2.0f)*0.5f; 

    float x_max = (nx_out-1)*dx_out;
    float z_max = (nz_out-1)*dz_out;

    nx = (int)(x_max / dx) + 1;
    nz = (int)(z_max / dz) + 1;

    // cubic interpolation    



        


    // cudaMemcpy(d_Vp_clean, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemset(d_gradient, 0.0f, matsize*sizeof(float));
}

void Inversion::set_calculated_data()
{
    stage_info = "Calculating seismograms to compute residuo...";

    for (srcId = 0; srcId < geometry->nrel; srcId++)
    {
        show_information();
        
        initialization();
        forward_solver();
        set_seismogram();
    }
}

void Inversion::show_information()
{
    auto clear = system("clear");
 
    std::string title = "\033[34mFull Waveform Inversion\033[0;0m";

    int width = 80;
    int padding = (width - title.length() + 8) / 2;

    std::string line(width, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << 
                                  ", x = " << (nx - 1)*dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << "Current frequency: " + std::to_string(int(freqs[freqId])) + " Hz\n\n"; 

    std::cout << line << "\n";
    std::cout << stage_info << "\n";
    std::cout << line << "\n\n";

    if (iteration == max_iteration)
    {
        std::cout << "Checking final residuals\n\n";
    }
    else
    {    
        std::cout << "Computing iteration " << iteration + 1 << " of " << max_iteration << "\n\n";

        if (iteration > 0) std::cout << "Previous residuals: " << residuo.back() << "\n";   
    }
}

void Inversion::check_convergence()
{
    float square_difference = 0.0f;

    for (int index = 0; index < nt*geometry->nTraces; index++)
    {
        seismic_data[index] = obs_data[index] - seismic_data[index];

        square_difference += seismic_data[index]*seismic_data[index];
    }

    residuo.push_back(sqrtf(square_difference));

    if ((iteration >= max_iteration))
    {
        std::cout << "Final residuals: "<< residuo.back() <<"\n";
        converged = true;
    }
    else
    {
        iteration += 1;
        converged = false;
    }
}

void Inversion::compute_gradient()
{
    ABC = false;
    
    for (srcId = 0; srcId < geometry->nrel; srcId++)
    {
        forward_propagation();
        backward_propagation();
    }
    


    cudaMemcpy(d_Vp, d_Vp_clean, matsize*sizeof(float), cudaMemcpyDeviceToDevice);

    ABC = true;
}

void Inversion::forward_propagation()
{
    stage_info = "Calculating gradient of the objective function ---> Forward propagation.";

    show_information();

    set_random_boundary();
    
    initialization();
    forward_solver();
}

void Inversion::backward_propagation()
{
    stage_info = "Calculating gradient of the objective function ---> Backward propagation.";

    show_information();

    initialization();
    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, matsize*sizeof(float));

    for (int tId = 0; tId < nt; tId++)
    {
        FWI<<<nBlocks, nThreads>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_seismogram, d_gradient, d_sumPs, d_rIdx, d_rIdz, geometry->spread, tId, nxx, nzz, nt, dx, dz, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }    
}

void Inversion::set_seismic_source()
{
    for (int timeId = 0; timeId < nt; timeId++)
        for (int spreadId = 0; spreadId < geometry->spread; spreadId++)
            seismogram[timeId + spreadId*nt] = seismic_data[timeId + spreadId*nt + srcId*geometry->spread*nt];     

    cudaMemcpy(d_seismogram, seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::optimization()
{
    stage_info = "Optimizing problem with title method ...";

    cudaMemcpy(partial, d_gradient, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, gradient);

    cudaMemcpy(partial, d_sumPs, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        gradient[index] /= sumPs[index];

    export_binary_float("gradient.bin", gradient, nPoints);




}

void Inversion::update_model()
{
    stage_info = "Updating model ...";

    // model_file = output_folder + "model_iteration_" + std::to_string(iteration) + "_" + std::to_string(int(freqs[freqId])) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + ".bin";  

    // update Vp and Vp_clean (cuda copy)

}

void Inversion::export_convergence()
{



}

void Inversion::export_final_model()
{
    model_file = output_folder + "final_model_" + std::to_string(int(freqs[freqId])) + "Hz_" + std::to_string(nz_out) + "x" + std::to_string(nx_out) + ".bin";
    
    float * vp = new float[nPoints_out];

    export_binary_float(model_file, vp, nPoints_out);
}

__global__ void FWI(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * gradient, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
    {
        for (int rId = 0; rId < spread; rId++)
        {
            Pr[(rIdz[rId] + 1) + rIdx[rId]*nzz] += 0.5f*seismogram[(nt-tId-1) + rId*nt] / (dx*dz); 
            Pr[(rIdz[rId] - 1) + rIdx[rId]*nzz] += 0.5f*seismogram[(nt-tId-1) + rId*nt] / (dx*dz); 
        }
    }    

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2Ps_dx2 = (- 9.0f*(Psold[i + (j-4)*nzz] + Psold[i + (j+4)*nzz])
                      +   128.0f*(Psold[i + (j-3)*nzz] + Psold[i + (j+3)*nzz])
                      -  1008.0f*(Psold[i + (j-2)*nzz] + Psold[i + (j+2)*nzz])
                      +  8064.0f*(Psold[i + (j+1)*nzz] + Psold[i + (j-1)*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dx*dx);

        float d2Ps_dz2 = (- 9.0f*(Psold[(i-4) + j*nzz] + Psold[(i+4) + j*nzz])
                      +   128.0f*(Psold[(i-3) + j*nzz] + Psold[(i+3) + j*nzz])
                      -  1008.0f*(Psold[(i-2) + j*nzz] + Psold[(i+2) + j*nzz])
                      +  8064.0f*(Psold[(i-1) + j*nzz] + Psold[(i+1) + j*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dz*dz);

        float d2Pr_dx2 = (- 9.0f*(Pr[i + (j-4)*nzz] + Pr[i + (j+4)*nzz])
                      +   128.0f*(Pr[i + (j-3)*nzz] + Pr[i + (j+3)*nzz])
                      -  1008.0f*(Pr[i + (j-2)*nzz] + Pr[i + (j+2)*nzz])
                      +  8064.0f*(Pr[i + (j+1)*nzz] + Pr[i + (j-1)*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dx*dx);

        float d2Pr_dz2 = (- 9.0f*(Pr[(i-4) + j*nzz] + Pr[(i+4) + j*nzz])
                      +   128.0f*(Pr[(i-3) + j*nzz] + Pr[(i+3) + j*nzz])
                      -  1008.0f*(Pr[(i-2) + j*nzz] + Pr[(i+2) + j*nzz])
                      +  8064.0f*(Pr[(i-1) + j*nzz] + Pr[(i+1) + j*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dz*dz);

        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];
        
        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];

        gradient[index] += dt*Pr[index]*(d2Ps_dx2 + d2Ps_dz2)*Vp[index]*Vp[index];   

        sumPs[index] += Ps[index]*Ps[index];
    }
}
