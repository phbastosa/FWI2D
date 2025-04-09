# include "inversion.cuh"

void Inversion::set_parameters()
{
    set_main_parameters();
    
    set_geometry();

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

    vp = new float[nPoints]();
}

void Inversion::set_model_dimension()
{    
    std::string initial_model_file;
    std::string current_model_file;
    
    if (freqId == 0)
    {
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

    set_model_interpolation();

    set_wavelet();
    set_coordinates();

    set_seismograms();

    sumPs = new float [nPoints]();
    partial = new float[matsize]();
    gradient = new float[nPoints]();

    cudaMalloc((void**)&(d_P), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vp), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pr), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Vp_clean), matsize*sizeof(float));
    cudaMalloc((void**)&(d_gradient), matsize*sizeof(float));

    cudaMemcpy(d_Vp, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vp_clean, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_gradient, 0.0f, matsize*sizeof(float));
}

void Inversion::set_model_interpolation()
{
    float N = 5.0f;
    float S = 0.5f;

    float p[CUBIC][CUBIC];

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
    dt = 1e-3f*floorf(1e3f*dt*2.0f)*0.5f; 

    float x_max = (float)(nx_out-1)*dx_out;
    float z_max = (float)(nz_out-1)*dz_out;
    float t_max = (float)(nt_out-1)*dt_out;

    nx = (int)(ceilf(x_max / dx)) + 1;
    nz = (int)(ceilf(z_max / dz)) + 1;
    nt = (int)(ceilf(t_max / dt)) + 1;

    nPoints = nx*nz;

    nxx = nx + 2*nb;
    nzz = nz + 2*nb;

    matsize = nxx*nzz;

    aux_vp = new float[nPoints]();

    int skipx = (int)((nx_out - 1)/(nx - 1));
    int skipz = (int)((nz_out - 1)/(nz - 1));

    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);
        int j = (int)(index / nz);   

        if ((i >= skipz) && (i < nz-skipz) && (j >= skipx) && (j < nx-skipx))
        {
            float z = i*dz;
            float x = j*dx;     
        
            float x0 = floorf(x/dx_out)*dx_out;
            float z0 = floorf(z/dz_out)*dz_out;

            float x1 = floorf(x/dx_out)*dx_out + dx_out;
            float z1 = floorf(z/dz_out)*dz_out + dz_out;

            float pdx = (x - x0) / (x1 - x0);  
            float pdz = (z - z0) / (z1 - z0); 

            for (int pIdz = 0; pIdz < CUBIC; pIdz++)
            {
                for (int pIdx = 0; pIdx < CUBIC; pIdx++)
                {
                    int pz = (int)(floorf(z0/dz_out)) + pIdz - 1;
                    int px = (int)(floorf(x0/dx_out)) + pIdx - 1;    
                    
                    if (pz < 0) pz = 0;
                    if (px < 0) px = 0;
                    
                    if (pz > nz_out) pz = nz_out;
                    if (px > nx_out) px = nx_out;
                    
                    p[pIdx][pIdz] = vp[pz + px*nz_out];
                }    
            }

            aux_vp[i + j*nz] = cubic2d(p, pdx, pdz); 
        }
    }

    for (int i = 0; i < skipz; i++)
    {
        for (int j = skipx; j < nx-skipx; j++)
        {
            aux_vp[i + j*nz] = aux_vp[skipz + j*nz];
            aux_vp[nz-i-1 + j*nz] = aux_vp[nz-skipz-1 + j*nz];
        }
    }

    for (int i = 0; i < nz; i++)
    {
        for (int j = 0; j < skipx; j++)
        {
            aux_vp[i + j*nz] = aux_vp[i + skipx*nz];
            aux_vp[i + (nx-j-1)*nz] = aux_vp[i + (nx-skipx-1)*nz];
        }
    }

    Vp = new float[matsize]();

    expand_boundary(aux_vp, Vp);    
}

void Inversion::set_observed_data()
{
    obs_data = new float[nt_out*geometry->nTraces]();
    cal_data = new float[nt_out*geometry->nTraces]();

    std::string input_file = input_folder + "seismogram_nt" + std::to_string(nt_out) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string(int(freqs[freqId])) + "Hz_" + std::to_string(int(1e3f*dt_out)) + "ms.bin";

    import_binary_float(input_file, obs_data, nt_out*geometry->nTraces); 

    float amp_max = 0.0f;

    # pragma omp parallel for
    for (int index = 0; index < nt_out*geometry->nTraces; index++) 
    {
        if (amp_max < fabsf(obs_data[index])) 
            amp_max = fabsf(obs_data[index]); 
    }

    # pragma omp parallel for
    for (int index = 0; index < nt_out*geometry->nTraces; index++) 
        obs_data[index] *= 1.0f / amp_max;
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

    set_data_interpolation();

    float amp_max = 0.0f;

    # pragma omp parallel for
    for (int index = 0; index < nt_out*geometry->nTraces; index++) 
    {
        if (amp_max < fabsf(cal_data[index])) 
            amp_max = fabsf(cal_data[index]); 
    }

    # pragma omp parallel for
    for (int index = 0; index < nt_out*geometry->nTraces; index++) 
        cal_data[index] *= 1.0f / amp_max;
}

void Inversion::set_data_interpolation()
{
    std::vector<double> trace_in(nt);      
    std::vector<double> trace_out(nt_out); 

    fftw_plan forward_plan;
    fftw_plan inverse_plan;

    fftw_complex * T_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nt);
    fftw_complex * T_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nt_out);

    int ncopy = std::min(nt, nt_out);
    int kcopy = (int)(ncopy / 2);
    
    for (int index = 0; index < geometry->nTraces; index++)
    {   
        # pragma omp parallel for
        for (int tId = 0; tId < nt; tId++)
            trace_in[tId] = (double)(seismic_data[tId + index*nt]);

        forward_plan = fftw_plan_dft_r2c_1d(nt, trace_in.data(), T_in, FFTW_ESTIMATE);
        fftw_execute(forward_plan);

        # pragma omp parallel for
        for (int tId = 0; tId < nt_out; tId++) 
        {
            T_out[tId][0] = 0.0;
            T_out[tId][1] = 0.0;
        }

        # pragma omp parallel for
        for (int tId = 0; tId <= kcopy; tId++) 
        {
            T_out[tId][0] = T_in[tId][0];
            T_out[tId][1] = T_in[tId][1];
        }
        
        # pragma omp parallel for
        for (int tId = 1; tId < kcopy; tId++) 
        {
            T_out[nt_out - tId][0] = T_in[nt - tId][0];
            T_out[nt_out - tId][1] = T_in[nt - tId][1];
        }

        inverse_plan = fftw_plan_dft_c2r_1d(nt_out, T_out, trace_out.data(), FFTW_ESTIMATE);
        fftw_execute(inverse_plan);
        
        # pragma omp parallel for
        for (int tId = 0; tId < nt_out; tId++) 
            cal_data[tId + index*nt_out] = (float)(trace_out[tId]);
    }

    fftw_free(T_in);
    fftw_free(T_out);
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
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

    export_binary_float("obs_data_" + std::to_string(iteration) + ".bin", obs_data, nt_out*geometry->nTraces);
    export_binary_float("cal_data_" + std::to_string(iteration) + ".bin", cal_data, nt_out*geometry->nTraces);

    for (int index = 0; index < nt_out*geometry->nTraces; index++)
    {
        cal_data[index] = obs_data[index] - cal_data[index];
        square_difference += cal_data[index]*cal_data[index];
    }

    export_binary_float("res_data_" + std::to_string(iteration) + ".bin", cal_data, nt_out*geometry->nTraces);

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

    export_binary_float("gradient_" + std::to_string(nz) + ".bin" , gradient, nPoints);




}

void Inversion::update_model()
{
    stage_info = "Updating model ...";

    // Gradient interpolation!!!

    // float p[CUBIC][CUBIC];

    // float * aux_vp = new float[nPoints_out]();

    // int skipx = (int)((nx_out - 1)/(nx - 1));
    // int skipz = (int)((nz_out - 1)/(nz - 1));

    // reduce_boundary(Vp, vp);

    // for (int index = 0; index < nPoints_out; index++)
    // {
    //     int i = (int)(index % nz_out);
    //     int j = (int)(index / nz_out);   

    //     if ((i >= skipz) && (i < nz_out-skipz) && (j >= skipx) && (j < nx_out-skipx))
    //     {
    //         float z = i*dz_out;    
    //         float x = j*dx_out;    

    //         float x0 = floorf(x/dx)*dx;
    //         float z0 = floorf(z/dz)*dz;

    //         float x1 = floorf(x/dx)*dx + dx;
    //         float z1 = floorf(z/dz)*dz + dz;        

    //         float pdx = (x - x0) / (x1 - x0);  
    //         float pdz = (z - z0) / (z1 - z0); 

    //         for (int pIdz = 0; pIdz < CUBIC; pIdz++)
    //         {
    //             for (int pIdx = 0; pIdx < CUBIC; pIdx++)
    //             {
    //                 int pz = (int)(floorf(z0/dz)) + pIdz - 1;
    //                 int px = (int)(floorf(x0/dx)) + pIdx - 1;    
                    
    //                 if (pz < 0) pz = 0;
    //                 if (px < 0) px = 0;
                    
    //                 if (pz > nz) pz = nz;
    //                 if (px > nx) px = nx;
                    
    //                 p[pIdx][pIdz] = vp[pz + px*nz];
    //             }    
    //         }

    //         vp_out[i + j*nz_out] = cubic2d(p, pdx, pdz); 
    //     }
    // }
    
    // for (int i = 0; i < skipz; i++)
    // {
    //     for (int j = skipx; j < nx_out-skipx; j++)
    //     {
    //         vp_out[i + j*nz_out] = vp_out[skipz + j*nz_out];
    //         vp_out[nz_out-i-1 + j*nz_out] = vp_out[nz_out-skipz-1 + j*nz_out];
    //     }
    // }

    // for (int i = 0; i < nz_out; i++)
    // {
    //     for (int j = 0; j < skipx; j++)
    //     {
    //         vp_out[i + j*nz_out] = vp_out[i + skipx*nz_out];
    //         vp_out[i + (nx_out-j-1)*nz_out] = vp_out[i + (nx_out-skipx-1)*nz_out];
    //     }
    // }






}

void Inversion::export_convergence()
{



}

void Inversion::export_final_model()
{
    model_file = output_folder + "final_model_" + std::to_string(int(freqs[freqId])) + "Hz_" + std::to_string(nz_out) + "x" + std::to_string(nx_out) + ".bin";





    export_binary_float(model_file, vp, nPoints_out);
}

void Inversion::refresh_memory()
{
    delete[] Vp;
    delete[] aux_vp;
    delete[] partial;
    delete[] obs_data;
    delete[] cal_data;
    delete[] gradient;
    delete[] seismogram;
    delete[] seismic_data;

    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFree(d_P);
    cudaFree(d_Pr);
    cudaFree(d_Vp);
    cudaFree(d_Pold);
    cudaFree(d_Prold);
    cudaFree(d_wavelet);
    cudaFree(d_Vp_clean);
    cudaFree(d_gradient);
    cudaFree(d_seismogram);
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
