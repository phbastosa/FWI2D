# include "inversion.cuh"

void Inversion::set_parameters()
{
    title = "\033[34mFull Waveform Inversion\033[0;0m";
    
    set_main_parameters();
    
    set_wavelet();
    set_geometry();
    set_properties();
    set_seismograms();
    set_coordinates();
    set_cerjan_dampers();

    input_folder = catch_parameter("inversion_input_folder", parameters);
    output_folder = catch_parameter("inversion_output_folder", parameters);

    max_iteration = std::stoi(catch_parameter("max_iteration", parameters));

    iteration = 0;

    sumPs = new float[nPoints]();
    partial = new float[matsize]();
    gradient = new float[nPoints]();

    cudaMalloc((void**)&(d_Pr), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Vp_clean), matsize*sizeof(float));
    cudaMalloc((void**)&(d_gradient), matsize*sizeof(float));

    cudaMemcpy(d_Vp, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vp_clean, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::set_observed_data()
{
    obs_data = new float[nt*geometry->nTraces]();

    std::string input_file = input_folder + "seismogram_nt" + std::to_string(nt) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string((int)(fmax)) + "Hz_" + std::to_string(int(1e6f*dt)) + "us.bin";

    import_binary_float(input_file, obs_data, nt*geometry->nTraces); 
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
    
    cudaMemset(d_gradient, 0.0f, matsize*sizeof(float));
}

void Inversion::show_information()
{
    auto clear = system("clear");

    std::string line(width, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << 
                                  ", x = " << (nx - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

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
        FWI<<<nBlocks, nThreads>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_seismogram, d_gradient, d_sumPs, d_rIdx, d_rIdz, geometry->spread, tId, nxx, nzz, nt, dh, dh, dt);
    
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
    stage_info = "Optimizing problem with ";

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

    //         float x0 = floorf(x/dh)*dh;
    //         float z0 = floorf(z/dh)*dh;

    //         float x1 = floorf(x/dh)*dh + dh;
    //         float z1 = floorf(z/dh)*dh + dh;        

    //         float pdx = (x - x0) / (x1 - x0);  
    //         float pdz = (z - z0) / (z1 - z0); 

    //         for (int pIdz = 0; pIdz < CUBIC; pIdz++)
    //         {
    //             for (int pIdx = 0; pIdx < CUBIC; pIdx++)
    //             {
    //                 int pz = (int)(floorf(z0/dh)) + pIdz - 1;
    //                 int px = (int)(floorf(x0/dh)) + pIdx - 1;    
                    
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
    model_file = output_folder + "final_model_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + ".bin";





//    export_binary_float(model_file, model, nPoints);
}

__global__ void FWI(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * gradient, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dh, float dh, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
    {
        for (int rId = 0; rId < spread; rId++)
        {
            Pr[(rIdz[rId] + 1) + rIdx[rId]*nzz] += 0.5f*seismogram[(nt-tId-1) + rId*nt] / (dh*dh); 
            Pr[(rIdz[rId] - 1) + rIdx[rId]*nzz] += 0.5f*seismogram[(nt-tId-1) + rId*nt] / (dh*dh); 
        }
    }    

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2Ps_dx2 = (- 9.0f*(Psold[i + (j-4)*nzz] + Psold[i + (j+4)*nzz])
                      +   128.0f*(Psold[i + (j-3)*nzz] + Psold[i + (j+3)*nzz])
                      -  1008.0f*(Psold[i + (j-2)*nzz] + Psold[i + (j+2)*nzz])
                      +  8064.0f*(Psold[i + (j+1)*nzz] + Psold[i + (j-1)*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dh*dh);

        float d2Ps_dz2 = (- 9.0f*(Psold[(i-4) + j*nzz] + Psold[(i+4) + j*nzz])
                      +   128.0f*(Psold[(i-3) + j*nzz] + Psold[(i+3) + j*nzz])
                      -  1008.0f*(Psold[(i-2) + j*nzz] + Psold[(i+2) + j*nzz])
                      +  8064.0f*(Psold[(i-1) + j*nzz] + Psold[(i+1) + j*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dh*dh);

        float d2Pr_dx2 = (- 9.0f*(Pr[i + (j-4)*nzz] + Pr[i + (j+4)*nzz])
                      +   128.0f*(Pr[i + (j-3)*nzz] + Pr[i + (j+3)*nzz])
                      -  1008.0f*(Pr[i + (j-2)*nzz] + Pr[i + (j+2)*nzz])
                      +  8064.0f*(Pr[i + (j+1)*nzz] + Pr[i + (j-1)*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dh*dh);

        float d2Pr_dz2 = (- 9.0f*(Pr[(i-4) + j*nzz] + Pr[(i+4) + j*nzz])
                      +   128.0f*(Pr[(i-3) + j*nzz] + Pr[(i+3) + j*nzz])
                      -  1008.0f*(Pr[(i-2) + j*nzz] + Pr[(i+2) + j*nzz])
                      +  8064.0f*(Pr[(i-1) + j*nzz] + Pr[(i+1) + j*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dh*dh);

        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];
        
        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];

        gradient[index] += dt*Pr[index]*(d2Ps_dx2 + d2Ps_dz2)*Vp[index]*Vp[index];   

        sumPs[index] += Ps[index]*Ps[index];
    }
}
