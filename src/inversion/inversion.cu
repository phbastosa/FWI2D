# include "inversion.cuh"

void Inversion::set_parameters()
{
    title = "\033[34mFull Waveform Inversion\033[0;0m";
    
    set_main_parameters();

    set_abc_dampers();

    set_wavelet();
    set_geometry();    
    set_properties();
    set_seismograms();
    
    set_ABC_dimension();
    set_RBC_dimension();
    get_RBC_dimension();
    set_coordinates();

    zmask = std::stof(catch_parameter("depth_mask", parameters));

    data_folder = catch_parameter("inv_dcal_folder", parameters);
    input_folder = catch_parameter("inv_input_folder", parameters);
    input_prefix = catch_parameter("inv_input_prefix", parameters);
    output_folder = catch_parameter("inv_output_folder", parameters);
    residuo_folder = catch_parameter("inv_residuo_folder", parameters);

    max_iteration = std::stoi(catch_parameter("max_iteration", parameters));

    iteration = 0;

    model = new float[nPoints]();

    partial1 = new float[nPoints]();
    partial2 = new float[rbc_matsize]();

    gradient = new float[nPoints]();
    obs_data = new float[nt*geometry->nrec]();

    set_initial_model();

    cudaMalloc((void**)&(d_Vp_rbc), rbc_matsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Ps_rbc), rbc_matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pr_rbc), rbc_matsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Ps_old_rbc), rbc_matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pr_old_rbc), rbc_matsize*sizeof(float));
    
    cudaMalloc((void**)&(d_sumPs_rbc), rbc_matsize*sizeof(float));
    cudaMalloc((void**)&(d_gradient_rbc), rbc_matsize*sizeof(float));
}

void Inversion::set_ABC_dimension()
{
    abc_nb = (int)(abc_length / dh) + 1;

    abc_nxx = nx + 2*abc_nb;
    abc_nzz = nz + 2*abc_nb;
    abc_matsize = abc_nxx*abc_nzz;
}

void Inversion::get_ABC_dimension()
{
    ABC = true;

    nb = abc_nb;
    nxx = abc_nxx;
    nzz = abc_nzz;
    matsize = abc_matsize;

    nBlocks = (int)((abc_matsize + NTHREADS - 1) / NTHREADS);
}

void Inversion::set_RBC_dimension()
{
    rbc_ratio = std::stof(catch_parameter("inv_rbc_ratio", parameters)); 
    rbc_varVp = std::stof(catch_parameter("inv_rbc_varVp", parameters)); 
    rbc_length = std::stof(catch_parameter("inv_rbc_length", parameters));

    rbc_nb = (int)(rbc_length / dh) + 1;

    rbc_nxx = nx + 2*rbc_nb;
    rbc_nzz = nz + 2*rbc_nb;
    rbc_matsize = rbc_nxx*rbc_nzz;
}

void Inversion::get_RBC_dimension()
{
    ABC = false;

    nb = rbc_nb;
    nxx = rbc_nxx;
    nzz = rbc_nzz;
    matsize = rbc_matsize;

    nBlocks = (int)((rbc_matsize + NTHREADS - 1) / NTHREADS);
}

void Inversion::set_initial_model()
{
    get_ABC_dimension();
    reduce_boundary(Vp, model);
    for (int index = 0; index < nPoints; index++)
        model[index] = 1.0f / (model[index]*model[index]);
}

void Inversion::compute_gradient()
{
    sum_res = 0;

    cudaMemset(d_gradient_rbc, 0.0f, rbc_matsize*sizeof(float));

    if (iteration == max_iteration) ++iteration;
    
    for (srcId = 0; srcId < geometry->nsrc; srcId++)
    {
        set_obs_data();
        get_cal_data();
        
        set_adjoint_source();

        if (iteration <= max_iteration)
        {
            update_RBC();

            initialization();
            forward_propagation();
            backward_propagation();
        }
    }

    if (iteration == max_iteration) --iteration;
    
    residuo.push_back(sqrtf(sum_res));    

    cudaMemcpy(partial2, d_gradient_rbc, rbc_matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial2, gradient);

    cudaMemcpy(partial2, d_sumPs_rbc, rbc_matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial2, partial1);

    for (int index = 0; index < nPoints; index++)
        partial1[index] = gradient[index] / partial1[index];

    float gmax = -1e9f;

    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);

        gradient[index] = 0.0f;    

        if((i > (int)(zmask/dh)) && (i < nz-1)) 
            gradient[index] = -1.0f*(partial1[index-1] - 2.0f*partial1[index] + partial1[index+1]) * idh2;
    
        gmax = gmax < fabsf(gradient[index]) ? fabsf(gradient[index]) : gmax; 
    }

    for (int index = 0; index < nPoints; index++)
        gradient[index] *= 1.0f / gmax;
}

void Inversion::set_obs_data()
{
    std::string data_file = input_folder + input_prefix + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, obs_data, nt*geometry->nrec);
}

void Inversion::get_cal_data()
{
    stage_info = "Computing sinthetic data";

    get_ABC_dimension();
    show_information();

    initialization();
    forward_solver();
}

void Inversion::set_adjoint_source()
{
    cudaMemcpy(seismogram, d_seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);

    for (int index = 0; index < nt*geometry->nrec; index++)
    {
        seismogram[index] = obs_data[index] - seismogram[index];
        sum_res += seismogram[index]*seismogram[index];
    }    

    cudaMemcpy(d_seismogram, seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::update_RBC()
{
    cudaMemcpy(Vp, d_Vp, abc_matsize*sizeof(float), cudaMemcpyDeviceToHost);

    get_ABC_dimension();
    reduce_boundary(Vp, partial1);
    get_RBC_dimension();
    expand_boundary(partial1, partial2);

    cudaMemcpy(d_Vp_rbc, partial2, rbc_matsize*sizeof(float), cudaMemcpyHostToDevice);

    set_random_boundary(d_Vp_rbc, rbc_ratio, rbc_varVp);
}

void Inversion::forward_propagation()
{
    stage_info = "Wavefield reconstruction: forward propagation";

    show_information();

    cudaMemset(d_Ps_rbc, 0.0f, rbc_matsize*sizeof(float));
    cudaMemset(d_Ps_old_rbc, 0.0f, rbc_matsize*sizeof(float));

    for (int tId = 0; tId < tlag + nt; tId++)
    {
        compute_pressure<<<nBlocks,NTHREADS>>>(d_Vp_rbc, d_Ps_rbc, d_Ps_old_rbc, d_wavelet, d_b1d, d_b2d, sIdx, sIdz, tId, nt, nb, nxx, nzz, idh2, dt, ABC);

        std::swap(d_Ps_rbc, d_Ps_old_rbc);
    }    
}

void Inversion::backward_propagation()
{
    stage_info = "Wavefield reconstruction: backward propagation";

    show_information();
    
    cudaMemset(d_Pr_rbc, 0.0f, rbc_matsize*sizeof(float));
    cudaMemset(d_Pr_old_rbc, 0.0f, rbc_matsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        inject_adjoint<<<sBlocks,NTHREADS>>>(d_Pr_rbc, d_rIdx, d_rIdz, d_seismogram, geometry->nrec, tId, nt, nzz, idh2);

        build_gradient<<<nBlocks,NTHREADS>>>(d_Ps_rbc, d_Ps_old_rbc, d_Pr_rbc, d_Pr_old_rbc, d_Vp_rbc, d_gradient_rbc, d_sumPs_rbc, nxx, nzz, nt, dt, idh2);
    
        std::swap(d_Ps_rbc, d_Ps_old_rbc);
        std::swap(d_Pr_rbc, d_Pr_old_rbc);
    }
}

void Inversion::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << "\n\n";
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << 
                                  ", x = " << (nx - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nsrc << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[srcId] << 
                                       ", x = " << geometry->xsrc[srcId] << ") m\n\n";

    std::cout << line << "\n";
    std::cout << stage_info << "\n";
    std::cout << line << "\n\n";

    if (iteration > max_iteration) 
        std::cout << "-------- Checking final residuo --------\n\n";
    else
    {    
        std::cout << "-------- Computing iteration " << iteration+1 << " of " << max_iteration << " --------\n\n";
        
        if (iteration > 0) std::cout << "Previous residuo: " << residuo.back() << "\n\n";   
    }
}

void Inversion::check_convergence()
{
    ++iteration;
    
    converged = (iteration > max_iteration) ? true : false;

    if (converged) std::cout << "Final residuo: "<< residuo.back() <<"\n\n";  
}

void Inversion::optimization()
{
    float dm = (1.0f / (vmin*vmin)) - (1.0f / (vmax*vmax));

    float a0 = 0.00f*dm;
    float a1 = 0.02f*dm;
    float a2 = 0.05f*dm;

    float f0 = residuo.back();

    stage_info = "Optimization via parabolic linesearch: first modeling";

    linesearch(a1); float f1 = sqrtf(sum_res);

    stage_info = "Optimization via parabolic linesearch: final modeling";
    
    linesearch(a2); float f2 = sqrtf(sum_res);

    float num = (a1*a1 - a2*a2)*f0 + (a2*a2 - a0*a0)*f1 + (a0*a0 - a1*a1)*f2;    
    float den = (a1 - a2)*f0 + (a2 - a0)*f1 + (a0 - a1)*f2;

    step = 0.5f*(num / den);
}

void Inversion::linesearch(float alpha)
{
    for (int index = 0; index < nPoints; index++)
    {
        partial1[index] = model[index] - alpha*gradient[index];
        partial1[index] = 1.0f / sqrtf(partial1[index]);
    }

    get_ABC_dimension();
    expand_boundary(partial1, Vp);

    cudaMemcpy(d_Vp, Vp, abc_matsize*sizeof(float), cudaMemcpyHostToDevice);

    sum_res = 0.0f;

    for (srcId = 0; srcId < geometry->nsrc; srcId++)    
    {
        set_obs_data();

        show_information();

        initialization();
        forward_solver();
        
        cudaMemcpy(seismogram, d_seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int index = 0; index < nt*geometry->nrec; index++)
        {    
            seismogram[index] = obs_data[index] - seismogram[index];
            sum_res += seismogram[index]*seismogram[index];
        }
    }
}

void Inversion::update_model()
{
    for (int index = 0; index < nPoints; index++)
    {
        model[index] = model[index] - step*gradient[index];
        partial1[index] = 1.0f / sqrtf(model[index]);
    }

    expand_boundary(partial1, Vp);
}

void Inversion::export_convergence()
{
    std::string residuo_path = residuo_folder + "convergence_" + std::to_string(max_iteration) + "_iterations.txt"; 

    std::ofstream resFile(residuo_path, std::ios::out);
    
    for (int r = 0; r < max_iteration; r++) 
        resFile << residuo[r] << "\n";

    resFile.close();

    std::cout << "Text file \033[34m" << residuo_path << "\033[0;0m was successfully written." << std::endl;
}

void Inversion::export_final_model()
{
    std::string model_file = output_folder + "model_FWI_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + ".bin";
    reduce_boundary(Vp, partial1);
    export_binary_float(model_file, partial1, nPoints);
}

__global__ void inject_adjoint(float * Pr, int * rIdx, int * rIdz, float * seismogram, int nr, int tId, int nt, int nzz, float idh2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < nr) && (tId < nt))
    {
        int sId = (nt-tId-1) + index*nt;
        int wId = rIdz[index] + rIdx[index]*nzz;

        atomicAdd(&Pr[wId], idh2*seismogram[sId]);
    }
}

__global__ void build_gradient(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * gradient, float * sumPs, int nxx, int nzz, int nt, float dt, float idh2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2Ps_dx2 = (-FDM1*(Psold[i + (j-4)*nzz] + Psold[i + (j+4)*nzz])
                          +FDM2*(Psold[i + (j-3)*nzz] + Psold[i + (j+3)*nzz])
                          -FDM3*(Psold[i + (j-2)*nzz] + Psold[i + (j+2)*nzz])
                          +FDM4*(Psold[i + (j-1)*nzz] + Psold[i + (j+1)*nzz])
                          -FDM5*(Psold[i + j*nzz]))*idh2;

        float d2Ps_dz2 = (-FDM1*(Psold[(i-4) + j*nzz] + Psold[(i+4) + j*nzz])
                          +FDM2*(Psold[(i-3) + j*nzz] + Psold[(i+3) + j*nzz])
                          -FDM3*(Psold[(i-2) + j*nzz] + Psold[(i+2) + j*nzz])
                          +FDM4*(Psold[(i-1) + j*nzz] + Psold[(i+1) + j*nzz])
                          -FDM5*(Psold[i + j*nzz]))*idh2;
        
        float d2Pr_dx2 = (-FDM1*(Pr[i + (j-4)*nzz] + Pr[i + (j+4)*nzz])
                          +FDM2*(Pr[i + (j-3)*nzz] + Pr[i + (j+3)*nzz])
                          -FDM3*(Pr[i + (j-2)*nzz] + Pr[i + (j+2)*nzz])
                          +FDM4*(Pr[i + (j-1)*nzz] + Pr[i + (j+1)*nzz])
                          -FDM5*(Pr[i + j*nzz]))*idh2;

        float d2Pr_dz2 = (-FDM1*(Pr[(i-4) + j*nzz] + Pr[(i+4) + j*nzz])
                          +FDM2*(Pr[(i-3) + j*nzz] + Pr[(i+3) + j*nzz])
                          -FDM3*(Pr[(i-2) + j*nzz] + Pr[(i+2) + j*nzz])
                          +FDM4*(Pr[(i-1) + j*nzz] + Pr[(i+1) + j*nzz])
                          -FDM5*(Pr[i + j*nzz]))*idh2;
        
        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];

        gradient[index] += dt*Pr[index]*(d2Ps_dx2 + d2Ps_dz2)*Vp[index]*Vp[index];   

        sumPs[index] += Ps[index]*Ps[index];
    }
}
