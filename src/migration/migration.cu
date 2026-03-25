# include "migration.cuh"

void Migration::set_parameters()
{    
    title = "\033[34mReverse Time Migration\033[0;0m";

    set_main_parameters();

    rbc_ratio = std::stof(catch_parameter("mig_rbc_ratio", parameters)); 
    rbc_varVp = std::stof(catch_parameter("mig_rbc_varVp", parameters)); 
    rbc_length = std::stof(catch_parameter("mig_rbc_length", parameters));

    nb = (int)(rbc_length / dh) + 1;

    set_wavelet();
    set_geometry();    
    set_seismograms();
    set_properties();
    set_coordinates();

    input_folder = catch_parameter("mig_input_folder", parameters);
    input_prefix = catch_parameter("mig_input_prefix", parameters);    
    output_folder = catch_parameter("mig_output_folder", parameters);

    image = new float[nPoints]();
    sumPs = new float[nPoints]();
    partial = new float[matsize]();

    cudaMalloc((void**)&(d_Pr), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_image), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), matsize*sizeof(float));

    cudaMemset(d_image, 0.0f, matsize*sizeof(float));
    cudaMemset(d_sumPs, 0.0f, matsize*sizeof(float));
}

void Migration::show_mig_info()
{
    show_information();

    std::string line(WIDTH, '-');

    std::cout << line << "\n";
    std::cout << stage_info << std::endl;
    std::cout << line << "\n";                                                                          
}

void Migration::forward_propagation()
{   
    stage_info = "Forward propagation";

    show_mig_info();

    set_random_boundary(d_Vp, rbc_ratio, rbc_varVp);
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage_info = "Backward propagation";

    show_mig_info();

    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, matsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        inject_seismogram<<<sBlocks, NTHREADS>>>(d_Pr, d_rIdx, d_rIdz, d_seismogram, geometry->nrec, tId, nt, nzz, idh2);

        cross_correlation<<<nBlocks, NTHREADS>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_image, d_sumPs, nxx, nzz, idh2, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }    
}

void Migration::set_seismic_source()
{
    std::string data_file = input_folder + input_prefix + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, seismogram, nt*geometry->nrec);
    cudaMemcpy(d_seismogram, seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
}

void Migration::export_seismic()
{
    cudaMemcpy(partial, d_image, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, image);

    cudaMemcpy(partial, d_sumPs, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        sumPs[index] = image[index] / sumPs[index];

    # pragma omp parallel for    
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);

        image[index] = 0.0f;    

        if((i > 0) && (i < nz-1)) 
            image[index] = -1.0f*(sumPs[index-1] - 2.0f*sumPs[index] + sumPs[index+1]) * idh2;
    }

    std::string output_file = output_folder + "RTM_section_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + "_" + std::to_string((int)(dh)) + "m.bin";
    export_binary_float(output_file, image, nPoints);
}

__global__ void inject_seismogram(float * __restrict__ Pr, const int * __restrict__ rIdx, const int * __restrict__ rIdz, 
                                  const float * __restrict__ seismogram, int nr, int tId, int nt, int nzz, float idh2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < nr) && (tId < nt))
    {
        int sId = (nt-tId-1) + index*nt;
        int wId = rIdz[index] + rIdx[index]*nzz;

        atomicAdd(&Pr[wId], idh2*seismogram[sId]);
    }
}

__global__ void cross_correlation(float * __restrict__ Ps, const float * __restrict__ Psold, const float * __restrict__ Pr, 
                                  float * __restrict__ Prold, const float * __restrict__ Vp, float * __restrict__ image, 
                                  float * __restrict__ sumPs, int nxx, int nzz, float idh2, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    const int base_j = j*nzz;
    
    const int jm4 = base_j - 4*nzz, jm3 = base_j - 3*nzz, jm2 = base_j - 2*nzz, jm1 = base_j - nzz;
    const int jp4 = base_j + 4*nzz, jp3 = base_j + 3*nzz, jp2 = base_j + 2*nzz, jp1 = base_j + nzz;
    
    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        const float vp2 = Vp[index]*Vp[index];

        float d2Ps_dx2 = (-FDM1*(Psold[i + jm4] + Psold[i + jp4])
                          +FDM2*(Psold[i + jm3] + Psold[i + jp3])
                          -FDM3*(Psold[i + jm2] + Psold[i + jp2])
                          +FDM4*(Psold[i + jm1] + Psold[i + jp1])
                          -FDM5*(Psold[i + base_j]))*idh2;

        float d2Ps_dz2 = (-FDM1*(Psold[(i-4) + base_j] + Psold[(i+4) + base_j])
                          +FDM2*(Psold[(i-3) + base_j] + Psold[(i+3) + base_j])
                          -FDM3*(Psold[(i-2) + base_j] + Psold[(i+2) + base_j])
                          +FDM4*(Psold[(i-1) + base_j] + Psold[(i+1) + base_j])
                          -FDM5*(Psold[i + base_j]))*idh2;
        
        float d2Pr_dx2 = (-FDM1*(Pr[i + jm4] + Pr[i + jp4])
                          +FDM2*(Pr[i + jm3] + Pr[i + jp3])
                          -FDM3*(Pr[i + jm2] + Pr[i + jp2])
                          +FDM4*(Pr[i + jm1] + Pr[i + jp1])
                          -FDM5*(Pr[i + base_j]))*idh2;

        float d2Pr_dz2 = (-FDM1*(Pr[(i-4) + base_j] + Pr[(i+4) + base_j])
                          +FDM2*(Pr[(i-3) + base_j] + Pr[(i+3) + base_j])
                          -FDM3*(Pr[(i-2) + base_j] + Pr[(i+2) + base_j])
                          +FDM4*(Pr[(i-1) + base_j] + Pr[(i+1) + base_j])
                          -FDM5*(Pr[i + base_j]))*idh2;
        
        Ps[index] = dt*dt*vp2*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*vp2*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];
        
        atomicAdd(&sumPs[index], Ps[index]*Ps[index]);
        atomicAdd(&image[index], Ps[index]*Pr[index]);
    }
}
