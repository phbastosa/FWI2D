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

void Migration::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << "\n";
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << "\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << 
                                  ", x = " << (nx - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << line << "\n";
    std::cout << stage_info << std::endl;
    std::cout << line << "\n";                                                                          
}

void Migration::forward_propagation()
{   
    stage_info = "Forward propagation";

    show_information();

    set_random_boundary(d_Vp, rbc_ratio, rbc_varVp);
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage_info = "Backward propagation";

    show_information();

    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, matsize*sizeof(float));

    float idh2 = 1.0f / (dh*dh);

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        inject_seismogram<<<sBlocks, NTHREADS>>>(d_Pr, d_rIdx, d_rIdz, d_seismogram, geometry->spread, tId, nt, nzz, idh2);

        RTM<<<nBlocks, NTHREADS>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_image, d_sumPs, nxx, nzz, idh2, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }    
}

void Migration::set_seismic_source()
{
    std::string data_file = input_folder + input_prefix + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, seismogram, nt*geometry->spread);
    cudaMemcpy(d_seismogram, seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
}

void Migration::export_seismic()
{
    cudaMemcpy(partial, d_image, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, image);

    cudaMemcpy(partial, d_sumPs, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        image[index] = image[index] / sumPs[index];

    # pragma omp parallel for    
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);

        if((i > 0) && (i < nz-1)) 
            sumPs[index] = -1.0f*(image[index-1] - 2.0f*image[index] + image[index+1]) / (dh*dh);
        else 
            sumPs[index] = 0.0f;    
    }

    std::string output_file = output_folder + "RTM_section_" + std::to_string(nz) + "x" + std::to_string(nx) + ".bin";
    export_binary_float(output_file, sumPs, nPoints);
}

__global__ void inject_seismogram(float * Pr, int * rIdx, int * rIdz, float * seismogram, int spread, int tId, int nt, int nzz, float idh2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId < nt))
    {
        int sId = (nt-tId-1) + index*nt;
        int wId = rIdz[index] + rIdx[index]*nzz;

        atomicAdd(&Pr[wId], idh2*seismogram[sId]);
    }
}

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * image, float * sumPs, int nxx, int nzz, float idh2, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);
    
    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2Ps_dx2 = idh2*(-FDM1*(Psold[i + (j-4)*nzz] + Psold[i + (j+4)*nzz])
                               +FDM2*(Psold[i + (j-3)*nzz] + Psold[i + (j+3)*nzz])
                               -FDM3*(Psold[i + (j-2)*nzz] + Psold[i + (j+2)*nzz])
                               +FDM4*(Psold[i + (j-1)*nzz] + Psold[i + (j+1)*nzz])
                               -FDM5*(Psold[i + j*nzz]));

        float d2Ps_dz2 = idh2*(-FDM1*(Psold[(i-4) + j*nzz] + Psold[(i+4) + j*nzz])
                               +FDM2*(Psold[(i-3) + j*nzz] + Psold[(i+3) + j*nzz])
                               -FDM3*(Psold[(i-2) + j*nzz] + Psold[(i+2) + j*nzz])
                               +FDM4*(Psold[(i-1) + j*nzz] + Psold[(i+1) + j*nzz])
                               -FDM5*(Psold[i + j*nzz]));
        
        float d2Pr_dx2 = idh2*(-FDM1*(Pr[i + (j-4)*nzz] + Pr[i + (j+4)*nzz])
                               +FDM2*(Pr[i + (j-3)*nzz] + Pr[i + (j+3)*nzz])
                               -FDM3*(Pr[i + (j-2)*nzz] + Pr[i + (j+2)*nzz])
                               +FDM4*(Pr[i + (j-1)*nzz] + Pr[i + (j+1)*nzz])
                               -FDM5*(Pr[i + j*nzz]));

        float d2Pr_dz2 = idh2*(-FDM1*(Pr[(i-4) + j*nzz] + Pr[(i+4) + j*nzz])
                               +FDM2*(Pr[(i-3) + j*nzz] + Pr[(i+3) + j*nzz])
                               -FDM3*(Pr[(i-2) + j*nzz] + Pr[(i+2) + j*nzz])
                               +FDM4*(Pr[(i-1) + j*nzz] + Pr[(i+1) + j*nzz])
                               -FDM5*(Pr[i + j*nzz]));
        
        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];
        
        atomicAdd(&sumPs[index], Ps[index]*Ps[index]);
        atomicAdd(&image[index], Ps[index]*Pr[index]);
    }
}
