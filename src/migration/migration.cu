# include "migration.cuh"

void Migration::set_parameters()
{    
    title = "\033[34mReverse Time Migration\033[0;0m";

    set_main_parameters();

    set_wavelet();
    set_geometry();
    set_properties();
    set_seismograms();
    set_coordinates();

    ABC = false;

    input_folder = catch_parameter("migration_input_folder", parameters);
    output_folder = catch_parameter("migration_output_folder", parameters);

    std::string input_file = input_folder + "seismogram_nt" + std::to_string(nt) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(int(1e6f*dt)) + "us.bin";

    import_binary_float(input_file, seismic_data, geometry->nTraces*nt); 

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

    std::string line(width, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << 
                                  ", x = " << (nx - 1)*dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << stage_info << std::endl;
    std::cout << "-------------------------------------------------------------------------------\n";                                                                          
}

void Migration::forward_propagation()
{   
    stage_info = "Forward propagation";

    show_information();

    set_random_boundary();
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage_info = "Backward propagation";

    show_information();

    initialization();
    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, matsize*sizeof(float));

    for (int tId = 0; tId < nt; tId++)
    {
        RTM<<<nBlocks, nThreads>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_seismogram, d_image, d_sumPs, d_rIdx, d_rIdz, geometry->spread, tId, nxx, nzz, nt, dx, dz, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }    
}

void Migration::set_seismic_source()
{
    for (int timeId = 0; timeId < nt; timeId++)
        for (int spreadId = 0; spreadId < geometry->spread; spreadId++)
            seismogram[timeId + spreadId*nt] = seismic_data[timeId + spreadId*nt + srcId*geometry->spread*nt];     

    cudaMemcpy(d_seismogram, seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
}

void Migration::image_enhancing()
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
        int j = (int)(index / nz);        

        if((i > 0) && (i < nz-1) && (j > 0) && (j < nx-1)) 
        {
            float d2I_dz2 = (image[(i-1) + j*nz] - 2.0f*image[i + j*nz] + image[(i+1) + j*nz]) / (dz * dz);

            image[index] = d2I_dz2;
        }
        else image[index] = 0.0f;    
    }
}

void Migration::export_seismic()
{
    std::string output_file = output_folder + "RTM_section_" + std::to_string(nz) + "x" + std::to_string(nx) + "_" + std::to_string((int)(fmax)) + "Hz.bin";
    export_binary_float(output_file, image, nPoints);
}

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt)
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
    
        sumPs[index] += Ps[index]*Ps[index]; 
        image[index] += Ps[index]*Pr[index];
    }
}

