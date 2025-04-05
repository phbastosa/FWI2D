# include "migration.cuh"

void Migration::set_parameters()
{    
    set_main_parameters();

    set_wavelet();
    set_geometry();
    set_properties();

    vmax = 0.0f;
    vmin = 1e9f;

    for (int index = 0; index < matsize; index++)
    {
        vmax = vmax < Vp[index] ? Vp[index] : vmax; 
        vmin = vmin > Vp[index] ? Vp[index] : vmin; 
    }

    ABC = false;

    rbc_ratio = std::stof(catch_parameter("rbc_ratio", parameters));
    rbc_varVp = std::stof(catch_parameter("rbc_varVp", parameters));

    set_coordinates();

    input_file = catch_parameter("migration_input_file", parameters);
    output_folder = catch_parameter("migration_output_folder", parameters);

    import_binary_float(input_file, seismic_data, geometry->nTraces*nt); 

    image = new float[nPoints]();
    sumPs = new float[nPoints]();
    sumPr = new float[nPoints]();

    partial = new float[matsize]();

    cudaMalloc((void**)&(d_Pr), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), matsize*sizeof(float));
    cudaMalloc((void**)&(d_image), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), matsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPr), matsize*sizeof(float));

    cudaMemset(d_image, 0.0f, matsize*sizeof(float));
    cudaMemset(d_sumPs, 0.0f, matsize*sizeof(float));
    cudaMemset(d_sumPr, 0.0f, matsize*sizeof(float));
}

void Migration::set_coordinates()
{
    float * h_Z = new float[nzz]();   
    # pragma omp parallel for 
    for (int i = 0; i < nzz; i++) 
        h_Z[i] = (float)(i)*dz;

    float * h_X = new float[nxx]();   
    # pragma omp parallel for 
    for (int j = 0; j < nxx; j++) 
        h_X[j] = (float)(j)*dx;
    
    cudaMalloc((void**)&(d_X), nxx*sizeof(float));
    cudaMemcpy(d_X, h_X, nxx*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(d_Z), nzz*sizeof(float));
    cudaMemcpy(d_Z, h_Z, nzz*sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_X;
    delete[] h_Z;
}

void Migration::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "                          \033[34mReverse Time Migration\033[0;0m\n";
    std::cout << "-------------------------------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << 
                                  ", x = " << (nx - 1)*dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << stage << std::endl;
    std::cout << "-------------------------------------------------------------------------------\n";                                                                          
}

void Migration::forward_propagation()
{   
    stage = "Forward propagation";

    show_information();

    set_random_boundary();
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage = "Backward propagation";

    show_information();

    initialization();
    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, matsize*sizeof(float));

    for (int tId = 0; tId < nt; tId++)
    {
        RTM<<<nBlocks, nThreads>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_seismogram, d_image, d_sumPs, d_sumPr, d_rIdx, d_rIdz, geometry->spread, tId, nxx, nzz, nt, dx, dz, dt);
    
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

    cudaMemcpy(partial, d_sumPr, matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPr);

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
    std::string output_file = output_folder + "RTM_section_" + std::to_string(nz) + "x" + std::to_string(nx) + "_" + std::to_string((int)(fmax)) + "Hz_" + std::to_string((int)(dz)) + "m.bin";
    export_binary_float(output_file, image, nPoints);
}

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, float * sumPr, int * rIdx, int * rIdz, int spread, int tId, int nxx, int nzz, int nt, float dx, float dz, float dt)
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
        sumPr[index] += Pr[index]*Pr[index]; 
        image[index] += Ps[index]*Pr[index];
    }
}

std::random_device mig;  
std::mt19937 mig_rng(mig()); 

void Migration::set_random_boundary()
{
    float x_max = (nxx-1)*dx;
    float z_max = (nzz-1)*dz;

    float xb = nb*dx;
    float zb = nb*dz;

    random_boundary_bg<<<nBlocks,nThreads>>>(d_Vp, nxx, nzz, nb, rbc_varVp);

    std::vector<Point> points = poissonDiskSampling(x_max, z_max, rbc_ratio);
    std::vector<Point> target;
    
    for (int index = 0; index < points.size(); index++)
    {
        if (!((points[index].x > 0.5f*xb) && (points[index].x < x_max - 0.5f*xb) && 
              (points[index].z > 0.5f*zb) && (points[index].z < z_max - 0.5f*xb)))
            target.push_back(points[index]);
    }
    
    for (int p = 0; p < target.size(); p++)
    {
        float xc = target[p].x;
        float zc = target[p].z;
        
        float r = std::uniform_real_distribution<float>(0.1f*rbc_ratio, 0.9f*rbc_ratio)(mig_rng);
        float A = std::uniform_real_distribution<float>(-rbc_varVp, rbc_varVp)(mig_rng);

        random_boundary_gp<<<nBlocks,nThreads>>>(d_Vp, d_X, d_Z, nxx, nzz, x_max, z_max, xb, zb, A, xc, zc, r, vmax, vmin, rbc_varVp);
    }
}

__device__ float get_random_value(float velocity, float function, float parameter, int index, float varVp)
{
    curandState state;
    curand_init(clock64(), index, 0, &state);
    
    float value = velocity + function*parameter*curand_normal(&state);

    value = value > velocity + varVp ? velocity + varVp : value;
    value = value < velocity - varVp ? velocity - varVp : value;            

    return value;
}

__global__ void random_boundary_bg(float * Vp, int nxx, int nzz, int nb, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);   

    if ((i >= nb) && (i < nzz-nb) && (j >= 0) && (j < nb))
    {
        float f1d = 1.0f - (float)(j) / (float)(nb);
        
        int index1 = i + j*nzz;
        int index2 = i + (nxx-j-1)*nzz;     
        int index3 = i + nb*nzz;
        int index4 = i + (nxx-nb)*nzz;     

        Vp[index1] = get_random_value(Vp[index3], f1d, varVp, index, varVp);
        Vp[index2] = get_random_value(Vp[index4], f1d, varVp, index, varVp);        
    }
    
    if ((i >= 0) && (i < nb) && (j >= nb) && (j < nxx-nb))    
    {
        float f1d = 1.0f - (float)(i) / (float)(nb);
        
        int index1 = i + j*nzz;
        int index2 = nzz-i-1 + j*nzz;     
        int index3 = nb + j*nzz;
        int index4 = nzz-nb + j*nzz;     

        Vp[index1] = get_random_value(Vp[index3], f1d, varVp, index, varVp);
        Vp[index2] = get_random_value(Vp[index4], f1d, varVp, index, varVp);
    }

    if ((i >= 0) && (i < nb) && (j >= i) && (j < nb))
    {
        float f1d = 1.0f - (float)(i) / (float)(nb);

        Vp[j + i*nzz] = get_random_value(Vp[nb + nb*nzz], f1d, varVp, index, varVp);
        Vp[i + j*nzz] = get_random_value(Vp[nb + nb*nzz], f1d, varVp, index, varVp);    

        Vp[j + (nxx-i-1)*nzz] = get_random_value(Vp[nb + (nxx-nb)*nzz], f1d, varVp, index, varVp);
        Vp[i + (nxx-j-1)*nzz] = get_random_value(Vp[nb + (nxx-nb)*nzz], f1d, varVp, index, varVp);

        Vp[nzz-j-1 + i*nzz] = get_random_value(Vp[nzz-nb + nb*nzz], f1d, varVp, index, varVp);
        Vp[nzz-i-1 + j*nzz] = get_random_value(Vp[nzz-nb + nb*nzz], f1d, varVp, index, varVp);

        Vp[nzz-j-1 + (nxx-i-1)*nzz] = get_random_value(Vp[nzz-nb + (nxx-nb)*nzz], f1d, varVp, index, varVp);
        Vp[nzz-i-1 + (nxx-j-1)*nzz] = get_random_value(Vp[nzz-nb + (nxx-nb)*nzz], f1d, varVp, index, varVp);
    }
}

__global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, float xb, float zb, float A, float xc, float zc, float r, float vmax, float vmin, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);   
    
    if (!((X[j] > xb) && (X[j] < x_max - xb) && 
          (Z[i] > zb) && (Z[i] < z_max - zb)))
    {
        Vp[i + j*nzz] += A*expf(-0.5f*(((X[j]-xc)/r)*((X[j]-xc)/r) + 
                                       ((Z[i]-zc)/r)*((Z[i]-zc)/r)));
        
        Vp[i + j*nzz] = Vp[i + j*nzz] > vmax + varVp ? vmax + varVp : Vp[i + j*nzz];
        Vp[i + j*nzz] = Vp[i + j*nzz] < vmin - varVp ? vmin - varVp : Vp[i + j*nzz];         
    }   
}
