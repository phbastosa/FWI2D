# include "modeling.cuh"

void Modeling::set_parameters()
{
    title = "\033[34mSeismic Modeling\033[0;0m";

    set_main_parameters();

    set_wavelet();
    set_geometry();    
    set_seismograms();
    set_abc_dampers();
    set_properties();
}

void Modeling::set_main_parameters()
{
    nx = std::stoi(catch_parameter("x_samples", parameters));        
    nz = std::stoi(catch_parameter("z_samples", parameters));        

    dh = std::stof(catch_parameter("model_spacing", parameters));

    nt = std::stoi(catch_parameter("time_samples", parameters));
    dt = std::stof(catch_parameter("time_spacing", parameters));

    fmax = std::stof(catch_parameter("max_frequency", parameters));
    
    data_folder = catch_parameter("mod_output_folder", parameters);
}

void Modeling::set_wavelet()
{
    float * signal_aux = new float[nt]();

    float t0 = 2.0f*sqrtf(M_PI) / fmax;
    float fc = fmax / (3.0f * sqrtf(M_PI));

    tlag = (int)(t0 / dt) + 1;

    for (int n = 0; n < nt; n++)
    {
        float td = n*dt - t0;

        float arg = M_PI*M_PI*M_PI*fc*fc*td*td;

        signal_aux[n] = 1e5f*(1.0f - 2.0f*arg)*expf(-arg);
    }

    double * time_domain = (double *) fftw_malloc(nt*sizeof(double));

    fftw_complex * freq_domain = (fftw_complex *) fftw_malloc(nt*sizeof(fftw_complex));

    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(nt, time_domain, freq_domain, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(nt, freq_domain, time_domain, FFTW_ESTIMATE);

    double df = 1.0 / (nt * dt);  
    
    std::complex<double> j(0.0, 1.0);  

    for (int k = 0; k < nt; k++) time_domain[k] = (double) signal_aux[k];

    fftw_execute(forward_plan);

    for (int k = 0; k < nt; ++k) 
    {
        double f = (k <= nt / 2) ? k * df : (k - nt) * df;
        
        std::complex<double> half_derivative_filter = std::pow(2.0 * M_PI * f * j, 0.5);  

        std::complex<double> complex_freq(freq_domain[k][0], freq_domain[k][1]);
        std::complex<double> filtered_freq = complex_freq * half_derivative_filter;

        freq_domain[k][0] = filtered_freq.real();
        freq_domain[k][1] = filtered_freq.imag();
    }

    fftw_execute(inverse_plan);    

    for (int k = 0; k < nt; k++) signal_aux[k] = (float) time_domain[k] / nt;

    cudaMalloc((void**)&(d_wavelet), nt*sizeof(float));

    cudaMemcpy(d_wavelet, signal_aux, nt*sizeof(float), cudaMemcpyHostToDevice);

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);

    delete[] signal_aux;
}

void Modeling::set_geometry()
{
    geometry = new Geometry();
    geometry->parameters = parameters;
    geometry->set_parameters();
        
    cudaMalloc((void**)&(d_rIdx), geometry->spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), geometry->spread*sizeof(int));
}

void Modeling::set_seismograms()
{
    sBlocks = (int)((geometry->spread + NTHREADS - 1) / NTHREADS);

    seismogram = new float[nt*geometry->spread]();

    cudaMalloc((void**)&(d_seismogram), nt*geometry->spread*sizeof(float));
}

void Modeling::set_abc_dampers()
{
    ABC = true;

    abc_length = std::stof(catch_parameter("abc_length", parameters));
    abc_factor = std::stof(catch_parameter("abc_factor", parameters));

    nb = (int)(abc_length / dh) + 4;

    float * b1d = new float[nb]();
    float * b2d = new float[nb*nb]();

    for (int i = 0; i < nb; i++) 
    {
        b1d[i] = expf(-powf(abc_factor * (nb - i), 2.0f));
    }

    for(int i = 0; i < nb; i++) 
    {
        for (int j = 0; j < nb; j++)
        {   
            b2d[j + i*nb] += b1d[i];
            b2d[i + j*nb] += b1d[i];
        }
    }

    for (int index = 0; index < nb*nb; index++)
        b2d[index] -= 1.0f;

    cudaMalloc((void**)&(d_b1d), nb*sizeof(float));
    cudaMalloc((void**)&(d_b2d), nb*nb*sizeof(float));

    cudaMemcpy(d_b1d, b1d, nb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2d, b2d, nb*nb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] b1d;
    delete[] b2d;
}

void Modeling::set_properties()
{
    nPoints = nx*nz;

    nxx = nx + 2*nb;
    nzz = nz + 2*nb;

    matsize = nxx*nzz;

    nBlocks = (int)((matsize + NTHREADS - 1) / NTHREADS);

    Vp = new float[matsize]();

    cudaMalloc((void**)&(d_P), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vp), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pold), matsize*sizeof(float));

    float * vp = new float[nPoints]();
    std::string vp_file = catch_parameter("model_file", parameters);
    import_binary_float(vp_file, vp, nPoints);

    vmax = 0.0f;
    vmin = 1e9f;

    for (int index = 0; index < nPoints; index++)
    {
        vmax = vmax < vp[index] ? vp[index] : vmax; 
        vmin = vmin > vp[index] ? vp[index] : vmin; 
    }

    expand_boundary(vp, Vp);
    
    cudaMemcpy(d_Vp, Vp, matsize*sizeof(float), cudaMemcpyHostToDevice);

    delete[] vp;
}

void Modeling::expand_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int) (index % nz);  
        int j = (int) (index / nz);    

        output[(i + nb) + (j + nb)*nzz] = input[i + j*nz];
    }

    for (int i = 0; i < nb; i++)
    {
        for (int j = nb; j < nxx - nb; j++)
        {
            output[i + j*nzz] = output[nb + j*nzz];
            output[(nzz - i - 1) + j*nzz] = output[(nzz - nb - 1) + j*nzz];
        }
    }

    for (int i = 0; i < nzz; i++)
    {
        for (int j = 0; j < nb; j++)
        {
            output[i + j*nzz] = output[i + nb*nzz];
            output[i + (nxx - j - 1)*nzz] = output[i + (nxx - nb - 1)*nzz];
        }
    }
}

void Modeling::reduce_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int) (index % nz);  
        int j = (int) (index / nz);    

        output[i + j*nz] = input[(i + nb) + (j + nb)*nzz];
    }
}

void Modeling::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << ", x = " << (nx - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n";
}

void Modeling::initialization()
{
    float sx = geometry->xsrc[geometry->sInd[srcId]];
    float sz = geometry->zsrc[geometry->sInd[srcId]];

    sIdx = (int)((sx + 0.5f*dh) / dh) + nb;
    sIdz = (int)((sz + 0.5f*dh) / dh) + nb;

    int * h_rIdx = new int[geometry->spread]();
    int * h_rIdz = new int[geometry->spread]();

    int spreadId = 0;

    for (int recId = geometry->iRec[srcId]; recId < geometry->fRec[srcId]; recId++)
    {
        float rx = geometry->xrec[recId];
        float rz = geometry->zrec[recId];
        
        h_rIdx[spreadId] = (int)((rx + 0.5f*dh) / dh) + nb;
        h_rIdz[spreadId] = (int)((rz + 0.5f*dh) / dh) + nb;

        ++spreadId;
    }

    cudaMemcpy(d_rIdx, h_rIdx, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_rIdx;
    delete[] h_rIdz;  
}

void Modeling::get_seismogram()
{
    cudaMemcpy(seismogram, d_seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyDeviceToHost);
    std::string data_file = data_folder + "seismogram_nt" + std::to_string(nt) + "_nr" + std::to_string(geometry->spread) + "_" + std::to_string(int(1e6f*dt)) + "us_shot_" + std::to_string(srcId+1) + ".bin";
    export_binary_float(data_file, seismogram, nt*geometry->spread);  
}

void Modeling::forward_solver()
{
    cudaMemset(d_P, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Pold, 0.0f, matsize*sizeof(float));

    float idh2 = 1.0f / (dh*dh);

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        compute_pressure<<<nBlocks, NTHREADS>>>(d_Vp, d_P, d_Pold, d_wavelet, d_b1d, d_b2d, sIdx, sIdz, tId, nt, nb, nxx, nzz, idh2, dt, ABC);
        
        compute_seismogram<<<sBlocks, NTHREADS>>>(d_P, d_rIdx, d_rIdz, d_seismogram, geometry->spread, tId, tlag, nt, nzz);     

        std::swap(d_P, d_Pold);
    }
}

__global__ void compute_pressure(float * Vp, float * P, float * Pold, float * wavelet, float * b1d, float * b2d, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float idh2, float dt, bool ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
        atomicAdd(&P[sIdz + sIdx*nzz], idh2*wavelet[tId]);    

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2P_dx2 = idh2*(-FDM1*(P[i + (j-4)*nzz] + P[i + (j+4)*nzz])
                              +FDM2*(P[i + (j-3)*nzz] + P[i + (j+3)*nzz])
                              -FDM3*(P[i + (j-2)*nzz] + P[i + (j+2)*nzz])
                              +FDM4*(P[i + (j+1)*nzz] + P[i + (j-1)*nzz])
                              -FDM5*(P[i + j*nzz]));

        float d2P_dz2 = idh2*(-FDM1*(P[(i-4) + j*nzz] + P[(i+4) + j*nzz])
                              +FDM2*(P[(i-3) + j*nzz] + P[(i+3) + j*nzz])
                              -FDM3*(P[(i-2) + j*nzz] + P[(i+2) + j*nzz])
                              +FDM4*(P[(i-1) + j*nzz] + P[(i+1) + j*nzz])
                              -FDM5*(P[i + j*nzz]));

        Pold[index] = dt*dt*Vp[index]*Vp[index]*(d2P_dx2 + d2P_dz2) + 2.0f*P[index] - Pold[index];
        
        if (ABC)
        {
            float damper = get_boundary_damper(b1d, b2d, i, j, nxx, nzz, nb);

            P[index] *= damper;
            Pold[index] *= damper;
        }
    }
}

__global__ void compute_seismogram(float * P, int * rIdx, int * rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId > tlag))
        seismogram[(tId - tlag) + index*nt] = P[rIdz[index] + rIdx[index]*nzz];    
}

__device__ float get_boundary_damper(float * b1d, float * b2d, int i, int j, int nxx, int nzz, int nb)
{
    float damper;

    // global case
    if ((i >= nb) && (i < nzz - nb) && (j >= nb) && (j < nxx - nb))
    {
        damper = 1.0f;
    }

    // 1D damping
    else if ((i >= 0) && (i < nb) && (j >= nb) && (j < nxx - nb)) 
    {
        damper = b1d[i];
    }         
    else if ((i >= nzz - nb) && (i < nzz) && (j >= nb) && (j < nxx - nb)) 
    {
        damper = b1d[nb - (i - (nzz - nb)) - 1];
    }         
    else if ((i >= nb) && (i < nzz - nb) && (j >= 0) && (j < nb)) 
    {
        damper = b1d[j];
    }
    else if ((i >= nb) && (i < nzz - nb) && (j >= nxx - nb) && (j < nxx)) 
    {
        damper = b1d[nb - (j - (nxx - nb)) - 1];
    }

    // 2D damping 
    else if ((i >= 0) && (i < nb) && (j >= 0) && (j < nb))
    {
        damper = b2d[i + j*nb];
    }
    else if ((i >= nzz - nb) && (i < nzz) && (j >= 0) && (j < nb))
    {
        damper = b2d[nb - (i - (nzz - nb)) - 1 + j*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx - nb) && (j < nxx))
    {
        damper = b2d[i + (nb - (j - (nxx - nb)) - 1)*nb];
    }
    else if((i >= nzz - nb) && (i < nzz) && (j >= nxx - nb) && (j < nxx))
    {
        damper = b2d[nb - (i - (nzz - nb)) - 1 + (nb - (j - (nxx - nb)) - 1)*nb];
    }

    return damper;
}

// Random Boundary Condition

std::random_device RBC;  
std::mt19937 RBC_RNG(RBC()); 

void Modeling::set_coordinates()
{
    float * h_Z = new float[nzz]();   
    # pragma omp parallel for 
    for (int i = 0; i < nzz; i++) 
        h_Z[i] = (float)(i)*dh;

    float * h_X = new float[nxx]();   
    # pragma omp parallel for 
    for (int j = 0; j < nxx; j++) 
        h_X[j] = (float)(j)*dh;
    
    cudaMalloc((void**)&(d_X), nxx*sizeof(float));
    cudaMemcpy(d_X, h_X, nxx*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(d_Z), nzz*sizeof(float));
    cudaMemcpy(d_Z, h_Z, nzz*sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_X;
    delete[] h_Z;
}

void Modeling::set_random_boundary(float * Vp, float ratio, float varVp)
{
    float x_max = (nxx-1)*dh;
    float z_max = (nzz-1)*dh;

    random_boundary_bg<<<nBlocks,NTHREADS>>>(Vp, nxx, nzz, nb, varVp);

    std::vector<Point> points = poissonDiskSampling(x_max, z_max, ratio);
    std::vector<Point> target;

    float fnb = (float)(nb)*dh;
    
    for (int index = 0; index < points.size(); index++)
    {
        if (!((points[index].x > 0.5f*fnb) && (points[index].x < x_max - 0.5f*fnb) && 
              (points[index].z > 0.5f*fnb) && (points[index].z < z_max - 0.5f*fnb)))
            target.push_back(points[index]);
    }
    
    for (int p = 0; p < target.size(); p++)
    {
        float xc = target[p].x;
        float zc = target[p].z;

        int xId = (int)(xc / dh);
        int zId = (int)(zc / dh);

        float r = std::uniform_real_distribution<float>(0.5f*ratio, ratio)(RBC_RNG);
        float A = std::uniform_real_distribution<float>(0.5f*varVp, varVp)(RBC_RNG);

        float factor = rand() % 2 == 0 ? -1.0f : 1.0f;

        random_boundary_gp<<<nBlocks,NTHREADS>>>(Vp, d_X, d_Z, nxx, nzz, x_max, z_max, nb, dh, factor, A, xc, zc, r, vmax, vmin, varVp);
    }
}

__global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, int nb, float dh, float factor, float A, float xc, float zc, float r, float vmax, float vmin, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);   

    float fnb = (float)(nb)*dh;
    
    if (!((X[j] > fnb) && (X[j] < x_max - fnb) && 
          (Z[i] > fnb) && (Z[i] < z_max - fnb)))
    {
        Vp[i + j*nzz] += factor*A*expf(-0.5f*(((X[j]-xc)/r)*((X[j]-xc)/r) + 
                                              ((Z[i]-zc)/r)*((Z[i]-zc)/r)));
        
        Vp[i + j*nzz] = Vp[i + j*nzz] > vmax + 0.5f*varVp ? vmax + 0.5f*varVp : Vp[i + j*nzz];
        Vp[i + j*nzz] = Vp[i + j*nzz] < vmin - 0.5f*varVp ? vmin - 0.5f*varVp : Vp[i + j*nzz];         
    }   
}

__global__ void random_boundary_bg(float * Vp, int nxx, int nzz, int nb, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);   

    if (!((i >= nb) && (i < nzz - nb) &&
          (j >= nb) && (j < nxx - nb)))
    {
        int ii = min(max(i, nb), nzz - nb - 1);
        int jj = min(max(j, nb), nxx - nb - 1);

        int index_ref = ii + jj*nzz;

        float val = Vp[index_ref];

        if (j < nb) {
            float fx = 1.0f - (float)(j) / (float)(nb);
            val = get_random_value(val, fx, varVp, index);
        } 
        else if (j >= nxx - nb) {
            float fx = 1.0f - (float)(nxx - j - 1) / (float)(nb);
            val = get_random_value(val, fx, varVp, index);
        }

        if (i < nb) {
            float fz = 1.0f - (float)(i) / (float)(nb);
            val = get_random_value(val, fz, varVp, index);
        } 
        else if (i >= nzz - nb) {
            float fz = 1.0f - (float)(nzz - i - 1) / (float)(nb);
            val = get_random_value(val, fz, varVp, index);
        }

        Vp[index] = val;        
    }
}

__device__ float get_random_value(float velocity, float function, float parameter, int index)
{
    curandState state;
    curand_init(clock64(), index, 0, &state);
    return velocity + function*parameter*curand_normal(&state);
}
