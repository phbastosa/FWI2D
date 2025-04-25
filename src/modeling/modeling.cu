# include "modeling.cuh"

void Modeling::set_parameters()
{
    set_main_parameters();
    
    set_wavelet();
    set_geometry();
    set_properties();
    set_seismograms();
    set_cerjan_dampers();
}

void Modeling::set_main_parameters()
{
    nx = std::stoi(catch_parameter("x_samples", parameters));        
    nz = std::stoi(catch_parameter("z_samples", parameters));        

    dx = std::stof(catch_parameter("x_spacing", parameters));
    dz = std::stof(catch_parameter("z_spacing", parameters));

    nt = std::stoi(catch_parameter("time_samples", parameters));
    dt = std::stof(catch_parameter("time_spacing", parameters));

    nb = std::stoi(catch_parameter("boundary_samples", parameters));
    
    bd = std::stof(catch_parameter("abc_damping", parameters));

    rbc_ratio = std::stof(catch_parameter("rbc_ratio", parameters));
    rbc_varVp = std::stof(catch_parameter("rbc_varVp", parameters));

    fmax = std::stof(catch_parameter("max_frequency", parameters));

    data_folder = catch_parameter("modeling_output_folder", parameters);

    ABC = true;

    nPoints = nx*nz;
    nxx = nx + 2*nb;
    nzz = nz + 2*nb;

    matsize = nxx*nzz;

    nThreads = 256;
    nBlocks = (int)((matsize + nThreads - 1) / nThreads);
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
    
    rIdx = new int[geometry->spread]();
    rIdz = new int[geometry->spread]();
    
    cudaMalloc((void**)&(d_skw), KW*sizeof(float));

    cudaMalloc((void**)&(d_kw0), geometry->spread*sizeof(float));
    cudaMalloc((void**)&(d_kw1), geometry->spread*sizeof(float));
    cudaMalloc((void**)&(d_kw2), geometry->spread*sizeof(float));
    cudaMalloc((void**)&(d_kw3), geometry->spread*sizeof(float));

    cudaMalloc((void**)&(d_rIdx), geometry->spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), geometry->spread*sizeof(int));
}

void Modeling::set_coordinates()
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

void Modeling::set_seismograms()
{
    sBlocks = (int)((geometry->spread + nThreads - 1) / nThreads); 
    seismogram = new float[nt*geometry->spread]();
    seismic_data = new float[nt*geometry->nTraces]();

    cudaMalloc((void**)&(d_seismogram), nt*geometry->spread*sizeof(float));
}

void Modeling::set_cerjan_dampers()
{
    float * damp1D = new float[nb]();
    float * damp2D = new float[nb*nb]();

    for (int i = 0; i < nb; i++) 
    {
        damp1D[i] = expf(-powf(bd * (nb - i), 2.0f));
    }

    for(int i = 0; i < nb; i++) 
    {
        for (int j = 0; j < nb; j++)
        {   
            damp2D[j + i*nb] += damp1D[i];
            damp2D[i + j*nb] += damp1D[i];
        }
    }

    for (int index = 0; index < nb*nb; index++)
        damp2D[index] -= 1.0f;

    cudaMalloc((void**)&(d_b1d), nb*sizeof(float));
    cudaMalloc((void**)&(d_b2d), nb*nb*sizeof(float));

    cudaMemcpy(d_b1d, damp1D, nb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2d, damp2D, nb*nb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] damp1D;
    delete[] damp2D;
}

void Modeling::set_properties()
{
    std::string vp_file = catch_parameter("model_file", parameters);

    Vp = new float[matsize]();

    float * vp = new float[nPoints]();

    cudaMalloc((void**)&(d_P), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vp), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Pold), matsize*sizeof(float));

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
    for (int i = 0; i < nz; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            output[(i + nb) + (j + nb)*nzz] = input[i + j*nz];
        }
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
        int x = (int) (index / nz);    
        int z = (int) (index % nz);  

        output[z + x*nz] = input[(z + nb) + (x + nb)*nzz];
    }
}

void Modeling::show_information()
{
    auto clear = system("clear");

    std::string title = "\033[34mSeismic Modeling\033[0;0m";

    int width = 80;
    int padding = (width - title.length() + 8) / 2;

    std::string line(width, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << ", x = " << (nx - 1)*dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n";
}

void Modeling::initialization()
{
    sIdx = (int)(geometry->xsrc[geometry->sInd[srcId]] / dx) + nb;
    sIdz = (int)(geometry->zsrc[geometry->sInd[srcId]] / dz) + nb;

    Point p, p00, p10, p01, p11;

    p = {geometry->xsrc[geometry->sInd[srcId]], 
         geometry->xsrc[geometry->sInd[srcId]]};
    
    p00 = {floorf(p.x/dx)*dx, floorf(p.z/dz)*dz};
    p10 = {floorf(p.x/dx)*dx + dx, floorf(p.z/dz)*dz};
    p01 = {floorf(p.x/dx)*dx, floorf(p.z/dz)*dz + dz};
    p11 = {floorf(p.x/dx)*dx + dx, floorf(p.z/dz)*dz + dz};
    
    float * skw = new float[KW];
    auto aux = kaiser(p, p00, p10, p01, p11, KS);
    for (int i = 0; i < KW; i++) skw[i] = aux[i]; 
        
    float * rkw0 = new float[geometry->spread];    
    float * rkw1 = new float[geometry->spread];    
    float * rkw2 = new float[geometry->spread];    
    float * rkw3 = new float[geometry->spread];    

    int spreadId = 0;

    for (int recId = geometry->iRec[srcId]; recId < geometry->fRec[srcId]; recId++)
    {
        rIdx[spreadId] = (int)(geometry->xrec[recId] / dx) + nb;
        rIdz[spreadId] = (int)(geometry->zrec[recId] / dz) + nb;

        p = {geometry->xrec[recId], geometry->zrec[recId]};
       
        p00 = {floorf(p.x/dx)*dx, floorf(p.z/dz)*dz};
        p10 = {floorf(p.x/dx)*dx + dx, floorf(p.z/dz)*dz};
        p01 = {floorf(p.x/dx)*dx, floorf(p.z/dz)*dz + dz};
        p11 = {floorf(p.x/dx)*dx + dx, floorf(p.z/dz)*dz + dz};
        
        auto rkw = kaiser(p, p00, p10, p01, p11, KS);

        rkw0[spreadId] = rkw[0];
        rkw1[spreadId] = rkw[1];
        rkw2[spreadId] = rkw[2];
        rkw3[spreadId] = rkw[3];

        ++spreadId;
    }

    cudaMemcpy(d_skw, skw, KW*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_kw0, rkw0, geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kw1, rkw1, geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kw2, rkw2, geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kw3, rkw3, geometry->spread*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rIdx, rIdx, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, rIdz, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);

    delete[] skw;
    delete[] rkw0;
    delete[] rkw1;
    delete[] rkw2;
    delete[] rkw3;
}

void Modeling::set_seismogram()
{
    cudaMemcpy(seismogram, d_seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyDeviceToHost);

    for (int timeId = 0; timeId < nt; timeId++)
        for (int spreadId = 0; spreadId < geometry->spread; spreadId++)
            seismic_data[timeId + spreadId*nt + srcId*geometry->spread*nt] = seismogram[timeId + spreadId*nt];    
}

void Modeling::export_output_data()
{
    std::string data_file = data_folder + "seismogram_nt" + std::to_string(nt) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(int(1e6f*dt)) + "us.bin";
    export_binary_float(data_file, seismic_data, nt*geometry->nTraces);    
}

void Modeling::forward_solver()
{
    cudaMemset(d_P, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Pold, 0.0f, matsize*sizeof(float));

    for (int tId = 0; tId < tlag + nt; tId++)
    {
        compute_pressure<<<nBlocks, nThreads>>>(d_Vp, d_P, d_Pold, d_wavelet, d_b1d, d_b2d, d_skw, sIdx, sIdz, tId, nt, nb, nxx, nzz, dx, dz, dt, ABC);
        
        compute_seismogram<<<sBlocks, nThreads>>>(d_P, d_rIdx, d_rIdz, d_seismogram, d_kw0, d_kw1, d_kw2, d_kw3, geometry->spread, tId, tlag, nt, nzz);     

        std::swap(d_P, d_Pold);
    }
}

__global__ void compute_pressure(float * Vp, float * P, float * Pold, float * d_wavelet, float * d_b1d, float * d_b2d, float * kw, int sIdx, int sIdz, int tId, int nt, int nb, int nxx, int nzz, float dx, float dz, float dt, bool ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
    {
        P[sIdz + sIdx*nzz] += kw[0]*d_wavelet[tId] / (dx*dz); 
        P[sIdz + (sIdx+1)*nzz] += kw[1]*d_wavelet[tId] / (dx*dz); 
        P[(sIdz+1) + sIdx*nzz] += kw[2]*d_wavelet[tId] / (dx*dz); 
        P[(sIdz+1) + (sIdx+1)*nzz] += kw[3]*d_wavelet[tId] / (dx*dz); 
    }

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2P_dx2 = (- 9.0f*(P[i + (j-4)*nzz] + P[i + (j+4)*nzz])
                     +   128.0f*(P[i + (j-3)*nzz] + P[i + (j+3)*nzz])
                     -  1008.0f*(P[i + (j-2)*nzz] + P[i + (j+2)*nzz])
                     +  8064.0f*(P[i + (j+1)*nzz] + P[i + (j-1)*nzz])
                     - 14350.0f*(P[i + j*nzz]))/(5040.0f*dx*dx);

        float d2P_dz2 = (- 9.0f*(P[(i-4) + j*nzz] + P[(i+4) + j*nzz])
                     +   128.0f*(P[(i-3) + j*nzz] + P[(i+3) + j*nzz])
                     -  1008.0f*(P[(i-2) + j*nzz] + P[(i+2) + j*nzz])
                     +  8064.0f*(P[(i-1) + j*nzz] + P[(i+1) + j*nzz])
                     - 14350.0f*(P[i + j*nzz]))/(5040.0f*dz*dz);

        Pold[index] = dt*dt*Vp[index]*Vp[index]*(d2P_dx2 + d2P_dz2) + 2.0f*P[index] - Pold[index];
        
        if (ABC)
        {
            float damper = get_boundary_damper(d_b1d, d_b2d, i, j, nxx, nzz, nb);

            P[index] *= damper;
            Pold[index] *= damper;
        }
    }
}

__global__ void compute_seismogram(float * P, int * d_rIdx, int * d_rIdz, float * seismogram, float * kw0, float * kw1, float * kw2, float * kw3, int spread, int tId, int tlag, int nt, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId >= tlag))
    {
        seismogram[(tId - tlag) + index * nt] = kw0[index]*P[d_rIdz[index] + d_rIdx[index]*nzz] +        
                                                kw1[index]*P[d_rIdz[index] + (d_rIdx[index]+1)*nzz] +
                                                kw2[index]*P[(d_rIdz[index]+1) + d_rIdx[index]*nzz] +
                                                kw3[index]*P[(d_rIdz[index]+1) + (d_rIdx[index]+1)*nzz];
    }
}

__device__ float get_boundary_damper(float * d_b1d, float * d_b2d, int i, int j, int nxx, int nzz, int nb)
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
        damper = d_b1d[i];
    }         
    else if ((i >= nzz - nb) && (i < nzz) && (j >= nb) && (j < nxx - nb)) 
    {
        damper = d_b1d[nb - (i - (nzz - nb)) - 1];
    }         
    else if ((i >= nb) && (i < nzz - nb) && (j >= 0) && (j < nb)) 
    {
        damper = d_b1d[j];
    }
    else if ((i >= nb) && (i < nzz - nb) && (j >= nxx - nb) && (j < nxx)) 
    {
        damper = d_b1d[nb - (j - (nxx - nb)) - 1];
    }

    // 2D damping 
    else if ((i >= 0) && (i < nb) && (j >= 0) && (j < nb))
    {
        damper = d_b2d[i + j*nb];
    }
    else if ((i >= nzz - nb) && (i < nzz) && (j >= 0) && (j < nb))
    {
        damper = d_b2d[nb - (i - (nzz - nb)) - 1 + j*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx - nb) && (j < nxx))
    {
        damper = d_b2d[i + (nb - (j - (nxx - nb)) - 1)*nb];
    }
    else if((i >= nzz - nb) && (i < nzz) && (j >= nxx - nb) && (j < nxx))
    {
        damper = d_b2d[nb - (i - (nzz - nb)) - 1 + (nb - (j - (nxx - nb)) - 1)*nb];
    }

    return damper;
}

std::random_device RBC;  
std::mt19937 RBC_RNG(RBC()); 

void Modeling::set_random_boundary()
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

        int xId = (int)(xc / dx);
        int zId = (int)(zc / dz);

        float r = std::uniform_real_distribution<float>(0.5f*rbc_ratio, rbc_ratio)(RBC_RNG);
        float A = std::uniform_real_distribution<float>(0.5f*rbc_varVp, rbc_varVp)(RBC_RNG);

        float factor = rand() % 2 == 0 ? -1.0f : 1.0f;

        random_boundary_gp<<<nBlocks,nThreads>>>(d_Vp, d_X, d_Z, nxx, nzz, x_max, z_max, xb, zb, factor, A, xc, zc, r, vmax, vmin, rbc_varVp);
    }
}

__device__ float get_random_value(float velocity, float function, float parameter, int index, float varVp)
{
    curandState state;
    curand_init(clock64(), index, 0, &state);
    return velocity + function*parameter*curand_normal(&state);
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

__global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, float xb, float zb, float factor, float A, float xc, float zc, float r, float vmax, float vmin, float varVp)
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
