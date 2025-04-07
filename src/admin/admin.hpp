# ifndef ADMIN_HPP
# define ADMIN_HPP

# include <cmath>
# include <string>
# include <chrono>
# include <vector>
# include <random>
# include <complex>
# include <fftw3.h>
# include <sstream>
# include <fstream>
# include <iostream>
# include <algorithm>

# define KW 4    // number of kaiser weights
# define KS 5.0f // kaiser weights space: controls smoothness

struct Point 
{
    float x;
    float z;
};

bool str2bool(std::string s);

void import_binary_float(std::string path, float * array, int n);
void export_binary_float(std::string path, float * array, int n);

void import_text_file(std::string path, std::vector<std::string> &elements);

float bessel_i0(float x);

std::vector<float> kaiser(const Point& p, const Point& p00, const Point& p10, 
                          const Point& p01, const Point& p11, float beta); 

float cubic1d(float P[4], float dx);
float cubic2d(float P[4][4], float dx, float dy);

std::string catch_parameter(std::string target, std::string file);

std::vector<std::string> split(std::string s, char delimiter);

std::vector<Point> poissonDiskSampling(float x_max, float z_max, float radius); 

# endif