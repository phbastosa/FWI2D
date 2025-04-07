# include "admin.hpp"

bool str2bool(std::string s)
{
    bool b;

    std::for_each(s.begin(), s.end(), [](char & c){c = ::tolower(c);});
    std::istringstream(s) >> std::boolalpha >> b;

    return b;
}

void import_binary_float(std::string path, float * array, int n)
{
    std::ifstream file(path, std::ios::in);

    if (!file.is_open())
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");
    
    file.read((char *) array, n * sizeof(float));
    
    file.close();    
}

void export_binary_float(std::string path, float * array, int n)
{
    std::ofstream file(path, std::ios::out);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    file.write((char *) array, n * sizeof(float));

    std::cout<<"\nBinary file \033[34m" + path + "\033[0;0m was successfully written."<<std::endl;

    file.close();
}

void import_text_file(std::string path, std::vector<std::string> &elements)
{
    std::ifstream file(path, std::ios::in);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    std::string line;
    while(getline(file, line))
        if (line[0] != '#') elements.push_back(line);

    file.close();
}

std::string catch_parameter(std::string target, std::string file)
{
    char spaces = ' ';
    char comment = '#';

    std::string line;
    std::string variable;

    std::ifstream parameters(file);

    if (!parameters.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + file + "\033[0;0m could not be opened!");

    while (getline(parameters, line))
    {           
        if ((line.front() != comment) && (line.front() != spaces))        
        {
            if (line.find(target) == 0)
            {
                for (int i = line.find("=")+2; i < line.size(); i++)
                {    
                    if (line[i] == '#') break;
                    variable += line[i];            
                }

                break;
            }
        }                 
    }
    
    parameters.close();

    if (variable.find('"') == 0)
    {
        remove(variable.begin(), variable.end(), '"');
    }
    else if (variable.find("[") == 0)
    {
        remove(variable.begin(), variable.end(), '[');
        remove(variable.begin(), variable.end(), ']');
    }

    variable.erase(remove(variable.begin(), variable.end(), ' '), variable.end());

    return variable;
}

std::vector<std::string> split(std::string s, char delimiter)
{
    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (getline(tokenStream, token, delimiter)) 
        tokens.push_back(token);
   
    return tokens;
}

float bessel_i0(float x) 
{
    float k = 1.0f;
    float sum = 1.0f; 
    float term = 1.0f;

    while (term > 1e-10f) 
    {
        term *= (x / (2.0f * k)) * (x / (2.0f * k));
        sum += term;
        k += 1.0f;
    }

    return sum;
}

std::vector<float> kaiser(const Point& p, const Point& p00, const Point& p10, 
                          const Point& p01, const Point& p11, float beta) 
{
    std::vector<float> w(4);

    std::vector<Point> corners = {p00, p10, p01, p11};

    float dmax = std::sqrt((p10.x - p00.x) * (p10.x - p00.x) + (p01.z - p00.z) * (p01.z - p00.z));
 
    float denom = bessel_i0(beta);
    
    float sum = 0.0;

    for (int i = 0; i < KW; i++) 
    {
        float dx = p.x - corners[i].x;
        float dz = p.z - corners[i].z;
 
        float r = sqrtf(dx*dx + dz*dz);

        float rnorm = std::min(2.0f*r/dmax, 1.0f);
        
        float arg = beta*sqrtf(1.0f - rnorm*rnorm);
        
        w[i] = bessel_i0(arg) / denom;

        sum += w[i];
    }

    for (int i = 0; i < KW; i++) w[i] /= sum;

    return w;
}

float cubic1d(float P[4], float dx)
{
    return P[1] + 0.5f*dx*(P[2] - P[0] + dx*(2.0f*P[0] - 5.0f*P[1] + 4.0f*P[2] - P[3] + dx*(3.0f*(P[1] - P[2]) + P[3] - P[0])));
}

float cubic2d(float P[4][4], float dx, float dy)
{    
    float p[4];
    p[0] = cubic1d(P[0], dy);
    p[1] = cubic1d(P[1], dy);
    p[2] = cubic1d(P[2], dy);
    p[3] = cubic1d(P[3], dy);    
    return cubic1d(p, dx);
}

std::random_device rd;  
std::mt19937 rng(rd()); 

std::vector<Point> poissonDiskSampling(float x_max, float z_max, float radius)
{
    float cell_size = radius / std::sqrt(2.0f);

    int grid_x = static_cast<int>(std::ceil(x_max / cell_size));
    int grid_z = static_cast<int>(std::ceil(z_max / cell_size));

    std::vector<Point> points;
    std::vector<Point> activeList;

    std::vector<Point> grid(grid_x*grid_z, {-1, -1});

    std::uniform_real_distribution<float> distX(0, x_max);
    std::uniform_real_distribution<float> distY(0, z_max);

    auto insertIntoGrid = [&](const Point& p) 
    {
        int gx = static_cast<int>(p.x / cell_size);
        int gz = static_cast<int>(p.z / cell_size);
        
        grid[gz + gx*grid_z] = p;
    };

    auto isValid = [&](const Point& p) 
    {
        if (p.x < 0 || p.z < 0 || p.x >= x_max || p.z >= z_max) return false;
    
        int gx = static_cast<int>(p.x / cell_size);
        int gz = static_cast<int>(p.z / cell_size);

        for (int i = -1; i <= 1; i++) 
        {
            for (int j = -1; j <= 1; j++) 
            {
                int nx = gx + j;
                int nz = gz + i;
                if (nx >= 0 && nz >= 0 && nx < grid_x && nz < grid_z) 
                {
                    Point neighbor = grid[nz + nx*grid_z];
                    if (neighbor.x != -1 && std::hypot(neighbor.x - p.x, neighbor.z - p.z) < radius) 
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    auto generateRandomPointAround = [&](const Point& p) 
    {
        std::uniform_real_distribution<float> distAngle(0, 2 * M_PI);
        std::uniform_real_distribution<float> distRadius(radius, 2 * radius);
        
        float angle = distAngle(rng);
        float r = distRadius(rng);
        
        return Point{p.x + r*std::cos(angle), p.z + r*std::sin(angle)};
    };

    Point initial = {distX(rng), distY(rng)};
    points.push_back(initial);
    activeList.push_back(initial);
    insertIntoGrid(initial);
    
    while (!activeList.empty()) 
    {
        int index = std::uniform_int_distribution<int>(0, activeList.size() - 1)(rng);
        Point point = activeList[index];
        bool found = false;

        for (int i = 0; i < 30; ++i) 
        {
            Point newPoint = generateRandomPointAround(point);
            if (isValid(newPoint)) 
            {
                points.push_back(newPoint);
                activeList.push_back(newPoint);
                insertIntoGrid(newPoint);
                found = true;
            }
        }
        if (!found) activeList.erase(activeList.begin() + index);
    }
    return points;
}
