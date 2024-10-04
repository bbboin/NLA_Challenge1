#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

#include "../include/myl4image.h" //also include stb_image.h and stb_image_write.h

using namespace std;
using namespace Eigen;

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(int argc, char* argv[]) {
  //Error if forget vector path
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <vector.mtx_path>" << endl;
    return 1;
  }

  //load vector x
  
  FILE* in = fopen("x.mtx", "r");
  if (in == NULL) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }

  char line[256];
  int size;
    
  // Read the header lines
  fgets(line, sizeof(line), in); // Skip the first line
  fgets(line, sizeof(line), in); // Read the size line
  sscanf(line, "%d", &size);

  // Allocate memory for the vector
  VectorXd x(size);
  
  if (x.data() == NULL) {
    perror("Error allocating memory");
    fclose(in);
    return EXIT_FAILURE;
  }

  // Read the vector values
  int index;
  float value;
  for (int i = 0; i < size; i++) {
    if (fgets(line, sizeof(line), in) != NULL) {
      sscanf(line, "%d %f", &index, &value);
      x[index-1] = value;
      }
  }

  fclose(in);

  //reshape, limit, export
  int height=341;
  int width=256;

  MatrixXd matX = Map<MatrixXd>(x.data(), height, width);
  
  matX = limit01(matX, height, width);

  save_image(matX, height, width, "output_x.png");




return 0;
}

