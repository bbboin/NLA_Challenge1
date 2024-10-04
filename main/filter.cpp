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
  //Error if forget image path
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <image_path>" << endl;
    return 1;
  }

  //assign image path
  const char* input_image_path = argv[1];
                                                                                //1
  // Load the image using stb_image
  int width, height, channels;
  // Force load as B&W
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);

  //error loading image
  if (!image_data) {
    cerr << "Error: Could not load image " << input_image_path
              << endl;
    return 1;
  }
  
  //print image parameters
  cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

  // Prepare Eigen matrix for B&W channel
  MatrixXd original(height, width);

  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * 1;  // 1 channels (B&W)
      original(i, j) = static_cast<double>(image_data[index])/255.0;
    }
  }

  // Free memory
  stbi_image_free(image_data);

                                                                                //2
  //generate noise
  MatrixXd noise = MatrixXd::Random(height, width)*50.0/255.0;

  //add noise
  MatrixXd noisy = original + noise;

  noisy = limit01(noisy, height, width);
  //export
  save_image(noisy, height, width, "output_noisy.png");
                                                                                      //3
  //reshape as vectors
  VectorXd v(Map<VectorXd>(original.data(), original.cols()*original.rows()));
  VectorXd w(Map<VectorXd>(noisy.data(), noisy.cols()*noisy.rows()));
  
  //print sizes and norms of vectors
  cout << "v size = " << v.rows() << endl;
  cout << "w size = " << w.rows() << endl;
  
  cout << "v norm = " << v.norm() << endl;
  cout << "w norm = " << w.norm() << endl;

  
                                                                                      //4
  //Construct A1 matrix for Hav2 smoothing kernel
  constexpr array<array<double,3>,3> m1={1./9.,1./9.,1./9.,
                                         1./9.,1./9.,1./9.,
                                         1./9.,1./9.,1./9.};

  SparseMatrix<double,RowMajor> A1(height*width,height*width);
  A1 = ker2mat(m1, height, width);
 
  //print number of non-zeros entries
  cout << "Non-zeros entries in A1: " << A1.nonZeros() << endl;
    
                                                                                  //5
  //multiply A1*w
  VectorXd A1w(height*width);
  A1w=A1*w;
  //reshape
  MatrixXd smooth = Map<MatrixXd>(A1w.data(), height, width);
  //export
  save_image(smooth, height, width, "output_smooth.png");

                                                                                  //6
  //construct A2 for Hsh2 sharpening kernel
  constexpr array<array<double,3>,3> m2={0.,-3.,0.,
                                        -1.,9.,-3.,
                                         0.,-1.,0.};
  
  SparseMatrix<double,RowMajor> A2(height*width,height*width);
  A2 = ker2mat(m2, height, width);
  
  //print number of non-zeros entries
  cout << "Non-zeros entries in A2: " << A2.nonZeros() << endl;

  //simmetry check
  cout << "Is A2 symmetric? : ";
  SparseMatrix<double,RowMajor> B = A2 - SparseMatrix<double,RowMajor>(A2.transpose());
  if (B.norm()==0) cout<< "Yes" << endl;
  else cout <<"No" << endl;
                                                                                    //7
  //multiply A2*v
  VectorXd A2v(height*width);
  A2v=A2*v;
  //reshape
  MatrixXd sharp = Map<MatrixXd>(A2v.data(), height, width);
  
  sharp = limit01(sharp, height, width);
  //export
  save_image(sharp, height, width, "output_sharp.png");
  
                                                                                    //8
  // Export matrix A2 in .mtx format
  
  saveMarket(A2, "./A2.mtx");
  
  cout << "Saved matrix A2 as A2.mtx" << endl;

  // Export vector w in .mtx format
  
  FILE* out = fopen("w.mtx","w");
  fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
  fprintf(out,"%d\n", height*width);
  for (int i=0; i<height*width; i++) {
    fprintf(out,"%d %f\n", i ,w(i));
  }
  fclose(out);

  cout << "Saved vector w as w.mtx" << endl;
  
                                                                                  //10
  
  //construct A3 for Hlap edge detection kernel
  constexpr array<array<double,3>,3> m3={0.,-1.,0.,
                                        -1.,4.,-1.,
                                         0.,-1.,0.};
  
  SparseMatrix<double,RowMajor> A3(height*width,height*width);
  A3 = ker2mat(m3, height, width);

  //simmetry check
  cout << "Is A3 symmetric? : ";
  SparseMatrix<double,RowMajor> C = A3 - SparseMatrix<double,RowMajor>(A3.transpose());
  if (C.norm()==0) cout<< "Yes" << endl;
  else cout <<"No" << endl;
                                                                                  //11
  //multiply A3*v
  VectorXd A3v(height*width);
  A3v=A3*v;
  //reshape
  MatrixXd edge = Map<MatrixXd>(A3v.data(), height, width);
  
  edge = limit01(edge, height, width);
  //export
  save_image(edge, height, width, "output_edge.png");
  
                                                                                  //12
  //build I+A3
  SparseMatrix<double,RowMajor> A4(height*width,height*width);
  A4.setIdentity();
  A4 = A4 + A3;
  
  // Set parameters for solver
  double tol2 = 1.e-10;                 // Convergence tolerance
  int maxit2 = 1000;           // Maximum iterations

  // Solving 
  ConjugateGradient<SparseMatrix<double,RowMajor>, Lower|Upper, IncompleteCholesky<double, Lower|Upper, NaturalOrdering<int> > > cg;
  cg.setMaxIterations(maxit2);
  cg.setTolerance(tol2);
  cg.compute(A4);
  VectorXd y = cg.solve(w);
  cout << " Eigen native CG" << endl;
  cout << "#iterations:     " << cg.iterations() << endl;
  cout << "relative residual: " << cg.error()      << endl;
  
                                                                                  //13
  //reshape, limit, export
  MatrixXd Y = Map<MatrixXd>(y.data(), height, width);
  
  Y = limit01(Y, height, width);

  save_image(Y, height, width, "output_y.png");

  return 0;
}


