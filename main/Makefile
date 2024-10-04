

main :
	g++ -I ${mkEigenInc} -O3 filter.cpp -o filter
	mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1
	g++ -I ${mkEigenInc} -O3 convertX.cpp -o convertX