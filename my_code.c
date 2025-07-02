#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define pi 3.14159265358979323846264338327
#define N_BCC_MODES 12

// contains stuff related to output
struct Output {
	char name[128];		// run name
	int T_print;		// interval for printing output
	int T_write;		// interval for saving state
};

// contains stuff related to data arrays
struct Arrays {
	int W;				// system width
	int H;				// system height
	int Z;				// system in z axis

	ptrdiff_t lH;		// local system height
	ptrdiff_t lW;		// local system width
	ptrdiff_t lZ;		// local system in z axis

	ptrdiff_t lh0;		// local vertical start index
	ptrdiff_t lw0;		// local horizontal start index
	ptrdiff_t lz0;		// local z axis start index

	double *A;	// operator for linear part, e^{-k^2 \hat{\mathcal{L}} \Delta t}
	double *B;	// operator for nonlinear part ...
	double *p;	// array for \psi_C
	double *q;	// another array
	
	fftw_plan p_P;	// FFTW plan for F(p)
	fftw_plan q_Q;	// F(q)
	fftw_plan Q_q;	// F^-1(Q)
};

// contains model parameters
struct Model {
	double Bl;
	double Bs;
	double v;
	double Me;
};

// contains stuff related to relaxation
struct Relaxation {
	time_t t0;	// start time
	int id;		// rank of process
	int ID;		// number of processes
	int t;		// time step count
	double f;	// average free energy density
	double p;	// average density
	double d;	// sampling step size for calculation box optimization
	int T;		// total number of iterations
	double dx;	// x-discretization
	double dy;	// y-
        double dz; 	// z-
	double dt;	// time step
	int T_optimize;		// interval for calculation box optimization
};

// seeds processes' random number generators
void seed_rngs(struct Relaxation *relaxation, FILE *input) {
	int seed;								
// seed for next process
	// seed for 0th process from input file
	fscanf(input, " %d", &seed);			// seed for 0th process from input file
	if(relaxation->id == 0) {
		if(seed == 0) srand(time(NULL)); // random seed
		else srand(seed);	         // user-specified seed
		seed = rand();			 // sample new seed for next process
	}
	else {
		MPI_Recv(&seed, 1, MPI_INT, relaxation->id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// receive from previous
		srand(seed);																			
// seed
		seed = rand();																			
// new seed for next
	}
	if(relaxation->id != relaxation->ID-1) MPI_Send(&seed, 1, MPI_INT, 
relaxation->id+1, 0, MPI_COMM_WORLD);	// send to next
}

void configure_arrays(struct Arrays *arrays, FILE *input) {
    if (fscanf(input, " %d %d %d", &arrays->W, &arrays->H, &arrays->Z) != 3) {
        printf("Invalid input for array dimensions!\n");
        exit(1);
    }

    // FFTW expects dimensions as (n0, n1, n2) -> usually (Z, H, W)
    ptrdiff_t lWHh = fftw_mpi_local_size_3d_transposed(
        arrays->Z, arrays->H, arrays->W / 2 + 1,  // Z, H, W/2+1 for R2C transposed
        MPI_COMM_WORLD,
        &arrays->lZ, &arrays->lz0,
        &arrays->lH, &arrays->lh0);

    ptrdiff_t lWHp = 2 * lWHh;  // padded real array size

    // Allocate memory
    arrays->A = (double *)fftw_malloc(lWHh * sizeof(double));
    arrays->B = (double *)fftw_malloc(lWHh * sizeof(double));

    arrays->p = (double *)fftw_malloc(lWHp * sizeof(double));
    arrays->q = (double *)fftw_malloc(lWHp * sizeof(double));

    // Set up FFTW plans: dimensions passed as (Z, H, W)
    arrays->p_P = fftw_mpi_plan_dft_r2c_3d(
        arrays->Z, arrays->H, arrays->W,
        arrays->p, (fftw_complex *)arrays->p,
        MPI_COMM_WORLD,
        FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);

    arrays->q_Q = fftw_mpi_plan_dft_r2c_3d(
        arrays->Z, arrays->H, arrays->W,
        arrays->q, (fftw_complex *)arrays->q,
        MPI_COMM_WORLD,
        FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);

    arrays->Q_q = fftw_mpi_plan_dft_c2r_3d(
        arrays->Z, arrays->H, arrays->W,
        (fftw_complex *)arrays->q, arrays->q,
        MPI_COMM_WORLD,
        FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
}

// configures output
void configure_output(struct Output *output, FILE *input) {
	// read intervals for printing and saving from input file
	if(fscanf(input, " %d %d", &output->T_print, &output->T_write) != 
2) {
		printf("Invalid input for output configuration!\n");
		exit(1);
	}
}

// configures model
void configure_model(struct Model *model, FILE *input) {
	// read model parameters from input file
	if(fscanf(input, " %lf %lf %lf %lf", &model->Bl, &model->Bs, &model->v, &model->Me) != 4) {
		printf("Invalid input for model parameters!\n");
		exit(1);
	}
}

// configures relaxation
void configure_relaxation(struct Relaxation *relaxation, FILE *input) {
	// read from input file
	if(fscanf(input, " %d %lf %lf %lf %lf %d", &relaxation->T, 
&relaxation->dx, &relaxation->dy, &relaxation->dz, &relaxation->dt, 
&relaxation->T_optimize) != 6) {
		printf("Invalid input for relaxation settings!\n");
		exit(1);
	}
}

// returns the one mode approximation for 3D
double OMA(double x, double y, double z, double A, double x0, double y0, double z0) {
    double n = 0.0; // Density field at point (x, y, z)
    double a = 1.0;  // For lattice spacing
    double q0 = (2.0 * pi / a) / sqrt(2.0);  // magnitude of reciprocal vector

// Unit vectors of BCC reciprocal directions
    int qx_int[N_BCC_MODES] = {  1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0 };
    int qy_int[N_BCC_MODES] = {  1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1 };
    int qz_int[N_BCC_MODES] = {  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1 };

    for (int i = 0; i < N_BCC_MODES; i++) {
        double qx = q0 * qx_int[i];
        double qy = q0 * qy_int[i];
        double qz = q0 * qz_int[i];
        double dx = x - x0;
        double dy = y - y0;
        double dz = z - z0;
        n += cos(qx * dx + qy * dy + qz * dz);
    }
    return 4*A*n;
}


// initializes the density field with a crystallite in center
// R is radius
void embedded_crystallite(struct Arrays *arrays, double dx, double dy, double dz, double l0, double no, double A, double R) {
int Wp=2*(arrays->W/2+1);
int w, h, z, gw, gh, gz, k, x0, y0, z0;
double R2 = R*R;
double xx,yy,zz,r2;
x0=arrays->W/2;
y0=arrays->H/2;
z0=arrays->Z/2;
for(z=0; z<arrays->lZ; z++){
       gz = arrays->lz0+z;
       zz =(gz-0.5*arrays->lZ)*dz;
       for(h=0; h<arrays->lH; h++){
           gh = arrays->lh0+h;
           yy=(gh-0.5*arrays->lH)*dy;
           for(w=0; w<arrays->lW; w++){
             gw = arrays->lw0+w; 
             xx = (gw-0.5*arrays->lW)*dx;
             k=z*arrays->lH*Wp+h*Wp+w;      
             r2 = xx*xx+yy*yy+zz*zz;
             if(r2<R2){
              arrays->q[k]=no+OMA(xx,yy,zz,A,x0,y0,z0);
}
}
}
}
} 

void initialize_system(struct Arrays *arrays, FILE *input) {
int init;
if(fscanf (input, " %d", &init)!=1){
    printf("Invalid initialization type!\n");
    exit(1);
}
if(init==1){
double dx, dy, dz, l0, no, A, R;
if(fscanf(input, " %lf %lf %lf %lf %lf %lf %lf", &dx, &dy, &dz, &l0, &no, &A, &R)!= 7){
    printf("Invalid initialization type!\n");
    exit(1);
}
embedded_crystallite(arrays, dx, dy, dz, l0, no, A, R);
}
}

// saves state into a file
void write_state(struct Arrays *arrays, struct Relaxation *relaxation, struct Output *output) {
    char filename[128];  // output filename
    sprintf(filename, "%s-t:%d.dat", output->name, relaxation->t);
    FILE *file;
    int Wp = 2 * (arrays->W / 2 + 1); // padded width for FFT
    int w, h, z, k, i;

    for (i = 0; i < relaxation->ID; i++) {
        MPI_Barrier(MPI_COMM_WORLD);  // synchronize MPI ranks

        if (relaxation->id == i) {
            // process 0 overwrites, others append
            file = (relaxation->id == 0) ? fopen(filename, "w") : fopen(filename, "a");

            for (z = 0; z < arrays->lZ; z++) {
                for (h = 0; h < arrays->lH; h++) {
                    k = Wp * arrays->lW * z + Wp * h; // start index for (z,h) slice
                    for (w = 0; w < arrays->W; w++, k++) {
                        fprintf(file, "%e\n", arrays->q[k]);
                    }
                }
            }

            fclose(file);
        }
    }
}

// prints output, format:
// time step count, elapsed wall-clock time (s), dx (dimensionless*), dy (*), dz (*), average free energy density (*), average density (*)
void print(struct Relaxation *relaxation, struct Output *output) {
	if(relaxation->id == 0) {			// only 0th process
		printf("%d %d %.9lf %.9lf %.9lf %.9lf %.9lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->dz, relaxation->f, relaxation->p);
		char filename[128];
		sprintf(filename, "%s.out", output->name);
		FILE *file;
		file = fopen(filename, "a");	// need only append, empty file was already generated
		fprintf(file, "%d %d %.9lf %.9lf %.9lf %.9lf %.9lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->dz, relaxation->f, relaxation->p);
		fclose(file);
	}
}

// updates operators for linear and nonlinear parts
void update_AB(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
    int w, h, z, gw, gh, gz, k;
    double kx, ky, kz, k2, k4, k6;
    double dkx = 2.0 * pi / (relaxation->dx * arrays->W);
    double dky = 2.0 * pi / (relaxation->dy * arrays->H);
    double dkz = 2.0 * pi / (relaxation->dz * arrays->Z);
    double d2, l, expl;
    double divWHZ = 1.0 / (arrays->W * arrays->H * arrays->Z);

    //int Wp = arrays->lW;  // local padded W (complex FFT)
    for (z = 0; z < arrays->lZ; z++) {
        gz = arrays->lz0 + z;
        if (gz < arrays->Z / 2)
            kz = gz * dkz;
        else
            kz = (gz - arrays->Z) * dkz;

        for (h = 0; h < arrays->lH; h++) {
            gh = arrays->lh0 + h;
            if (gh < arrays->H / 2)
                ky = gh * dky;
            else
                ky = (gh - arrays->H) * dky;

            for (w = 0; w < arrays->lW; w++) {
                gw = arrays->lw0 + w;
                kx = gw * dkx;

                // Compute k*k, k*k*k*k, etc.
                k2 = -4.0 * pi * pi * (kx * kx + ky * ky + kz * kz);
                k4 = k2 * k2;
                k6 = k4 * k2;

                d2 = k2 - 2.0;
                d2 *= d2;

                // Compute linear operator L_hat in spectral space
                l = model->Bl - model->Bs * 2.0 * d2;
                expl = exp(-k2 * l * relaxation->dt);

                k = (z * arrays->lH + h) * arrays->lW + w;

                arrays->A[k] = expl;

                if (l == 0.0)
                    arrays->B[k] = -k2 * relaxation->dt;  // Avoid division by zero
                else
                    arrays->B[k] = (expl - 1.0) / l;

                arrays->B[k] *= divWHZ;  // Normalization
            }
        }
    }
}
// scales p arrays (in k-space) by 1/(W*H) ((I)FFTs cause scaling of data by sqrt(W*H*Z))
void scale_P(struct Arrays *arrays) {
	fftw_complex *P;			// complex data pointers
	P = (fftw_complex*)&arrays->p[0];
	double divWHZ = 1.0/(arrays->W*arrays->H* arrays->Z);
	int i;
	int lA = arrays->lW*arrays->H* arrays->Z;
	for(i = 0; i < lA; i++) P[i] *= divWHZ;
}

// scales q arrays (in k-space) by 1/(W*H*Z)
void scale_Q(struct Arrays *arrays) {
	fftw_complex *Q;
	Q = (fftw_complex*)&arrays->q[0];
	double divWHZ = 1.0/(arrays->W*arrays->H* arrays->Z);
	int i;
	int lA = arrays->lW*arrays->H* arrays->Z;
	for(i = 0; i < lA; i++) Q[i] *= divWHZ;
}

// computes average free energy density (f) and average density (p) (f & p -> fp)
// Fix from here
void fp(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int Wp = 2 * (arrays->W / 2 + 1);  // real padded width
	int lZ = arrays->lZ; int lH = arrays->lH; int lW = arrays->lW;

	double dkx = 2.0 * pi / (relaxation->dx * arrays->W);
	double dky = 2.0 * pi / (relaxation->dy * arrays->H);
	double dkz = 2.0 * pi / (relaxation->dz * arrays->Z);

	// Copy local region of q into p
	memcpy(arrays->p, arrays->q, lZ * lH * Wp * sizeof(double));

	fftw_execute(arrays->q_Q);   // forward FFT
	scale_Q(arrays);            // scale spectral coefficients

	// Access complex FFT of q
	fftw_complex *Q = (fftw_complex *)arrays->q;
	// Apply operator in Fourier space
	for (int z = 0; z < lZ; z++) {
		int gz = arrays->lz0 + z;
		double kz = (gz < arrays->Z / 2) ? gz * dkz : (gz - arrays->Z) * dkz;
		double kz2 = kz * kz;

		for (int h = 0; h < lH; h++) {
			int gh = arrays->lh0 + h;
			double ky = (gh < arrays->H / 2) ? gh * dky : (gh - arrays->H) * dky;
			double ky2 = ky * ky;

			for (int w = 0; w < (lW / 2 + 1); w++) {
				double kx = w * dkx;
				double kx2 = kx * kx;

				double k2 = -4.0 * pi * pi * (kx2 + ky2 + kz2);
				double d2 = (k2 - 2.0) * (k2 - 2.0);

				int izx = (z * lH + h) * (arrays->W / 2 + 1) + w;

				Q[2*izx+0] *= d2;
				Q[2*izx+1] *= d2;
			}
		}
	}

	fftw_execute(arrays->Q_q);  // inverse FFT: back to real space

	relaxation->f = 0.0;
	relaxation->p = 0.0;

	for (int z = 0; z < lZ; z++) {
		int gz = arrays->lz0 + z;
		double kz = (gz < arrays->Z / 2) ? gz * dkz : (gz - arrays->Z) * dkz;
		double kz2 = kz * kz;

		for (int h = 0; h < lH; h++) {
			int gh = arrays->lh0 + h;
			double ky = (gh < arrays->H / 2) ? gh * dky : (gh - arrays->H) * dky;
			double ky2 = ky * ky;

			for (int w = 0; w < arrays->W; w++) {
				double kx = w * dkx;
				double kx2 = kx * kx;

				double k2 = -4.0 * pi * pi * (kx2 + ky2 + kz2);
				double k4 = k2 * k2;

				int izx = (z * lH + h) * Wp + w;

				double p = arrays->p[izx];
			//	double q = arrays->q[izx];
			
				double p2 = p*p;	
				double p4 = p2*p2;

				// Free energy density: match your model's form
				relaxation->f += 0.5 * (model->Bl + model->Bs * (2 * k2 + k4)) * p2 - 
                                                 (model->v / 6.0) * p2 * p + 0.25 * p4;

				// Restore original q
				arrays->q[izx] = p;

				// Average density
				relaxation->p += p;
			}
		}
	}

	// MPI reduction
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &relaxation->p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	double divVol = 1.0 / (arrays->W * arrays->H * arrays->Z);
	relaxation->f *= divVol;
	relaxation->p *= divVol;

	fftw_execute(arrays->p_P);  // FFT: back to spectral space
	scale_P(arrays);            // scale back
}


//void init_operators(struct Arrays *arrays, struct Relaxation *relaxation, struct Model *model, double dt) {
//	int Wp = 2 * (arrays-> W / 2 + 1);
//	int size = arrays->Z * arrays-> H * (arrays->W / 2 + 1);
//
//	double C1 = fabs(model->Me* model->Bl);
//	double C2 = fabs(2*model->Me* model->Bs);
//	double C3 = fabs(model->Me*model->Bs);

//	for (int z = 0; z < arrays->Z; z++) {
//		double kz = (z <= arrays-> Z / 2) ? z : z - arrays-> Z;
//		for (int y = 0; y < arrays->H; y++) {
//			double ky = (y <= arrays->H / 2) ? y : y - arrays-> H;
//			for (int x = 0; x <= arrays->W / 2; x++) {
//				double kx = x;
//				double k2 = kx*kx + ky*ky + kz*kz;
//				double k4 = k2 * k2;
//				double k6 = k4 * k2;

//				int izx = (z * arrays->H + y) * (arrays->W / 2 + 1) + x;
//				double Lk = -C1 * k2 + C2 * k4 - C3 * k6;

//				arrays->A[izx] = 1.0 / (1.0 - dt * Lk);
//				arrays->B[izx] = dt * arrays->A[izx];
//			}
//		}
//	}
// }

// performs one iteration of the semi-implicit spectral method in 3D with operator splitting
void step(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	int W = arrays->W;
	int H = arrays->H;
	int Z = arrays->Z;
	int Wp = 2 * (W / 2 + 1);
	int slice_size = H * Wp;

	double C1 = fabs(model->Me*model->Bl);
	double C2 = fabs(2*model->Me*model->Bs);
	double C3 = fabs(model->Me*model->Bs);

	if (relaxation->t % 100 == 0) {
		memcpy(arrays->p, arrays->q, slice_size * arrays->lZ * sizeof(double));
		fftw_execute(arrays->p_P);  // real -> spectral
		scale_P(arrays);
	}

	double q, q2, q4;

	fftw_complex *P = (fftw_complex *)&arrays->p[0];  // Spectral of real state
	fftw_complex *Q = (fftw_complex *)&arrays->q[0];  // Spectral of nonlinear term

	// Spectral update: P = A*P + B*Q, then copy to Q
	// === A1: Build nonlinear term in real space ===
	for (int z = 0; z < Z; z++) {
		for (int y = 0; y < H; y++) {
			int izx = z * slice_size + y * Wp;
			for (int x = 0; x < W; x++) {
				q = arrays->q[izx + x];
				q2 = q * q;
				q4 = q2 * q2;
				arrays->q[izx + x] =(model->Me*model->Bl-C1)*q+
					model->Me*(-0.5*model->v*q2+(1.0/3.0)*q*q2)+
                                  q*(model->Me*model->Bs*2-C2*0.5)+q*(model->Me*model->Bs-C3*0.5);
			}
		}
	}

	fftw_execute(arrays->q_Q);  // real -> Q

	// === Spectral update: P = A*P + B*Q ===
//	for (z = 0; z < Z; z++) {
//		for (y = 0; y < H; y++) {
//			for (x = 0; x <= W / 2; x++) {
//				izx = (z * H + y) * (W / 2 + 1) + x;
//				P[izx][0] = arrays->A[izx] * P[izx][0] + arrays->B[izx] * Q[izx][0];
//				P[izx][1] = arrays->A[izx] * P[izx][1] + arrays->B[izx] * Q[izx][1];
//				Q[izx][0] = P[izx][0];  // Copy to Q for IFFT
//				Q[izx][1] = P[izx][1];
//			}
//		}
//	}

	for (int z = 0; z < arrays->Z; z++) {
		double kz = (z <= arrays-> Z / 2) ? z : z - arrays-> Z;
		for (int y = 0; y < arrays->H; y++) {
			double ky = (y <= arrays->H / 2) ? y : y - arrays-> H;
			for (int x = 0; x <= arrays->W / 2; x++) {
				double kx = x;
				double k2 = kx*kx + ky*ky + kz*kz;
				double k4 = k2 * k2;
				double k6 = k4 * k2;

				int izx = (z * arrays->H + y) * (arrays->W / 2 + 1) + x;
				P[izx] = C1*k2*Q[izx]+C2*k4*Q[izx]+C3*k6*Q[izx];

				//copy P to Q
				Q[izx] = P[izx];


//				arrays->A[izx] = 1.0 / (1.0 - relaxation->dt * Lk);
//				arrays->B[izx] = relaxation->dt * arrays->A[izx];
			}
		}
	}


	fftw_execute(arrays->Q_q);  // Q -> real q
}


// optimizes calculation box size for system
// samples slightly different box sizes by varying dx, dy and dz and interpolates optimum quadratically
void optimize(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
	// sample sizes
	double dxs[] = {relaxation->dx, relaxation->dx-relaxation->d, relaxation->dx+relaxation->d, relaxation->dx, relaxation->dx};
	double dys[] = {relaxation->dy, relaxation->dy, relaxation->dy, relaxation->dy-relaxation->d, relaxation->dy+relaxation->d};
	double dzs[] = {relaxation->dz, relaxation->dz, relaxation->dz, relaxation->dz-relaxation->d, relaxation->dz+relaxation->d};
	double fs[7];	// average free energy densities of sizes sampled
	for(int i = 0; i < 7; i++) {		// sample sizes
		relaxation->dx = dxs[i];
		relaxation->dy = dys[i];
		relaxation->dz = dzs[i];
		fp(arrays, model, relaxation);	// compute energy (and density (not needed here though))
		fs[i] = relaxation->f;	// save average free energy density
	}
	// interpolate new dx and dy (minimum of surface f = a+b*dx+c*dy+d*dx^2+e*dy^2)
relaxation->dx = (relaxation->d * (fs[1] - fs[2]) + 2.0 * dxs[0] * (-2.0 * fs[0] + fs[1] + fs[2])) /
                     (2.0 * (fs[1] - 2.0 * fs[0] + fs[2]));

    relaxation->dy = (relaxation->d * (fs[3] - fs[4]) + 2.0 * dys[0] * (-2.0 * fs[0] + fs[3] + fs[4])) /(2.0 * (fs[3] - 2.0 * fs[0] + fs[4]));

    relaxation->dz = (relaxation->d * (fs[5] - fs[6]) + 2.0 * dzs[0] * (-2.0 * fs[0] + fs[5] + fs[6])) /(2.0 * (fs[5] - 2.0 * fs[0] + fs[6]));

	// check that change in discretization is acceptable
	double l0 = 4.0*pi/sqrt(3.0);	// approximate dimensionless length scale (lattice constant)
	double dw = arrays->W*(relaxation->dx-dxs[0]);		// change in horizontal system size
	double dh = arrays->H*(relaxation->dy-dys[0]);		// ... vertical ...
	double dz = arrays->Z*(relaxation->dz-dzs[0]);		// ... vertical ...
	double dr = sqrt(dw*dw+dh*dh+dz*dz);					// "change vector"
	double x = 0.25*l0;	          	// limit to 1/4 of lattice constant (to ensure stability)
	if(dr > x) {	// if the change in system dimensions exceeds 1/4 of the lattice constant ...
		x /= dr;
		relaxation->dx = x*relaxation->dx+(1.0-x)*dxs[0];	// ... truncate the change to 1/4 of the lattice constant
		relaxation->dy = x*relaxation->dy+(1.0-x)*dys[0];
		relaxation->dz = x*relaxation->dz+(1.0-x)*dzs[0];

	}

	// update A and B
	update_AB(arrays, model, relaxation);	// dx, dy and dz changed -> need to update operators
	// update sampling step size (tries to keep it in the same ballpark with dr (for, hopefully, more accurate optimization))
	double ddx = relaxation->dx-dxs[0];
	double ddy = relaxation->dy-dys[0];
	double ddz = relaxation->dz-dzs[0];

	double ddr = sqrt(ddx*ddx+ddy*ddy+ddz*ddz);	// discretization change
	if(ddr < relaxation->d) relaxation->d *= 0.5;		// if change vector < d, halve d
	else relaxation->d *= 2.0;				// otherwise double
	if(relaxation->d < 1.0e-6) relaxation->d *= 2.0;	// can cause numerical issues if d gets too small
}

// relaxes the system for T time steps
void relax(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation, struct Output *output) {
	for(relaxation->t = 0; relaxation->t <= relaxation->T; relaxation->t++) {
		if(relaxation->T_optimize > 0 && relaxation->t > 0 && relaxation->t%relaxation->T_optimize == 0) optimize(arrays, model, relaxation);	// optimize?
		if(relaxation->t%output->T_print == 0) {	// print output?
			fp(arrays, model, relaxation);
			print(relaxation, output);
		}
		if(relaxation->t%output->T_write == 0) write_state(arrays, relaxation, output);		// write out state?
		if(relaxation->t < relaxation->T) step(arrays, model, relaxation);	// perform iteration step
	}
}

// frees allocated arrays
void clear_arrays(struct Arrays *arrays) {
	fftw_free(arrays->p);
	fftw_free(arrays->q);
	fftw_free(arrays->A);
	fftw_free(arrays->B);
}

int main(int argc, char **argv) {
	// init MPI
	MPI_Init(&argc, &argv);
	fftw_mpi_init();
	// create structs
	struct Arrays arrays;
	struct Model model;
	struct Relaxation relaxation;
	struct Output output;
	relaxation.t0 = time(NULL);
	relaxation.t = 0;
	relaxation.d = 0.0001;
	MPI_Comm_rank(MPI_COMM_WORLD, &relaxation.id);
	MPI_Comm_size(MPI_COMM_WORLD, &relaxation.ID);
	strcpy(output.name, argv[1]);
	// input stream
	char filename[128];
	sprintf(filename, "%s.in", output.name);
	FILE *input = fopen(filename, "r");
	if(input == NULL) {
		printf("Input file not found!\n");
		return 2;
	}
	// create empty output file
	sprintf(filename, "%s.out", output.name);
	FILE *out = fopen(filename, "w");
	fclose(out);
	// read input
	char label;
	char line[1024];
	while(fscanf(input, " %c", &label) != EOF) {
		if(label == '#' && fgets(line, 1024, input)) {}			// comment
		else if(label == '\n') {}								// empty line
		else if(label == 'S') {									// seed random number generators
			seed_rngs(&relaxation, input);
		}
		else if(label == 'O') {									// output
			configure_output(&output, input);
		}
		else if(label == 'A') {			// set up arrays and FFTW plans
			configure_arrays(&arrays, input);
		}
        	else if(label == 'I') {                 // initialization
			initialize_system(&arrays, input);
		}
		else if(label == 'M') {									// model
			configure_model(&model, input);
		}
		else if(label == 'R') {									// relaxation
			configure_relaxation(&relaxation, input);
			update_AB(&arrays, &model, &relaxation);
			relax(&arrays, &model, &relaxation, &output);
		}
		else {
			printf("Invalid label: %c!\n", label);				// bad input
			return 1;
		}
	}
	fclose(input);
	
	clear_arrays(&arrays);										// clean-up
	MPI_Finalize();

	return 0;

}
