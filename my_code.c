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
        char name[128];         // run name
        int T_print;            // interval for printing output
        int T_write;            // interval for saving state
};

// contains stuff related to data arrays
struct Arrays {
        int W;                          // system width
        int H;                          // system height
        int Z;                          // system in z axis

        ptrdiff_t lH;           // local system height
        ptrdiff_t lW;           // local system width
        ptrdiff_t lZ;           // local system in z axis

        ptrdiff_t lh0;          // local vertical start index
        ptrdiff_t lw0;          // local horizontal start index
        ptrdiff_t lz0;          // local z axis start index

        double *A;      // operator for linear part, e^{-k^2 \hat{\mathcal{L}} \Delta t}
        double *B;      // operator for nonlinear part ...
        double *p;      // array for n density field
        double *q;      // another array
        double *u_n;    // for smoothed density field
        double u_n_min; // min smoothed density field
        double u_n_max; // max smoothed density field

        fftw_plan p_P;  // FFTW plan for F(p)
        fftw_plan q_Q;  // F(q)
        fftw_plan Q_q;  // F^-1(Q)
        fftw_plan u_U_n; // for smoothed density field
        fftw_plan U_u_n;
};

// contains model parameters
struct Model {
        double Bl;
        double Bs;
        double v;
        double Me;
        double sigma_u;
        double no;
};

// contains stuff related to relaxation
struct Relaxation {
        time_t t0;      // start time
        int id;         // rank of process
        int ID;         // number of processes
        int t;          // time step count
        double f;       // average free energy density
        double p;       // average density
        double d;       // sampling step size for calculation box optimization
        int T;          // total number of iterations
        double dx;      // x-discretization
        double dy;      // y-
        double dz;      // z-
        double dt;      // time step
        int T_optimize;         // interval for calculation box optimization
};

// seeds processes' random number generators
void seed_rngs(struct Relaxation *relaxation, FILE *input) {
        int seed;                                                               
// seed for next process
        // seed for 0th process from input file
        fscanf(input, " %d", &seed);                    // seed for 0th process from input file
        if(relaxation->id == 0) {
                if(seed == 0) srand(time(NULL)); // random seed
                else srand(seed);                // user-specified seed
                seed = rand();                   // sample new seed for next process
        }
        else {
                MPI_Recv(&seed, 1, MPI_INT, relaxation->id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    // receive from previous
                srand(seed);    // seed
                seed = rand(); // new seed for next
        }
        if(relaxation->id != relaxation->ID-1) MPI_Send(&seed, 1, MPI_INT, relaxation->id+1, 0, MPI_COMM_WORLD);        // send to next
}

void configure_arrays(struct Arrays *arrays, FILE *input) {
    if (fscanf(input, " %d %d %d", &arrays->W, &arrays->H, &arrays->Z) !=3) {
        printf("Invalid input for array dimensions!\n");
        exit(1);
    }

    // FFTW expects dimensions as (n0, n1, n2) -> usually (Z, H, W) // Z, H, W/2+1 for R2C transposed
    ptrdiff_t lWHh = fftw_mpi_local_size_3d_transposed(arrays->Z, arrays->H, arrays->W / 2 + 1, MPI_COMM_WORLD,   &arrays->lZ, &arrays->lz0, &arrays->lH, &arrays->lh0);
    ptrdiff_t lWHp = 2 * lWHh;  // padded real array size

    // Allocate memory 

arrays->A = (double *)fftw_malloc(lWHh * sizeof(double));
if (!arrays->A) { fprintf(stderr, "fftw_malloc failed for A\n"); 
MPI_Abort(MPI_COMM_WORLD, 1); }
arrays->B = (double *)fftw_malloc(lWHh * sizeof(double));
if (!arrays->B) { fprintf(stderr, "fftw_malloc failed for B\n"); 
MPI_Abort(MPI_COMM_WORLD, 1); }
arrays->p = (double *)fftw_malloc(lWHp * sizeof(double));
if (!arrays->p) { fprintf(stderr, "fftw_malloc failed for p\n"); 
MPI_Abort(MPI_COMM_WORLD, 1); }
arrays->q = (double *)fftw_malloc(lWHp * sizeof(double));
if (!arrays->q) { fprintf(stderr, "fftw_malloc failed for q\n"); 
MPI_Abort(MPI_COMM_WORLD, 1); }
arrays->u_n = (double *)fftw_malloc(lWHp * sizeof(double));
if (!arrays->u_n) { fprintf(stderr, "fftw_malloc failed for u_n\n"); 
MPI_Abort(MPI_COMM_WORLD, 1); }

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

    arrays->u_U_n = fftw_mpi_plan_dft_r2c_3d(
        arrays->Z, arrays->H, arrays->W,
        arrays->q, (fftw_complex *)arrays->u_n,
        MPI_COMM_WORLD,
        FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);

    arrays->U_u_n = fftw_mpi_plan_dft_c2r_3d(
        arrays->Z, arrays->H, arrays->W,
        (fftw_complex *)arrays->u_n, arrays->q,
        MPI_COMM_WORLD,
        FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
}

// configures output
void configure_output(struct Output *output, FILE *input) {
        fscanf(input, " %d %d", &output->T_print, &output->T_write);    
// read intervals for printing and saving from input file
}

// configures model
void configure_model(struct Model *model, FILE *input) {
        // read model parameters from input file
        if(fscanf(input, " %lf %lf %lf %lf %lf %lf", &model->Bl, &model->Bs, &model->v, &model->Me, &model->no, &model->sigma_u) != 6) {
                printf("Invalid input for model parameters!\n");
                exit(1);
        }
}

// configures relaxation
void configure_relaxation(struct Relaxation *relaxation, FILE *input) {
fscanf(input, " %d %lf %lf %lf %lf %d", &relaxation->T, &relaxation->dx, &relaxation->dy, &relaxation->dz, &relaxation->dt, &relaxation->T_optimize); // read from input file
}

// returns the one mode approximation for 3D
double compute_amplitude(struct Model *model, FILE *input) {
    double Q = 9216.0 * model->no;
    double R = 138240.0;

    // Compute alpha = 3 * (Bl - 0.75 * B_s)
    double alpha = 3.0 * (model->Bl - 0.75 * model->Bs);

    // Compute the constant part of the quadratic term
    double P = 8.0 * alpha + 576.0 * model->no * model->no;

    // Compute discriminant
    double discriminant = Q * Q - 4.0 * R * P;

    if (discriminant < 0) {
        printf("Discriminant < 0: no real solution for A.\n");
        return 0.0;
    }

    // Use only the physically meaningful (smaller positive) root
    double A1 = (-Q + sqrt(discriminant)) / (2.0 * R);
    double A2 = (-Q - sqrt(discriminant)) / (2.0 * R);

    // Return the non-zero root with correct sign (usually A2 is physical)
     return (fabs(A2) > 1e-12) ? A2 : A1;

//    if (fabs(A2) > 1e-12)
//       return A2;
//    else
//       return A1;
}

// One-Mode Approximation field generator (BCC)
double OMA(double x, double y, double z, double x0, double y0, double z0, struct Model *model) {

    double aa = 0.0; // Density field at point (x, y, z)
    double a = 1.0;  // Lattice spacing
    double q0 = (2.0 * pi / a) / sqrt(2.0);  // Magnitude of reciprocal vector
    int i;

    // Compute A dynamically
    double A;
    A = compute_amplitude(model, NULL);

    // BCC reciprocal lattice directions
    int qx_int[N_BCC_MODES] = {  1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0 };
    int qy_int[N_BCC_MODES] = {  1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1 };
    int qz_int[N_BCC_MODES] = {  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1 };

    for (i = 0; i < N_BCC_MODES; i++) {
        double qx = q0 * qx_int[i];
        double qy = q0 * qy_int[i];
        double qz = q0 * qz_int[i];
        aa += 4 * A * (cos(qx) * cos(qy) + cos(qy) * cos(qz) + cos(qx) * cos(qz));
    }
    return aa; 
}

// initializes the density field with a crystallite in center
// R is radius
void embedded_crystallite(struct Model *model, struct Arrays *arrays, double dx, double dy, double dz, double no, double R, double Bl, double Bs) {
int Wp=2*(arrays->W/2+1);
int w, h, z, gw, gh, gz, k, x0, y0, z0;
double R2 = R*R;
double xx, yy, zz, r2;

//double Bl = model->Bl;
//double Bs = model->Bs;
//double no = model->no;

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
                //arrays->q[k] = no; // (may be) set to average density        
                gw = arrays->lw0+w; 
             xx = (gw-0.5*arrays->lW)*dx;
             k=z*arrays->lH*Wp+h*Wp+w;      
             r2 = xx*xx+yy*yy+zz*zz;
             if(r2<R2){
        arrays->q[k] = no+OMA(xx, yy, zz, x0, y0, z0, model);
}
}
}
}
} 

void initialize_system(struct Model *model, struct Arrays *arrays, FILE *input) {
int init;
if(fscanf (input, " %d", &init)!=1){
    printf("Invalid initialization type!\n");
    exit(1);
}
if(init==1){
double dx, dy, dz, no, R, Bl, Bs;
if(fscanf(input, " %lf %lf %lf %lf %lf %lf %lf", &dx, &dy, &dz,  &no, &R, &Bl, &Bs)!= 7){
    printf("Invalid initialization type!\n");
    exit(1);
}
embedded_crystallite(model, arrays, dx, dy, dz, no, R, Bl, Bs);
}
}

// saves state into a file
void write_state(struct Arrays *arrays, struct Relaxation *relaxation, 
struct Output *output) {
    char filename[128];  // output filename
    sprintf(filename, "%s-t:%d.dat", output->name, relaxation->t);
    FILE *file;
    int Wp = 2 * (arrays->W / 2 + 1); // padded width for FFT
    int w, h, z, k, i;

    for (i = 0; i < relaxation->ID; i++) {
        MPI_Barrier(MPI_COMM_WORLD);  // synchronize MPI ranks

        if (relaxation->id == i) {
            // process 0 overwrites, others append
            file = (relaxation->id == 0) ? fopen(filename, "w") : 
fopen(filename, "a");

            for (z = 0; z < arrays->lZ; z++) {
                for (h = 0; h < arrays->lH; h++) {
                    k = z * arrays->lH * Wp + h * Wp; // start index for (z,h) slice
                    for (w = 0; w < arrays->W; w++, k++) {
                        fprintf(file, "%e %e\n", arrays->q[k], arrays->u_n[k]);
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
        if(relaxation->id == 0) {                       // only 0th process
printf("%d %d %.9lf %.9lf %.9lf %.9lf %.9lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->dz, relaxation->f, relaxation->p);
                char filename[128];
                sprintf(filename, "%s.out", output->name);
                FILE *file;
                file = fopen(filename, "a");    // need only append, empty file was already generated
fprintf(file, "%d %d %.9lf %.9lf %.9lf %.9lf %.9lf\n", relaxation->t, (int)(time(NULL)-relaxation->t0), relaxation->dx, relaxation->dy, relaxation->dz, relaxation->f, relaxation->p);
                fclose(file);
        }
}

void update_AB(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
    int lZ, lH, gh, w, z, h, gz, idx;
    double kz, ky, kx, k2, k4, k6, L, denom, safeL;

    lZ = arrays->lZ;
    lH = arrays->lH;
    double dkx = 2.0 * pi / (relaxation->dx * arrays->W);
    double dky = 2.0 * pi / (relaxation->dy * arrays->H);
    double dkz = 2.0 * pi / (relaxation->dz * arrays->Z);
    double dt  = relaxation->dt;

    // Constants from model
    double C1 = fabs(model->Me * model->Bl);
    double C2 = fabs(2.0 * model->Me * model->Bs);
    double C3 = fabs(model->Me * model->Bs);

    const double EPS = 1e-8; // Small safety constant

    for (z = 0; z < lZ; z++) {
        gz = arrays->lz0 + z;
        kz = (gz <= arrays->Z / 2) ? gz * dkz : (gz - arrays->Z) * dkz;

        for (h = 0; h < lH; h++) {
            gh = arrays->lh0 + h;
            ky = (gh <= arrays->H / 2) ? gh * dky : (gh - arrays->H) * dky;

            for (w = 0; w <= arrays->W / 2; w++) {
                kx = w * dkx;
                k2 = -4*pi*pi*(kx * kx + ky * ky + kz * kz);
                k4 = k2 * k2;
                k6 = k4 * k2;

                // Dispersion relation term
                L = C1 * k2 + C2 * k4 + C3 * k6;

                // Denominator for implicit part
                denom = 1.0 - dt * L;

                // Clamp denom to avoid division by very small numbers
                if (fabs(denom) < EPS) {
                    denom = (denom >= 0.0) ? EPS : -EPS;
                }

                idx = (z * lH + h) * (arrays->W / 2 + 1) + w;

                arrays->A[idx] = 1.0 / denom;

                // Avoid division by zero when computing B
                safeL = (fabs(L) < EPS) ? EPS : L;
                arrays->B[idx] = (arrays->A[idx] - 1.0) / (dt * safeL);
            }
        }
    }
}

// scales p arrays (in k-space) by 1/(W*H) ((I)FFTs cause scaling of data by sqrt(W*H*Z))
void scale_P(struct Arrays *arrays) {
        fftw_complex *P;                        // complex data pointers
        P = (fftw_complex*)&arrays->p[0];
        double divWHZ; 
        divWHZ = 1.0/(arrays->W*arrays->H* arrays->Z);
        int i, lA;
        lA = arrays->lW*arrays->H* arrays->Z;
        for(i = 0; i < lA; i++) P[i] *= divWHZ;
}

// scales q arrays (in k-space) by 1/(W*H*Z)
void scale_Q(struct Arrays *arrays) {
        fftw_complex *Q;
        Q = (fftw_complex*)&arrays->q[0];
        double divWHZ = 1.0/(arrays->W*arrays->H* arrays->Z);
        int i, lA;
        lA = arrays->lW*arrays->H* arrays->Z;
        for(i = 0; i < lA; i++) Q[i] *= divWHZ;
}

// computes average free energy density (f) and average density (p) (f & p -> fp)
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
        fftw_complex *Q = (fftw_complex *)&arrays->q[0];
        // Apply operator in Fourier space
         int z, h, w;
        for (z = 0; z < lZ; z++) {
                int gz = arrays->lz0 + z;
                double kz = (gz < arrays->Z / 2) ? gz * dkz : (gz - arrays->Z) * dkz;
                double kz2 = kz * kz;

                for (h = 0; h < lH; h++) {
                        int gh = arrays->lh0 + h;
                        double ky = (gh < arrays->H / 2) ? gh * dky : (gh - arrays->H) * dky;
                        double ky2 = ky * ky;

                        for (w = 0; w < (lW / 2 + 1); w++) {
                                double kx = w * dkx;
                                double kx2 = kx * kx;

                                double k2 = -4.0 * pi * pi * (kx2 + ky2 + kz2);
                               // double d2 = (k2 - 1.0) * (k2 - 1.0); 
                                double d2 = (k2 - 2.0) * (k2 - 2.0);

                                int izx = (z * lH + h) * (arrays->W / 2 + 1) + w;

                                Q[izx] *= d2;
                        }
                }
        }

        fftw_execute(arrays->Q_q);  // inverse FFT: back to real space

//      relaxation->f = 0.0;
//      relaxation->p = 0.0;

        for (z = 0; z < lZ; z++) {
                int gz = arrays->lz0 + z;
                double kz = (gz < arrays->Z / 2) ? gz * dkz : (gz - arrays->Z) * dkz;
                double kz2 = kz * kz;

                for (h = 0; h < lH; h++) {
                        int gh = arrays->lh0 + h;
                        double ky = (gh < arrays->H / 2) ? gh * dky : (gh - arrays->H) * dky;
                        double ky2 = ky * ky;

                        for (w = 0; w < arrays->W; w++) {
                                double kx = w * dkx;
                                double kx2 = kx * kx;

                                double k2 = -4.0 * pi * pi * (kx2 + ky2 + kz2);
                                double k4 = k2 * k2;

                                int izx = (z * lH + h) * Wp + w;

                                double p = arrays->p[izx];
                        //      double q = arrays->q[izx];
                        
                                double p2 = p*p;        
                                double p4 = p2*p2;

                                // Free energy density: match your model's form
                                relaxation->f += 0.5 * (model->Bl + model->Bs * (2 * k2 + k4)) * p2 - (model->v / 6.0) * p2 * p + 0.25 * p4;

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

void update_us(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
        int Wp = 2 * (arrays->W / 2 + 1);
        int w, gw, h, z, gz, k;
        double divWHZ = 1.0 / arrays->W / arrays->H / arrays->Z;

        // Copy q fields to u fields
        for(z = 0; z < arrays->lZ; z++) {
                for(h = 0; h < arrays->H; h++) {
                        k = Wp * arrays->H * z + Wp * h;
                        for(w = 0; w < arrays->W; w++) {
                                arrays->u_n[k + w] = arrays->q[k + w];
                        }
                }
        }
        // Forward FFT
        fftw_execute(arrays->u_U_n);

        // Smoothing in Fourier space
        double dkx = 2.0 * pi / (relaxation->dx * arrays->W);
        double dky = 2.0 * pi / (relaxation->dy * arrays->H);
        double dkz = 2.0 * pi / (relaxation->dz * arrays->Z);
        double a = -1.0 / (2.0 * model->sigma_u * model->sigma_u);
        double kx2, ky, kz, k2;

        fftw_complex *U_n = (fftw_complex*)&arrays->u_n[0];

        for(w = 0; w < arrays->lW; w++) {
                gw = arrays->lw0 + w;
                kx2 = gw * dkx;
                kx2 *= kx2;
                for(h = 0; h < arrays->H; h++) {
                        if(h < arrays->H / 2) ky = h * dky;
                        else ky = (h - arrays->H) * dky;
                        for(z = 0; z < arrays->Z; z++) {
                                gz = arrays->lz0 + z;
                                if(gz < arrays->Z / 2) kz = gz * dkz;
                                else kz = (gz - arrays->Z) * dkz;
                                k2 = kx2 + ky * ky + kz * kz;
                                k = Wp * arrays->H * z + Wp * h + w;
                                U_n[k] *= exp(a * k2) * divWHZ;
                        }
                }
        }

        // Inverse FFT
        fftw_execute(arrays->U_u_n);

 // Find min/max
        arrays->u_n_min = 1.0e100;
        arrays->u_n_max = -1.0e100;

        for(z = 0; z < arrays->lZ; z++) {
                for(h = 0; h < arrays->H; h++) {
                        k = Wp * arrays->H * z + Wp * h;
                        for(w = 0; w < arrays->W; w++) {
                                double val_n = arrays->u_n[k + w];
                                if(val_n < arrays->u_n_min) arrays->u_n_min = val_n;
                                if(val_n > arrays->u_n_max) arrays->u_n_max = val_n;
                        }
                }
        }
        MPI_Allreduce(MPI_IN_PLACE, &arrays->u_n_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &arrays->u_n_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

// performs one iteration of the semi-implicit spectral method in 3D with operator splitting
void step(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
    int Wp = 2 * (arrays->W / 2 + 1);
    int lZ = arrays->lZ, lH = arrays->lH;
    double dt = relaxation->dt;
int z, h, w, idx;
    for (z = 0; z < lZ; z++) {
        for (h = 0; h < lH; h++) {
            for (w = 0; w < arrays->W; w++) {
                idx = (z * lH + h) * Wp + w;
                double n = arrays->q[idx];
                double n2 = n * n, n3 = n2 * n;
                double a1 = model->Me * model->Bl * n + model->Me * (-0.5 * model->v * n2 + n3 / 3.0);
                arrays->q[idx] += dt * a1;
            }
        }
    }
    fftw_execute(arrays->q_Q);
    fftw_complex *Q = (fftw_complex *)arrays->q;

    for (z = 0; z < lZ; z++) {
        for (h = 0; h < lH; h++) {
            for (w = 0; w <= arrays->W / 2; w++) {
                idx = (z * lH + h) * (arrays->W / 2 + 1) + w;
                Q[idx] *= arrays->A[idx];
            }
        }
    }
    fftw_execute(arrays->Q_q);
    update_us(arrays, model, relaxation);       
}

// optimizes calculation box size for system
// samples slightly different box sizes by varying dx, dy and dz and interpolates optimum quadratically
void optimize(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation) {
    // ----- SETTINGS -----
    const int enforce_cubic_box = 0;  // set to 1 to enforce dx = dy = dz (for symmetric BCC box)
    const double l0 = 2.0 * pi * sqrt(2.0);  // BCC lattice constant when q0 = 1

    // ----- SAMPLE SIZES -----
    double dx0 = relaxation->dx;
    double dy0 = relaxation->dy;
    double dz0 = relaxation->dz;

    double dd = relaxation->d;

    double dxs[3] = {dx0 - dd, dx0, dx0 + dd};
    double dys[3] = {dy0 - dd, dy0, dy0 + dd};
    double dzs[3] = {dz0 - dd, dz0, dz0 + dd};

    double fs[7];  // Free energy densities
    int i;
    // Sample around original size (central +/- delta in each direction)
    for (i = 0; i < 7; i++) {
        // Reset to original
        relaxation->dx = dx0;
        relaxation->dy = dy0;
        relaxation->dz = dz0;

        if (i == 1) relaxation->dx = dxs[0];  // dx - d
        if (i == 2) relaxation->dx = dxs[2];  // dx + d
        if (i == 3) relaxation->dy = dys[0];  // dy - d
        if (i == 4) relaxation->dy = dys[2];  // dy + d
        if (i == 5) relaxation->dz = dzs[0];  // dz - d
        if (i == 6) relaxation->dz = dzs[2];  // dz + d

        // Compute free energy
        fp(arrays, model, relaxation);
        fs[i] = relaxation->f;
    }
   // ----- INTERPOLATE NEW STEP SIZES -----
    // Quadratic interpolation: f = a + b*dx + c*dx^2 => minimum at dx = (f(-d)-f(d) + 2dx0(f(d)+f(-d)-2f0)) / 2(f(-d)-2f0+f(d))

    double new_dx = (dd * (fs[1] - fs[2]) + 2.0 * dx0 * (-2.0 * fs[0] + fs[1] + fs[2])) /
                    (2.0 * (fs[1] - 2.0 * fs[0] + fs[2]));
    double new_dy = (dd * (fs[3] - fs[4]) + 2.0 * dy0 * (-2.0 * fs[0] + fs[3] + fs[4])) /
                    (2.0 * (fs[3] - 2.0 * fs[0] + fs[4]));
    double new_dz = (dd * (fs[5] - fs[6]) + 2.0 * dz0 * (-2.0 * fs[0] + fs[5] + fs[6])) /
                    (2.0 * (fs[5] - 2.0 * fs[0] + fs[6]));

    if (enforce_cubic_box) {
        // Enforce isotropic (cubic) box: average the three
        double avg_d = (new_dx + new_dy + new_dz) / 3.0;
        new_dx = new_dy = new_dz = avg_d;
    }

    // ----- LIMIT DISCRETIZATION CHANGE -----
    double delta_x = arrays->W * (new_dx - dx0);
    double delta_y = arrays->H * (new_dy - dy0);
    double delta_z = arrays->Z * (new_dz - dz0);
    double dr = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
    double limit = 0.25 * l0;

    if (dr > limit) {
        double scale = limit / dr;
        new_dx = scale * new_dx + (1.0 - scale) * dx0;
        new_dy = scale * new_dy + (1.0 - scale) * dy0;
        new_dz = scale * new_dz + (1.0 - scale) * dz0;
    }

    // ----- UPDATE DISCRETIZATION -----
    relaxation->dx = new_dx;
    relaxation->dy = new_dy;
    relaxation->dz = new_dz;

    update_AB(arrays, model, relaxation);  // Update spectral operators due to changed grid

  // ----- ADAPTIVE STEP SIZE TUNING -----
    double ddx = new_dx - dx0;
    double ddy = new_dy - dy0;
    double ddz = new_dz - dz0;

    double new_dr = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
    if (new_dr < dd)
        relaxation->d *= 0.5;
    else
        relaxation->d *= 2.0;

    if (relaxation->d < 1.0e-6)
        relaxation->d *= 2.0;  // prevent numerical underflow
}

// relaxes the system for T time steps
void relax(struct Arrays *arrays, struct Model *model, struct Relaxation *relaxation, struct Output *output) {
        for(relaxation->t = 0; relaxation->t <= relaxation->T; relaxation->t++) {
                if(relaxation->T_optimize > 0 && relaxation->t > 0 && relaxation->t % relaxation->T_optimize == 0) optimize(arrays, model, relaxation); // optimize?
                if(relaxation->t % output->T_print == 0) {      // print output?
                        fp(arrays, model, relaxation);
                        print(relaxation, output);
                }
                if(relaxation->t % output->T_write == 0) 
write_state(arrays, relaxation, output);                // save state?
                if(relaxation->t < relaxation->T) step(arrays, model, relaxation);      // perform iteration step
        }
}

// frees allocated arrays
void clear_arrays(struct Arrays *arrays) {
        fftw_free(arrays->p);
        fftw_free(arrays->q);
        fftw_free(arrays->A);
        fftw_free(arrays->B);
        fftw_free(arrays->u_n);
}

int main(int argc, char **argv) {
    // init MPI
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <run_name>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

    char filename[128];
    snprintf(filename, sizeof(filename), "%s.in", output.name);
    FILE *input = fopen(filename, "r");
    if (input == NULL) {
        fprintf(stderr, "Input file %s not found!\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    snprintf(filename, sizeof(filename), "%s.out", output.name);
    FILE *out = fopen(filename, "w");
    if (out) fclose(out);

  char label;

char line[1024];
    while (fscanf(input, " %c", &label) != EOF) {
        if (label == '#' && fgets(line, 1024, input)) {
        } else if (label == 'S') {
            seed_rngs(&relaxation, input);
        } else if (label == 'O') {
            configure_output(&output, input);
        } else if (label == 'A') {
            configure_arrays(&arrays, input);
            // Fix 6: Check FFTW plan success
            if (!arrays.p_P || !arrays.q_Q || !arrays.Q_q) {
                fprintf(stderr, "FFTW plan creation failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else if (label == 'I') {
            initialize_system(&model, &arrays, input);
        } else if (label == 'M') {
            configure_model(&model, input);
        } else if (label == 'R') {
            if (fscanf(input, " %d %lf %lf %lf %lf %d", &relaxation.T, 
&relaxation.dx, &relaxation.dy, &relaxation.dz, &relaxation.dt, 
&relaxation.T_optimize) != 6) {
                fprintf(stderr, "Invalid relaxation parameters!\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            update_AB(&arrays, &model, &relaxation);
            update_us(&arrays, &model, &relaxation);
            relax(&arrays, &model, &relaxation, &output);
        } else {
            fprintf(stderr, "Invalid label in input: %c\n", label);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    fclose(input);

    clear_arrays(&arrays);
    MPI_Finalize();
    return 0;
}
