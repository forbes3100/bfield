// ============================================================================
//   testmath.cpp -- Test FDTD Electromagnetic Field Solver's Math
//
//   Copyright 2023 Scott Forbes
//
// This file is part of BField.
// BField is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
// BField is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
// You should have received a copy of the GNU General Public License along
// with BField. If not, see <https://www.gnu.org/licenses/>.
// ============================================================================

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif
#include "bfield_server.h"

int3 Nl;
const char* testName;
Material* air;
Material* copper;
Material* teflon;

size_t  nlc;        // # of LC cells, for comparison
double  errThresh;  // LC-compare error threshold
double  errMin;     //   minimum base value threshold
bool    showAll;    //   show all comparisons
int     mismatches; // mismatch count
double  eTotal;     // total error
double  stopPctThresh; // percentage error above which to stop comparing
bool    dontPropagateMismatches = false;

dbl3    Etm1, Htm1;  // E & H at previous step, for checkLC
double3* Elc;
double3* Hlc;
double3* Elctm1;
double3* Hlctm1;
int lcProbeAdjs[12] = {  // (X,Y) adj for:
     0,  0,   // E.x
     0,  0,   // E.y
     0,  0,   // E.z
     0,  0,   // H.x
     0,  0,   // H.y
     0,  0,   // H.z
};

inline double3& lcell(double3* a, size_t li, size_t lj, size_t lk) {
                    return a[li + lj*Nl.i + lk*Nl.i*Nl.j]; }

#ifdef __APPLE__
void mach_abs_diff(uint64_t end, uint64_t start, struct timespec *tp) {
        uint64_t diff = end - start;
        static mach_timebase_info_data_t info = {0,0};

        if (info.denom == 0)
                mach_timebase_info(&info);

        uint64_t elapsednano = diff * (info.numer / info.denom);

        tp->tv_sec = elapsednano * 1e-9;
        tp->tv_nsec = elapsednano - (tp->tv_sec * 1e9);
}
#endif

// ---------------------------------------------------------------------------
// Read gaussian pulse function data from LC file.

int nGauss = 0;
double* Xgauss;
double* Ygauss;

void gaussianInit() {
    char cwd[200];
    char* r = getcwd(cwd, 200);
    if (r == NULL) {
        perror("getcwd");
        throw new Err("Bad working dir");
    }
    printf("cwd=%s\n", cwd);
    const char* gausLCFile = "lc_output/small/src-pulse.xy";
    FILE* gf = fopen(gausLCFile, "r");
    if (gf == 0) {
        throw new Err("Missing %s. Unzip lc_output.zip? Or Product>Scheme>Edit>"
                      "Run>Options>Working Dir to '$PROJECT_DIR'", gausLCFile);
    }
    char* line = NULL;
    size_t linecap = 0;
    ssize_t ll = getline(&line, &linecap, gf);
    for (int i = 0; ll > 0 && i < 3; i++)
        ll = getline(&line, &linecap, gf);
    nGauss = atoi(line);
    printf("Reading %d lines of LC gaussian data\n", nGauss);
    Xgauss = new double[nGauss];
    Ygauss = new double[nGauss];
    ll = getline(&line, &linecap, gf);
    ll = getline(&line, &linecap, gf);

    for (int i = 0; ll > 0; i++) {
        int n = sscanf(line, "%lg %lg", &Xgauss[i], &Ygauss[i]);
        if (n != 2)
            throw new Err("bad line %d in %s", i+5, gausLCFile);
        ll = getline(&line, &linecap, gf);
    }
    fclose(gf);
}

// ---------------------------------------------------------------------------
// Gaussian pulse for sources.

double gaussianLC(Source* s, double t) {
    if (nGauss < 2)
        throw new Err("no gaussian data?");
    double gdt = Xgauss[1] - Xgauss[0];
    int i = min((t - s->tstart) / gdt, nGauss);
    return s->scale * Ygauss[i];
}

// ----------------------------------------------------------------------------
// Check fields against given LC values.
// All internal H,E values in V/m for comparison.

void checkLC1(double f, double3* lcp, const char* name,
                          const int3& I, int* lcAdj) {
    char nm = name[0];
    char ax = name[1];
    int i = I.i;
    int j = I.j;
    int k = I.k;
    size_t idx = osp->idxcell(i,j,k);
    int ai = ax - 'x';
    int3 N = osp->N;
    int J1 = osp->J1;
    int k1 = osp->k1;
    int lJ1 = N.i;
    int lk1 = N.i*N.j;
    size_t lidx = (i+lcAdj[0]) + (j+lcAdj[1])*lJ1 + k*lk1;
    lcp += lidx;
    double lc = *(&lcp->x + ai);

    dbl3& mE1 = osp->mE1;
    dbl3& mE2 = osp->mE2;
    dbl3& mH1 = osp->mH1;
    dbl3& mH2 = osp->mH2;
    dbl3& J = osp->J;
    double mX1  = (*((nm == 'E' ? &mE1.x  : &mH1.x)  + ai))[idx];
    double mX2  = (*((nm == 'E' ? &mE2.x  : &mH2.x)  + ai))[idx];
    double Xtm1 = (*((nm == 'E' ? &Etm1.x : &Htm1.x) + ai))[idx];
    double Xlctm1 = *((nm == 'E' ? &Elctm1[lidx].x :
                                   &Hlctm1[lidx].x) + ai);
    double X1, X2, X3, X4, L1, L2, L3, L4;
    double J_ = 0.;
    if (nm == 'E') {
        switch (ax) {
            case 'x':
                X1 = Htm1.z[idx]; X2 = Htm1.z[idx-J1];
                X3 = Htm1.y[idx]; X4 = Htm1.y[idx-k1];
                L1 = Hlctm1[lidx].z; L2 = Hlctm1[lidx-lJ1].z;
                L3 = Hlctm1[lidx].y; L4 = Hlctm1[lidx-lk1].y;
                J_ = J.x[idx];
                break;
            case 'y':
                X1 = Htm1.x[idx]; X2 = Htm1.x[idx-k1];
                X3 = Htm1.z[idx]; X4 = Htm1.z[idx- 1];
                L1 = Hlctm1[lidx].x; L2 = Hlctm1[lidx-lk1].x;
                L3 = Hlctm1[lidx].z; L4 = Hlctm1[lidx-  1].z;
                J_ = J.y[idx];
                break;
            default:
                X1 = Htm1.y[idx]; X2 = Htm1.y[idx- 1];
                X3 = Htm1.x[idx]; X4 = Htm1.x[idx-J1];
                L1 = Hlctm1[lidx].y; L2 = Hlctm1[lidx-  1].y;
                L3 = Hlctm1[lidx].x; L4 = Hlctm1[lidx-lJ1].x;
                J_ = J.z[idx];
                break;
        }
   } else {
        dbl3& E = osp->E;
        switch (ax) {
            case 'x':
                X1 = E.z[idx+J1]; X2 = E.z[idx];
                X3 = E.y[idx+k1]; X4 = E.y[idx];
                L1 = Elc[lidx+lJ1].z; L2 = Elc[lidx].z;
                L3 = Elc[lidx+lk1].y; L4 = Elc[lidx].y;
                break;
            case 'y':
                X1 = E.x[idx+k1]; X2 = E.x[idx];
                X3 = E.z[idx+ 1]; X4 = E.z[idx];
                L1 = Elc[lidx+lk1].x; L2 = Elc[lidx].x;
                L3 = Elc[lidx+  1].z; L4 = Elc[lidx].z;
                break;
            default:
                X1 = E.y[idx+ 1]; X2 = E.y[idx];
                X3 = E.x[idx+J1]; X4 = E.x[idx];
                L1 = Elc[lidx+  1].y; L2 = Elc[lidx].y;
                L3 = Elc[lidx+lJ1].x; L4 = Elc[lidx].x;
                break;
        }
    }
    double base = max( max(
                max(max(fabs(f), fabs(lc)), max(fabs(Xtm1), fabs(Xlctm1))),
                max(max(fabs(X1), fabs(X2)), max(fabs(X3), fabs(X4)))),
                max(max(fabs(L1), fabs(L2)), max(fabs(L3), fabs(L4))));
    double e = fabs(f - lc);
    if (base != 0) {
        e /= base;
    }
    bool miss = e > errThresh && base > errMin;

    if ((showAll && (f != 0 || lc != 0)) || miss) {
        if (osp->verbose) {
            bool hardE = nm == 'E' && osp->cellType[idx] == HARD_E_SOURCE;
            printf("%d %s:   %c[%d,%d,%d].%c",
                   osp->step, miss ? "mismatch":"        ", nm, i, j, k, ax);
            if (hardE) {
                printf(" = hard E source");
                printf("\n     Field: % 12.6g", f);
                printf("\n        LC: % 12.6g", lc);
            } else {
                printf(" = m%c1[%d,%d,%d].%c *  %c[%d,%d,%d].%c",
                       nm, i, j, k, ax, nm, i, j, k, ax);
                printf(" +  m%c2[%d,%d,%d].%c * (", nm, i, j, k, ax);
                if (nm == 'E') {
                    X1 /= zd; X2 /= zd; X3 /= zd; X4 /= zd;
                    L1 /= zd; L2 /= zd; L3 /= zd; L4 /= zd;
                    switch (ax) {
                        case 'x':
                            printf("(H[%d,%d,%d].z   -  H[%d,%d,%d].z)",
                                   i, j, k, i, j-1, k);
                            printf("  - (H[%d,%d,%d].y   -  H[%d,%d,%d].y)",
                                   i, j, k, i, j, k-1);
                            printf("  - J[%d,%d,%d].x)", i, j, k);
                            break;
                        case 'y':
                            printf("(H[%d,%d,%d].x   -  H[%d,%d,%d].x)",
                                   i, j, k, i, j, k-1);
                            printf("  - (H[%d,%d,%d].z   -  H[%d,%d,%d].z)",
                                   i, j, k, i-1, j, k);
                            printf("  - J[%d,%d,%d].y)", i, j, k);
                            break;
                        default:
                            printf("(H[%d,%d,%d].y   -  H[%d,%d,%d].y)",
                                   i, j, k, i-1, j, k);
                            printf("  - (H[%d,%d,%d].x   -  H[%d,%d,%d].x)",
                                   i, j, k, i, j-1, k);
                            printf("  - J[%d,%d,%d].z)", i, j, k);
                            break;
                    }
               } else {
                    // H: optionally display values in F/m
                    f /= zd; Xtm1 /= zd; base /= zd;
                    lc /= zd; Xlctm1 /= zd;

                    switch (ax) {
                        case 'x':
                            printf("(E[%d,%d,%d].z   -  E[%d,%d,%d].z)",
                                   i, j+1, k, i, j, k);
                            printf("  - (E[%d,%d,%d].y   -  E[%d,%d,%d].y))",
                                   i, j, k+1, i, j, k);
                            break;
                        case 'y':
                            printf("(E[%d,%d,%d].x   -  E[%d,%d,%d].x)",
                                   i, j, k+1, i, j, k);
                            printf("  - (E[%d,%d,%d].z   -  E[%d,%d,%d].z))",
                                   i+1, j, k, i, j, k);
                            break;
                        default:
                            printf("(E[%d,%d,%d].y   -  E[%d,%d,%d].y)",
                                   i+1, j, k, i, j, k);
                            printf("  - (E[%d,%d,%d].x   -  E[%d,%d,%d].x))",
                                   i, j+1, k, i, j, k);
                            break;
                    }
                }
                printf("\n     Field: % 12.6g = % 12.6g * % 12.6g + % 12.6g * (",
                       f, mX1, Xtm1, mX2);
                printf("(% 12.6g - % 12.6g) - (% 12.6g - % 12.6g))",
                       X1, X2, X3, X4);
                if (nm == 'E')
                    printf(" - % 12.6g", J_);
                printf("\n        LC: % 12.6g =              * % 12.6g"
                       " +              * (", lc, Xlctm1);
                printf("(% 12.6g - % 12.6g) - (% 12.6g - % 12.6g))",
                       L1, L2, L3, L4);
            }
            if (e < 10)
                printf(" (%3.3f%% /%6.3g)\n\n", e*100, base);
            else
                printf("\n\n");
        } else {
            printf("%d %s: %c[%d,%d,%d].%c = % g -> % g",
                osp->step, miss ? "mismatch":"        ", nm, i, j, k, ax, f, lc);
            if (e < 10)
                printf(" (%3.3f%% /%6.3g)\n", e*100, base);
            else
                printf("\n");
        }
        if (miss) {
            mismatches++;
            eTotal += e;
        }
    }
}

// ----------------------------------------------------------------------------

bool checkLC() {
    //integVolts();   // a test
    int3 N = osp->N;
    dbl3& E = osp->E;

    if (osp->zo0 < 1)
        throw new Err("%s: zo0 must be > 0", testName);
    if (osp->zo0 + osp->nzo > N.k - 1)
        throw new Err("%s: zo0+nzo must be < %d (max z - 1)", testName, N.k-1);

    for (int i = 0; i < N.i; i++) {
        for (int j = 0; j < N.j; j++) {
            for (int k = 0; k < N.k; k++) {
                if (k >= osp->zo0 and k < osp->zo0+osp->nzo) {
                    size_t idx = osp->idxcell(i,j,k);
                    const int3 I = int3(i,j,k);
                    int* p = lcProbeAdjs;
                    checkLC1(E.x[idx], Elc, "Ex", I, p); p += 2;
                    checkLC1(E.y[idx], Elc, "Ey", I, p); p += 2;
                    checkLC1(E.z[idx], Elc, "Ez", I, p);
                }
            }
        }
    }
    if (mismatches > 0 && 100*eTotal/mismatches > stopPctThresh)
        return true;

    dbl3& H = osp->H;
    for (int i = 0; i < N.i; i++) {
        for (int j = 0; j < N.j; j++) {
            for (int k = 0; k < N.k; k++) {
                if (k >= osp->zo0 and k < osp->zo0+osp->nzo) {
                    size_t idx = osp->idxcell(i,j,k);
                    const int3 I = int3(i,j,k);
                    int* p = lcProbeAdjs + 6;
                    checkLC1(H.x[idx], Hlc, "Hx", I, p); p += 2;
                    checkLC1(H.y[idx], Hlc, "Hy", I, p); p += 2;
                    checkLC1(H.z[idx], Hlc, "Hz", I, p);
                }
            }
        }
    }
    return (mismatches > 0 && 100*eTotal/mismatches > stopPctThresh);
}

// ----------------------------------------------------------------------------
// Keep copy of this step's values for next step's mismatch printouts.

void copyTMinus1() {
    dbl3& E = osp->E;
    dbl3& H = osp->H;

    for (size_t idx = 0; idx < osp->nCells; idx++) {
        Etm1.x[idx] = E.x[idx];
        Etm1.y[idx] = E.y[idx];
        Etm1.z[idx] = E.z[idx];
        Htm1.x[idx] = H.x[idx];
        Htm1.y[idx] = H.y[idx];
        Htm1.z[idx] = H.z[idx];
    }
    for (size_t idx = 0; idx < nlc; idx++) {
        Elctm1[idx] = Elc[idx];
        Hlctm1[idx] = Hlc[idx];
    }
}

// ----------------------------------------------------------------------------
// Copy this step's values from LC, so mismatches don't accumulate.

void copyLC2Field() {
    dbl3& E = osp->E;
    dbl3& H = osp->H;
    int3 N = osp->N;
    int lJ1 = N.i;
    int lk1 = N.i*N.j;
    for (int i = 0; i < N.i; i++) {
        for (int j = 0; j < N.j; j++) {
            for (int k = 0; k < N.k; k++) {
                size_t idx = osp->idxcell(i,j,k);
                size_t lidx = (i+lcProbeAdjs[0]) + (j+lcProbeAdjs[1])*lJ1 + k*lk1;
                E.x[idx] = Elc[lidx].x;
                E.y[idx] = Elc[lidx].y;
                E.z[idx] = Elc[lidx].z;
                H.x[idx] = Hlc[lidx].x;
                H.y[idx] = Hlc[lidx].y;
                H.z[idx] = Hlc[lidx].z;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Read LC probe-out files. Files and dest array are in k,j,i order, no PML.

void readLCData1(const char* fileNameFmt, double3* dest, int step) {
    size_t lorignx = osp->Norig.i;
    size_t lorigny = osp->Norig.j;
    size_t lnxy = Nl.i * Nl.j;
    size_t _ = 0;
    //printf("readLCData %s to %ldx%ldx%ld\n",
    //        fileNameFmt, Nl.i, Nl.j, Nl.k);
    for (int axis = 0; axis < 3; axis++) {
        for (size_t k = osp->zo0; k < osp->zo0+osp->nzo; k++) {
            double3* d = (double3*)((double*)(dest+k*lnxy) + axis);
            char* path = new char[101];
            strncpy(path, "lc_output/", 100);
            size_t i = strlen(path);
            char axisc = 'x' + axis;
            snprintf(path+i, 101-i, fileNameFmt, (int)k, axisc, step);
            FILE* f = fopen(path, "r");
            if (f == NULL)
                throw new Err("couldn't read LC file %s", path);
            for (int j = 0; j < lorigny; j++) {
                int i = 0;
                for ( ; i < lorignx-6; i += 6) {
                    _ = fscanf(f, "%13le%13le%13le%13le%13le%13le\n",
                               &d[0].x, &d[1].x, &d[2].x,
                               &d[3].x, &d[4].x, &d[5].x);
                    //printf("%d:Hlc.%c[%ld,%ld,%ld]=(%g %g %g %g %g %g)\n",
                    //       step, axisc, ij-(ij/Nl.i*Nl.i), ij/Nl.i, k,
                    //       d[0].x, d[1].x, d[2].x,
                    //       d[3].x, d[4].x, d[5].x);
                    d += 6;
                }
                for ( ; i < lorignx; i++) {
                    _ = fscanf(f, "%13le", &d->x);
                    d++;
                }
                for ( ; i < Nl.i; i++) {
                    d->x = d->y = d->z = 0.;
                    d++;
                }
          }
            size_t pos = ftell(f);
            fseek(f, 0, SEEK_END);
            size_t rem = ftell(f) - pos - 1;
             if (rem != 0)
                throw new Err("LC file %s too long by %d",
                    path, rem);
            fclose(f);
            delete[] path;
        }
    }
}

// ----------------------------------------------------------------------------

void readLCData(const char* path, int step) {
    char s[40];
    snprintf(s, 39, "%sHz%%d%%c%%03d.out", path);
    readLCData1(s, Hlc, step);
    for (size_t i = 0; i < nlc; i++) {
        Hlc[i].x *= z0;     // H in V/m internally
        Hlc[i].y *= z0;
        Hlc[i].z *= z0;
    }
    snprintf(s, 39, "%sEz%%d%%c%%03d.out", path);
    readLCData1(s, Elc, step);

    if (osp->verbose == 3) {
        printf("%d, LC: --------\n", step);
        bool printing = true;
        for (size_t i = 0; i < Nl.i; i++) {
            for (size_t j = 0; j < Nl.j; j++) {
                for (size_t k = osp->zo0; k < osp->zo0+osp->nzo; k++) {
                    double3 H = lcell(Hlc, i,j,k);
                    H.x /= zd; H.y /= zd; H.z /= zd;
                    double3 E = lcell(Elc, i,j,k);
                    if (H.length() == 0 && E.length() == 0) {
                        if (printing) {
                            printing = false;
                            printf("...\n");
                        }
                    } else {
                        printing = true;
                        int f = 6;  // fraction digits to display
                        int w = f+7;
                        printf("  H[%ld,%ld,%ld]=[% *.*g, % *.*g, % *.*g]"
                           " E=[% *.*g, % *.*g, % *.*g]\n", i, j, k,
                           w, f, H.x, w, f, H.y, w, f, H.z,
                           w, f, E.x, w, f, E.y, w, f, E.z);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------

void runLCTestSim() {
    int nCells = osp->nCells;
    Etm1.init(nCells);
    Htm1.init(nCells);
    Nl = osp->N;
    nlc = Nl.i * Nl.j * Nl.k;
    Hlc = new double3[nlc];
    Elc = new double3[nlc];
    Elctm1 = new double3[nlc];
    Hlctm1 = new double3[nlc];
    mismatches = 0;
    eTotal = 0.;
    size_t n = strlen(testName) + 2;
    char* path = new char[n];
    snprintf(path, n, "%s/", testName);
    bool missed = false;
    stopPctThresh = 1.;
    //stopPctThresh = 101.;

    for (osp->step = 0; osp->step < nsteps; osp->step++) {
        readLCData(path, osp->step);
        missed = checkLC();
        if (missed)
            break;
        if (dontPropagateMismatches) {
            copyLC2Field();
        }
        copyTMinus1();
        osp->stepE();
        osp->stepH();
    }
    if (!missed) {
        readLCData(path, osp->step);
        checkLC();
    }
    fieldEnd();
    delete[] Hlc;
    delete[] Elc;
    printf("\nTest %s %s, %d mismatches",
           testName, missed ? "stopped early" : "done", mismatches);
    if (mismatches > 0)
        printf(", %2.2f%% avg error", 100*eTotal / mismatches);
    printf(".\n");
}

// ----------------------------------------------------------------------------

void runTestSim() {
    osp->conductBorder = true;
    size_t pl = strlen(testName) + 2;
    char* path = new char[pl];
    snprintf(path, pl, "%s/", testName);

    printf("Running %d steps...\n", nsteps);
#ifdef __APPLE__
    uint64_t start = mach_absolute_time();
#endif
    for (osp->step = 0; osp->step < nsteps; osp->step++) {
        osp->stepE();
        osp->stepH();
    }
#ifdef __APPLE__
    uint64_t end = mach_absolute_time();
    struct timespec tp;
    mach_abs_diff(end, start, &tp);
    double elapsed = tp.tv_sec + tp.tv_nsec * 1e-9;
    long n = (osp->Ncb.i-1) * (osp->Ncb.j-1) * (osp->Ncb.k-1);
    int nops = 18 + 18 + 3;
    printf("Elapsed = %5.3f seconds, %3.2f ns/stepH, %3.1f MFLOPS\n",
           elapsed, 1e9*elapsed/(n*nsteps*2),
           nsteps*n*nops/(1e6*elapsed));
 #endif
    fieldEnd();
    printf("\nTest %s done.\n", testName);
}

// ----------------------------------------------------------------------------

void runSim() {
    osp->conductBorder = true;
    for (osp->step = 0; osp->step < nsteps; osp->step++) {
        osp->stepE();
        osp->stepH();
    }
    fieldEnd();
    printf("\nTest %s done\n", testName);
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/small.

void testSmall() {
    testName = "small";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;       // input coordinates units, m

    osp = new OuterSpace("0Fields", air,
                                   double3(0., 0., 0.), double3(8., 4., 4.));
    osp->dx = 0.001;    // grid spacing, m
    nsteps = 2;         // number of steps to run
    osp->zo0 = 1;       // first z layer
    osp->nzo = 2;       // number of z layers
    osp->ncb = 5;       // PML border thickness
    osp->verbose = 3;   // debug level

    Source* s = new Source("8Src", double3(2., 2., 2.), double3(3., 2., 2.));
    s->excite = 'E';
    //s->setFunc("Custom");
    //s->customFunc = gaussianLC;
    s->setFunc("Gaussian_Pulse");
    //s->isHard = false;
    //s->R = 50.;
    s->isHard = true;
    s->axis = 0;
    //s->scale = 0.168535 / 0.22018;
    s->scale = 1.;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1e-10;
    s->tfall = 0.;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/medium.

void testMedium() {
    testName = "medium";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    osp = new OuterSpace("0Fields", air,
                                  double3(0., 0., 0.), double3(16., 8., 8.));
    osp->dx = 0.001;
    nsteps = 3;
    osp->zo0 = 1;
    osp->nzo = 6;
    osp->ncb = 5;
    osp->verbose = 1;

#define MEDIUM_HARD
#ifdef MEDIUM_HARD
    Source* s = new Source("8Src", double3(3., 3., 3.), double3(5., 5., 5.));
    s->isHard = true;
    s->setFunc("Gaussian_Pulse");
    s->scale = 1.;
    s->sigma = pi;
#else
    Source* s = new Source("8Src", double3(3., 3., 3.), double3(5., 5., 5.));
    s->isHard = false;
    s->R = 50.;
    s->setFunc("Gaussian_Pulse");
    s->scale = 0.0012;
    s->sigma = pi;
#endif
    s->excite = 'E';
    s->axis = 0;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1e-10;
    s->tfall = 0.;
    s->verbose = true;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/dielectric.

void testDielectric() {
    char tn[20];
    snprintf(tn, 20, "dielectric");
    testName = tn;
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    osp = new OuterSpace("0Fields", air,
                                  double3(0., 0., 0.), double3(16., 8., 8.));
    osp->dx = 0.001;
    osp->zo0 = 2;
    osp->nzo = 5;
    osp->ncb = 5;
    nsteps = 3;
    osp->verbose = 3;

    new MatBlock("3dielectric", "M", teflon,
#ifndef NO_HE_OFFSET
                                  double3(4.5, 0., 0.), double3(16., 8., 8.));
#else
                                  double3(5.0, 0., 0.), double3(16., 8., 8.));
#endif

    Source* s = new Source("8Src", double3(4., 4., 4.), double3(5., 4., 4.));
    s->excite = 'E';
    s->isHard = true;
    s->scale = 1.;
    s->setFunc("Gaussian_Pulse");
    s->axis = 0;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 2.3e-11;
    s->tfall = 0.;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/conductor.

void testConductor() {
    testName = "conductor";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    double3 loc = double3(0., 0., 0.);
    double3 size = double3(12., 12., 12.);
    osp = new OuterSpace("0Fields", air, loc, loc+size);
    osp->dx = 0.001;
    nsteps = 5;
    osp->zo0 = 4;
    osp->nzo = 5;
    osp->ncb = 5;
    osp->verbose = 3;

    //loc = double3(9., 6., 5.5);           // wire
    //hsize = double3(6., 2., 1.) * 0.5;
    loc = double3(5., 0., 0.);              // wall
    size = double3(7., 12., 12.);
    //loc = double3(5., 7., 0.);             // y-edge
    //size = double3(7., 7., 12.);
    //loc = double3(8.5, 4.5, 6.);             // y-edge-2
    //hsize = double3(7., 5., 12.) * 0.5;
    new MatBlock("5Conductor", "M", copper, loc, loc+size);

    Source* s = new Source("8Src", double3(4., 6., 5.), double3(4., 6., 6.));
    s->excite = 'E';
    s->setFunc("Gaussian_Pulse");
    s->isHard = true;
    s->axis = 2;
    s->scale = 1.;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1;
    s->tfall = 1e-10;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/zcoax.

void testZCoax() {
    testName = "zcoax";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    osp = new OuterSpace("0Fields", copper, double3(0,0,0), double3(12,12,12));
    osp->dx = 0.001;
    nsteps = 5;
    osp->zo0 = 4;
    osp->nzo = 5;
    osp->ncb = 5;
    osp->verbose = 3;

    new MatBlock("3air",  "M", air,    double3(1, 1, 1), double3(3, 11,11));
    new MatBlock("3diel", "M", teflon, double3(3, 1, 1), double3(12,11,11));
    new MatBlock("5Cond", "M", copper, double3(3, 5, 5), double3(12,7, 7 ));

    Source* s = new Source("8Src",     double3(1, 5, 5), double3(3, 7, 7 ));
    s->excite = 'E';
    s->setFunc("Custom");
    s->customFunc = gaussianLC;
    s->isHard = false;
    s->R = 50.;
    s->axis = 0;
    s->scale = 0.168535 / 0.22018;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1;
    s->tfall = 1e-10;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/trace1sm.

void testTrace1sm() {
    testName = "trace1sm";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    double3 loc = double3(-0.8, 0., 0.7);
    double3 hsize = double3(1.2, 1.2, 1.2) * 0.5;
    osp = new OuterSpace("0Fields", air, loc-hsize, loc+hsize);
    osp->dx = 0.0001;
    nsteps = 5;
    osp->zo0 = 1;
    osp->nzo = 8;
    osp->ncb = 5;
    osp->verbose = 3;

#ifdef TEMP
    loc = double3(3., 0., -0.05);
    hsize = double3(10., 3., 1.5) * 0.5;
    new MatBlock("1PCB", "M", fr4, loc-hsize, loc+hsize);

    loc = double3(2.6, 0., 0.15);
    hsize = double3(10.5, 3.3, 0.3) * 0.5;
    new MatBlock("5GPlane", "M", copper, loc-hsize, loc+hsize);

    loc = double3(3.6, 0.05, 0.75);
    hsize = double3(9., 0.3, 0.1) * 0.5;
    new MatBlock("5Trace", "M", copper, loc-hsize, loc+hsize);

#endif
    loc = double3(-0.8, 0.05, 0.5);
    hsize = double3(0.2, 0.3, 0.4) * 0.5;
    Source* s = new Source("8Src", loc-hsize, loc+hsize);
    s->excite = 'E';
    s->setFunc("Custom");
    s->customFunc = gaussianLC;
    s->isHard = false;
    s->R = 50.;
    s->axis = 2;
    //s->scale = 0.3350746548;
    s->scale = 1.;
    s->sigma = pi;
    s->tstart = 0.;
    s->trise = 50e-12;
    s->duration = 1;
    s->tfall = 10e-12;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test MatBlock place() of adjacent blocks.

void testAlpha() {
    testName = "alpha";
    printf("\nRunning test %s\n\n", testName);
    mismatches = 0;
    setupInit();
    unit = 0.001;

    double3 loc = double3(2., 2., 2.);
    double3 hsize = double3(4., 4., 4.) * 0.5;
    osp = new OuterSpace("0Fields", air, loc-hsize, loc+hsize);
    osp->dx = 0.001;
    nsteps = 5;
    osp->zo0 = 1;
    osp->nzo = 1;
    osp->ncb = 5;
    osp->verbose = 3;

    loc = double3(1., 1., 1.);
    hsize = double3(2., 2., 2.) * 0.5;
    new MatBlock("A", "M", copper, loc-hsize, loc+hsize);

    loc = double3(1., 3., 1.);
    hsize = double3(2., 2., 2.) * 0.5;
    new MatBlock("B", "M", copper, loc-hsize, loc+hsize);

    loc = double3(3., 3., 1.);
    hsize = double3(2., 2., 2.) * 0.5;
    new MatBlock("C", "M", copper, loc-hsize, loc+hsize);

    gaussianInit();
    osp->runInit();
    runSim();
}

// ----------------------------------------------------------------------------
// Test and compare with ?

void testLarge() {
    testName = "large";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    osp = new OuterSpace("0Fields", air,
                                  double3(0., 0., 0.), double3(64., 64., 64.));
    osp->dx = 0.001;
    nsteps = 1000;
    osp->zo0 = 3;
    osp->nzo = 3;
    osp->ncb = 5;
    osp->verbose = 0;

    double3 loc = double3(20., 20., 20.);
    double3 hsize = double3(1., 0., 0.) * 0.5;
    Source* s = new Source("8Src", loc-hsize, loc+hsize);
    s->excite = 'E';
    s->setFunc("Custom");
    s->customFunc = gaussianLC;
    s->isHard = true;
    s->axis = 0;
    s->scale = 1.;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1e-10;
    s->tfall = 0.;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/wall.

void testWall() {
    testName = "wall";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;       // input coordinates units, m

    osp = new OuterSpace("0Fields", air,
                                   double3(0., 0., 0.), double3(8., 4., 4.));
    osp->dx = 0.001;
    nsteps = 5;
    osp->zo0 = 1;
    osp->nzo = 2;
    osp->ncb = 1;
    osp->verbose = 3;

    Source* s = new Source("8Src", double3(2., 2., 2.), double3(3., 2., 2.));
    s->excite = 'E';
    s->setFunc("Custom");
    s->customFunc = gaussianLC;
//#define VOLTS_TEST
#ifdef VOLTS_TEST
    s->isHard = false;
    s->R = 50.;
    s->scale = 1.;
#else
    s->isHard = true;
    //s->scale = 0.000168535 / 0.0001;
    s->scale = 1.;
#endif
    s->axis = 0;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1e-10;
    s->tfall = 0.;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

// ----------------------------------------------------------------------------
// Test and compare with lc_output/platesm.

void testPlatesm() {
    testName = "platesm";
    printf("\nRunning test %s\n\n", testName);
    setupInit();
    unit = 0.001;

    double3 loc = double3(6., 6., 6.);
    double3 hsize = double3(12., 12., 12.) * 0.5;
    osp = new OuterSpace("0Fields", air, loc-hsize, loc+hsize);
    osp->dx = 0.001;
    nsteps = 5;
    osp->zo0 = 4;
    osp->nzo = 5;
    osp->ncb = 5;
    osp->verbose = 3;

#define AS_IN_LC
#ifdef AS_IN_LC
    loc = double3(6., 6., 9.);
    hsize = double3(8., 8., 2.) * 0.5;
#else
    loc = double3(6.5, 6.5, 9.5);
    hsize = double3(9., 9., 3.) * 0.5;
#endif
    new MatBlock("plate0", "M", copper, loc-hsize, loc+hsize);
#ifdef AS_IN_LC
    loc = double3(6., 6., 3.);
#else
    loc = double3(6.5, 6.5, 3.5);
#endif
    new MatBlock("plate1", "M", copper, loc-hsize, loc+hsize);

    loc = double3(4.5, 6.5, 6.);
    hsize = double3(3., 3., 4.) * 0.5;
    Source* s = new Source("src1", loc-hsize, loc+hsize);
    s->excite = 'E';
    s->setFunc("Custom");
    s->customFunc = gaussianLC;
    s->isHard = false;
    s->R = 50.;
    s->axis = 2;
    //s->scale = 1.29003487;
    s->scale = 1.;
    s->tstart = 0.;
    s->trise = 1e-10;
    s->duration = 1;
    s->tfall = 1e-10;
    s->sigma = pi;

    gaussianInit();
    osp->runInit();
    runLCTestSim();
}

struct Test {
    void (*fn) ();
    int wantMiss;
};

Test tests[] = {
    {testSmall, 0},
    {testMedium, 0},
    {testDielectric, 0},
    {testConductor, 1},
    {testZCoax, 18},
    {testAlpha, 0},
    {testWall, 1},
    {testPlatesm, 64},
};

// ----------------------------------------------------------------------------

int main(int argc, const char* argv[]) {

    air = new Material("Air", 1., 1., 0., 0.);
    copper = new Material("Copper", 1., 1., 5.8e7, 0.);
    teflon = new Material("Teflon", 1., 2.8, 0., 0.);
    errThresh = 1e-2;
    //errThresh = 100.;
    errMin = 1e-11;
    showAll = true;

    try {
#if 0
        // run all tests
        for (Test& test: tests) {
            (test.fn)();
            if (mismatches != test.wantMiss)
                throw new Err("Test '%s' mismatches wrong! expected %d",
                              testName, test.wantMiss);
            else
                printf("As expected.\n");
        }
        printf("All tests done.\n");
#else
        //errThresh = 1e-5;
        //errMin = 1e-99;
        //showAll = true;
        //testSmall();      // 0 miss, 2s, hard
        //testMedium();     // 0 miss, 3s, hard
        //testDielectric(); // 0 miss, 3s, hard
        //testConductor();  // 0 miss, 3s, hard (wall config only)
        // TO BE CHECKED AGAIN:
        //testZCoax();      // 30+ miss, 43.3% err, 2s, soft x0.7654
        // TO BE CHECKED:
        //testTrace1sm();   // 48+ miss, 66.7% err, 1s, soft x0.335
        //testAlpha();      // --
        //testLarge();      // --
        //testWall();       // 1+ miss, 73.8% err, 2s, hard x1.685
        testPlatesm();    // 80+ miss, 40.2% err, 2s, soft x1.2900
#endif
    } catch (Err* err) {
        err->report();
        exit(-1);
    }
}
