// ============================================================================
//   bfield_server.h -- FDTD Electromagnetic Field Solver Definitions
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

#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <libpng16/png.h>
#include <cstdint>

// --- Test Modes ---

#define GAUSSIAN_MATCH_LC
//#define GAUSSIAN_MATCH_LC_BY_FITTING
//#define H_BEFORE_E_NEG_Y
#define H_BEFORE_E
//#define NO_HE_OFFSET_GRID
//#define NO_HE_OFFSET_HALF
#define INCLUSIVE_ALPHA
//#define USE_VECTORS
#define DISPLAY_H_V_PER_M

const int maxName = 30;

// --- General Structures and Utilities ---

#ifdef USE_VECTORS
const int wv = 16;   // size of vectors
typedef double vdbl __attribute__((ext_vector_type(wv),aligned(wv)));
#else
const int wv = 1;
typedef double vdbl;
#endif

struct int3 {
    int     i, j, k;

    int3() { }
    int3(int ii, int jj, int kk) { i = ii; j = jj; k = kk; }
    int3(const int3& B) { i = B.i; j = B.j; k = B.k; }

    int3 operator = (const int3 B) {
        i = B.i; j = B.j; k = B.k;
        return *this;
    }
    int operator[] (size_t idx) {
        return *(&i + idx);
    }
    int3 operator + (const int3 B) {
        return int3(i+B.i, j+B.j, k+B.k);
    }
    int3 operator - (const int3 B) {
        return int3(i-B.i, j-B.j, k-B.k);
    }
    int3 operator / (const int3 B) {
        return int3(i/B.i, j/B.j, k/B.k);
    }
    int3 operator += (const int3 B) {
        i += B.i; j += B.j; k += B.k;
        return *this;
    }
    int3 operator -= (const int3 B) {
        i -= B.i; j -= B.j; k -= B.k;
        return *this;
    }
    int3 operator + (int b) {
        return int3(i+b, j+b, k+b);
    }
    int3 operator - (int b) {
        return int3(i-b, j-b, k-b);
    }
    int3 operator * (int b) {
        return int3(i*b, j*b, k*b);
    }
    int3 operator += (int b) {
        i += b; j += b; k += b;
        return *this;
    }
    int3 operator -= (int b) {
        i -= b; j -= b; k -= b;
        return *this;
    }
};

struct double3;
struct dbl3;

struct vdbl3 {
    vdbl* x;
    vdbl* y;
    vdbl* z;

    void operator=(dbl3& B);
};

struct dbl3 {
    double* x;
    double* y;
    double* z;

    dbl3() { ; }
    void init(size_t nCells) {
        x = new double[nCells];
        y = new double[nCells];
        z = new double[nCells];
    }
    void free() {
        delete[] x;
        delete[] y;
        delete[] z;
    }
    void set(size_t idx, double3 B);
    double3 get(size_t idx);
};


inline void vdbl3::operator=(dbl3& B) {
    x = (vdbl*)B.x; y = (vdbl*)B.y; z = (vdbl*)B.z;
}

struct double3 {
    double x, y, z;

    double3() { }
    double3(double xx, double yy, double zz) { x = xx; y = yy; z = zz; }
    double3(const double3& B) { x = B.x; y = B.y; z = B.z; }
    double3(const int3& B) { x = float(B.i); y = float(B.j); z = float(B.k); }
    double3(const dbl3& B, size_t idx) {
        x = B.x[idx]; y = B.y[idx]; z = B.z[idx];
    }

    double3 operator = (const double3 B) {
        x = B.x; y = B.y; z = B.z;
        return *this;
    }
    double operator[] (size_t i) {
        return *(&x + i);
    }
    double3 operator + (const double3 B) {
        return double3(x+B.x, y+B.y, z+B.z);
    }
    double3 operator - (const double3 B) {
        return double3(x-B.x, y-B.y, z-B.z);
    }
    double3 operator / (const double3 B) {
        return double3(x/B.x, y/B.y, z/B.z);
    }
    double3 operator += (const double3 B) {
        x += B.x; y += B.y; z += B.z;
        return *this;
    }
    double3 operator -= (const double3 B) {
        x -= B.x; y -= B.y; z -= B.z;
        return *this;
    }
    double3 operator + (double b) {
        return double3(x+b, y+b, z+b);
    }
    double3 operator - (double b) {
        return double3(x-b, y-b, z-b);
    }
    double3 operator * (double b) {
        return double3(x*b, y*b, z*b);
    }
    double3 operator += (double b) {
        x += b; y += b; z += b;
        return *this;
    }
    double3 operator -= (double b) {
        x -= b; y -= b; z -= b;
        return *this;
    }

    double length() { return sqrt(x*x + y*y + z*z); }
};

inline int max(int a, int b) { return (a > b) ? a : b; }
inline double min(double a, double b) { return (a < b) ? a : b; }
inline double max(double a, double b) { return (a > b) ? a : b; }

inline double3 min(double3 A, double3 B) {
    return double3(min(A.x, B.x), min(A.y, B.y), min(A.z, B.z));
}
inline double3 max(double3 A, double3 B) {
    return double3(max(A.x, B.x), max(A.y, B.y), max(A.z, B.z));
}

inline void dbl3::set(size_t idx, double3 B) {
        x[idx] = B.x; y[idx] = B.y; z[idx] = B.z; }
inline double3 dbl3::get(size_t idx) {
        return double3(x[idx], y[idx], z[idx]); }

uint8_t clampu8(double v);
uint8_t clampu8s(double v);
int16_t clamp16(double v);
 
// Physical constants
const double mu0 = 1.2566370614e-6; // permeability constant, H/m, [1,1,-2,-2]
const double ep0 = 8.854187817e-12; // vacuum permittivity, F/m, [-1,-3,4,2]
extern double c0; // = 2.998e8;    // speed of light, m/s, [0,1,-1,0]
extern double z0; // = 376.7303113; // ohms, impedance of free space
extern double zd; // = 1 for displaying H in V/m, = z0 for F/m
// ([1,2,-2,-2]-[-1,-2,4,2])/2 = [2,4,-6,-4]/2 = [1,2,-3,-2] = V/A = ohms
const double pi = 3.14159265359;

// ---------------------------------------------------------------------------
// Report an error and exit. Same args as printf.

class Err {
public:
    char* message;
    Err(const char* fmt, ...);
    void report();
};

// ---------------------------------------------------------------------------
// Simulation globals.

extern double unit;         // input coordinates units, m
extern int nsteps;          // number of time steps to run
enum Mode {IDLE, RUN, PAUSE};
extern Mode mode;
extern int mismatches;      // mismatch count

struct SubSpace;

enum CellType {NORMAL, HARD_E_SOURCE};

// ============================================================================
// Physical material constants.

struct Material {
    static Material* materials;
    Material* next;
    char    name[maxName+1];
    double  mur;    // relative permeability
    double  epr;    // relative permittivity
    double  sige;   // electric conductivity
    double  sigh;   // magnetic conductivity

    Material(const char* name, double mur, double epr, double sige,
             double sigh);
    Material(const char** args, int argc);
    static Material* find(const char* name);
    static void deleteAll();
};

// ============================================================================
// Virtual class: a dimensioned structure in the sim world.

struct Block {
    Block*  next;
    char    name[maxName+1];
    char    func[maxName+1];    // function: 'gaus', etc.
    double3 Bsg;    // untrimmed bounds, in global (client) coordinates
    double3 Beg;
    double3 Bs;     // trimmed-to-Fields bounds, Fields-relative
    double3 Be;
    int3    Is;     // trimmed bounds as indices
    int3    Ie;
    int     verbose;

    void initICoords1(double e);
    static void initICoords(Block* blocks, double extend=0);
};

// ============================================================================
// A volume of material in the sim world.

struct MatBlock: Block {
    static MatBlock* matBlks;
    char            mtype;
    Material*       material;
    int             pixShift;  // 24 for Alpha channel, 0 for Red
    png_uint_32***  voxels;  // 3D image, 2x, y-swapped, abgr values
    void    insert();

    // 3-deep stack of (alpha, material) sets per cell, to be blended
    struct AStack {
        float ah;
        float ae;
        Material* mat;
    };
    static const int asz = 6;
    static AStack* aStacks;

    MatBlock(const char* name, const char* mtype, Material* mat,
             double3 Bs, double3 Be, bool insertMat=true);
    MatBlock(char** args, int argc, bool insertMat=true);
    ~MatBlock() { }
    void place(int nix, int niy, int niz);
    virtual void place();
    void alphaPush(int i, int j, int k, double ah, double ae);
    static void deleteAll();
    static void placeInit();
    static void placeAll();
    static void placeFinish();
    static void initICoords(double extend) {
            Block::initICoords((Block*)matBlks, extend); }
};

// ============================================================================
// Image-based MatBlock, from a stack of one or more image files.

struct IMBlock: MatBlock {
    int     nix, niz; // niz images, each nix*nix

    IMBlock(char** args, int argc);
    ~IMBlock();
    void place();
};

// ============================================================================
// Image-based MatBlock, from a stack of one or more image files.

struct LayerMBlock: MatBlock {
    int     nix, niy; // image dimensions

    LayerMBlock(char** args, int argc);
    ~LayerMBlock();
    void place();
};

// ============================================================================
// Source block for injecting signals.

struct Source: Block {
    static Source* sources;
    bool    isHard;
    double  R;
    int     axis;
    double  scale;
    double  tstart;
    double  trise;
    double  duration;
    double  tfall;
    double  sigma;
    char    excite;
    double  (*customFunc)(Source* src, double t);

    Source(const char* name, double3 Bs, double3 Be);
    Source(char** args, int argc);
    void setFunc(const char* s) { strncpy(func, s, maxName); }
    static void deleteAll();
    void inject();
    static void preInjectAll();
    static void postInjectAll();
    static void initICoords() { Block::initICoords((Block*)sources); }
};

// ============================================================================
// Probe block for measuring fields.

struct Probe: Block {
    static Probe* probes;
    static bool printing;
    char    fieldName;
    char    dispType;
    float   dispScale;
    int     sfactor;
    double3 S;
    bool    printedProbe;

    virtual void wrInit(int nP) = 0;
    virtual void wrElem(size_t ic) = 0;
    virtual void wrZero() = 0;
    virtual size_t wrCount() = 0;
    virtual void sumElem(size_t ic, double sign=1.) = 0;
    virtual char* Pchar() = 0;

    Probe(char** args, int argc);
    void update(char** args, int argc);
    void printCell(size_t idx);
    void print();
    void writeCells(int fd);
    void sumCells(int fd);

    static void deleteAll();
    static Probe* find(const char* name);
    static void add(char** args, int argc);
    static void writeCells(int fd, char* name);
    static void initICoords(double extend) {
            Block::initICoords((Block*)probes, extend); }
};

struct ProbeGeneric: Probe {
    void wrInit(int nP) { }
    void wrElem(size_t ic) { }
    void wrZero() { }
    size_t wrCount() { return 0; }
    void sumElem(size_t ic, double sign=1.) { }
    char* Pchar() { return 0; }

    ProbeGeneric(char** args, int argc): Probe(args, argc) { }
};

// ============================================================================
// A simulation space or subspace.
//
// j,k:      ba    0                 N     M
//        +-+-PML-+------ // -------+-PML-+-+
// idx/J1: 0 1     5                 13  17 18    (for N.j=8, ncb=5)
//        \____________ Ncb.j ______________/
//
// i:                     ba     0                 N     M
//       + . . // . . . +-+-PML-+------ // -------+-PML-+-+ . . . // . . +
// idx:   0             15 16    20                28   32 33             48
//                      \_____________ Ncb.i _____________/   (N.i=16, wv=16)
// ic:    0                1                             2                3
//       +---- // --------+------------ // -------------+-------- // ----+
//       \____________________________ Nv.i _____________________________/

struct Space: MatBlock {
    int3    N;          // interior grid dimensions, in cells
    int3    Ncb;        // grid dimensions, in cells, including PML+cond
    int3    Nv;         // grid dimensions, vector-aligned
    int     nb;         // PML area thickness
    int     ncb;        // border width: PML+conductive, or subspace inferface
    int     ba;         // PML area base (inside conductive layer)
    int3    M;          // PML area end, high sides
    int     ncbv;       // vector-aligned border width, PML+conductive
    int     zo0;        // verbose==3 cells to print, Z start
    int     nzo;        // verbose==3 cells to print, Z number
    double  dx;         // grid spacing: cubic cell size, =dy=dz, m
    int     step;       // step no.

    // Derived globals
    double  dt;         // time step, 1/s, for stability
                        // i stride = 1
    int     J1;         // j stride
    int     k1;         // k stride
    int     nCells;     // total number of cells
    double3 Bsgf, Begf; // Fields bounds, in global coordinates
    double  goffs;      // input coordinate offset (-dx/2)

    // icell-indexed vector arrays
    dbl3    E;      // E electric field, V/m, [1,1,-3,-1]
    dbl3    H;      // H˜ = sqrt(mu0/ep0)*H, normalized magnetic field, V/m
                    //      (H magnetic field, A/m, [0,-1,0,1])
    double* mur;    // relative permeability
    double* epr;    // relative permittivity
    double* sige;   // electric conductivity, S/m, on E grid
    double* sigh;   // magnetic conductivity, S/m, on H grid
    dbl3    mE1;    // pre-computed term coefs for H and E, no PML
    dbl3    mE2;
    dbl3    mH1;
    dbl3    mH2;
    dbl3    J;      // source open-circuit current density
    double* volt;   // voltage of cell relative to origin
    CellType* cellType; // special type code: hard source, etc.

    // vectorized pointers to same icell-indexed vector arrays
    int     vJ1;        // j stride in vdbls
    int     vk1;        // k stride in vdbls
    vdbl3   vE, vH;
    vdbl*   vmur;
    vdbl*   vepr;
    vdbl*   vsige;
    vdbl*   vsigh;
    vdbl3   vmE1, vmE2, vmH1, vmH2;
    vdbl3   vJ;

    SubSpace* children; // linked list of child spaces
    Space*  next;       // next in list of all spaces

    static Space* spaces;

    Space();
    Space(const char* name, Material* mat, double3& Bs, double3& Be):
                      MatBlock(name, "M", mat, Bs, Be, false) { insert(); }
    Space(char** args, int argc): MatBlock(args, argc, false) { insert(); }
    ~Space();
    void insert();
    void allocate();
    Space* find(const char* name);
    size_t idxcell(size_t i, size_t j, size_t k) {
        return (k+ncb)*k1 + (j+ncb)*J1 + (i+ncbv);
    }
    vdbl v(double* A, size_t idx) { return *(vdbl*)&A[idx]; }
    void coord2grid(const double3* B, int3* I);
    void integVolts();
    void printCell(size_t idx);
    void dumpHE(int step);
    void stepH();
    void stepE();
    static void deleteAll();
};

// ============================================================================
// Subspace parameters.

struct SubSpace: Space {
    Space*  parent;     // parent space
    SubSpace* nextSib;  // next sibling
    int3    Ip;         // position of this space in parent space

    SubSpace(Space*  parent);
    SubSpace(char** args, int argc);
    static void initICoords();
};


// ============================================================================
// Simulation main space extra parameters.

struct OuterSpace: Space {
    int3    Nb;         // PML area size, minus conductive layer
    double  extend;     // how far objects may go outside Fields box, m
    int     conductBorder;  // outer 1 cell shell is conductive (reflective)

    OuterSpace(const char* name, Material* mat, double3 Bs, double3 Be):
                            Space(name, mat, Bs, Be) { }
    OuterSpace(char** args, int argc);
    void runInit();
    void allocate();
    void initPMLarea(int x0, int xn, int y0, int yn, int z0, int zn,
                     int3 Id);
    void stepPMLareaH(size_t idxb, int x0, int xn, int y0, int yn,
                      int z0, int zn);
    void stepPMLareaE(size_t idxb, int x0, int xn, int y0, int yn,
                      int z0, int zn);
    void stepH();
    void stepE();
};

extern OuterSpace* osp;     // main (outer) space

void setupInit();
void fieldEnd();
