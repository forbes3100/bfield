// ============================================================================
//   bfield_server.cpp -- FDTD Electromagnetic Field Solver Server
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

// see bfield_server.h to set test configuration

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "bfield_server.h"

uint8_t clampu8(double v) {
    if (v > 255.)  v = 255.;
    if (v < 0.)  v = 0.;
    return (uint8_t)(v);
}

uint8_t clampu8s(double v) {
    if (v >  127.)  v =  127.;
    if (v < -128.)  v = -128.;
    return (uint8_t)(v + 128.);
}

int16_t clamp16(double v) {
    if (v > SHRT_MAX)  v = SHRT_MAX;
    if (v < SHRT_MIN)  v = SHRT_MIN;
    return (int16_t)v;
}

double c0, z0, zd;

// ---------------------------------------------------------------------------
// Report an error and exit. Same args as printf.

Err::Err(const char* fmt, ...) {
	va_list	ap;
	va_start(ap, fmt);
    this->message = new char[101];
	vsnprintf(this->message, 100, fmt, ap);
	va_end(ap);
}

void Err::report() {
    printf("\n**** ERROR: %s\n\n", this->message);
}

// ---------------------------------------------------------------------------
// Read one PNG image file, returning a 2D array of pixels.

png_uint_32** readPNG(const char* path, int* nix, int* niy) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL)
        throw new Err("can't open file %s", path);

    png_struct* png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
            NULL, NULL, NULL);
    if (png == NULL)
        throw new Err("can't create png read struct");

    png_info* info = png_create_info_struct(png);
    if (info == NULL)
        throw new Err("can't allocate png mem");

    if (setjmp(png_jmpbuf(png)))
        throw new Err("libpng jmpbuf error");

    png_init_io(png, fp);
    png_read_info(png, info);
    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    *nix = width;
    *niy = height;

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    size_t rowbytes = png_get_rowbytes(png,info);
    png_byte** row_pointers = new png_bytep[height];
    for (int j = 0; j < height; j++) {
        row_pointers[j] = new png_byte[rowbytes];
    }
    png_read_image(png, row_pointers);
    fclose(fp);
    return (png_uint_32**)row_pointers;
}

// ---------------------------------------------------------------------------
// Simulation globals.

OuterSpace* osp;    // outer space
Space* sp;          // current space

// Simulation parameters
double  unit;       // input coordinates units, m
int     nsteps;     // number of time steps to run
Mode    mode;

// ---------------------------------------------------------------------------
// Gaussian pulse for sources.

double gaussian(double t, double offset, double scale, double sigma,
                double tstart, double trise, double duration, double tfall) {

    double trcent = tstart + trise;  // time of center of rising peak
#ifdef GAUSSIAN_MATCH_LC_BY_FITTING
    // 4 parameters determined by guassian_fit.py from LC source waveform
    t += osp->dt;
    const double scaleFit = 9.99999582e-01;
    const double trcentFit = 9.54929705e-11;
    const double triseOverSigmaFit = 2.25079233e-11;
    const double offset = -7.16491954e-08;
    scale *= scaleFit;
    trcent += trcentFit - 1e-10;
    const double sigma = 1e-10 / triseOverSigmaFit;
#endif

    trcent *= sigma / pi;
    const double fwtm2 = -4 * log(10);
    if (t < trcent) {
        double tr = t - trcent;     // time relative to center of peak
        return scale * exp(fwtm2*(tr*tr/(trise*trise))) + offset;
    }
    double tfcent = trcent + duration;
    if (t > tfcent) {
        double tf = t - tfcent;
        return scale * exp(fwtm2*(tf*tf/(tfall*tfall))) + offset;
    }
    return scale + offset;
}

// ----------------------------------------------------------------------------
// Convert 3D coordinates to grid indices.

void Space::coord2grid(const double3* B, int3* I) {
    double hdx = dx / 2;
    I->i = floor((B->x + hdx) / dx);
    I->j = floor((B->y + hdx) / dx);
    I->k = floor((B->z + hdx) / dx);
}


// ============================================================================
// Physical material constants.

Material* Material::materials;

Material::Material(const char* name, double mur, double epr, double sige,
                   double sigh) {
    strncpy(this->name, name, maxName);
    this->mur = mur;
    this->epr = epr;
    this->sige = sige;
    this->sigh = sigh;
    next = materials;
    materials = this;
}

Material::Material(const char** args, int argc) {
    if (argc != 6)
        throw new Err("Material: wrong number of args");
    strncpy(name, args[1], maxName);
    mur = atof(args[2]);
    epr = atof(args[3]);
    sige = atof(args[4]);
    sigh = atof(args[5]);
    next = materials;
    materials = this;
}

Material* Material::find(const char* name) {
    Material* m = materials;
    for ( ; m; m = (Material*)m->next)
        if (strcmp(m->name, name) == 0)
            break;
    if (!m)
        throw new Err("material %s not found", name);
    return m;
}

void Material::deleteAll() {
    if (materials) {
        Material* mNext;
        for (Material* m = materials; m; m = mNext) {
            mNext = (Material*)m->next;
            delete m;
        }
        materials = 0;
    }
}

// ============================================================================
// Block virtual struct: a dimensioned structure in the sim world.

void Block::initICoords1(double ex) {
    Bs = min(max(Bsg + sp->goffs, osp->Bsg-ex), osp->Beg+ex) - osp->Bsg;
    Be = min(max(Beg + sp->goffs, osp->Bsg-ex), osp->Beg+ex) - osp->Bsg;
    sp->coord2grid(&Bs, &Is);
    sp->coord2grid(&Be, &Ie);
    printf("%s @ %d:%d, %d:%d, %d:%d\n", name,
           Is.i, Ie.i, Is.j, Ie.j, Is.k, Ie.k);
}

void Block::initICoords(Block* blocks, double ex) {
    // trim block to fit within Fields block + extension on all sides,
    // and also compute grid coordinates
    for (Block* bl = blocks; bl; bl = bl->next)
        bl->initICoords1(ex);
}

// ============================================================================
// A volume of material in the sim world.

MatBlock* MatBlock::matBlks;
MatBlock::AStack* MatBlock::aStacks;

MatBlock::MatBlock(const char* name, const char* mtype, Material* mat,
                   double3 Bs, double3 Be, bool insertMat) {
    strncpy(this->name, name, maxName);
    this->mtype = mtype[0]; // X Y or Z or cylinders, C for others
    material = mat;
    Bsg = Bs * unit;
    Beg = Be * unit;
    if (insertMat)
        insert();
}

MatBlock::MatBlock(char** args, int argc, bool insertMat) {
    if (argc != 9) {
        if (argc < 2) {
            throw new Err("wrong number of args");
        } else {
            throw new Err("wrong number of args: %s %s...", args[0], args[1]);
        }
    }
    strncpy(name, args[1], maxName);
    mtype =   args[0][1];
    material = Material::find(args[2]);
    Bsg.x = atof(args[3]) * unit;
    Beg.x = atof(args[4]) * unit;
    Bsg.y = atof(args[5]) * unit;
    Beg.y = atof(args[6]) * unit;
    Bsg.z = atof(args[7]) * unit;
    Beg.z = atof(args[8]) * unit;
    if (insertMat)
        insert();
}

void MatBlock::insert() {
    voxels = 0;
    // append this MatBlock to end of linked list to preserve stacking order
    next = NULL;
    MatBlock* mb = matBlks;
    if (!mb) {
        matBlks = this;
    } else {
        while (mb->next)
            mb = (MatBlock*)mb->next;
        mb->next = this;
    }
}

void MatBlock::deleteAll() {
    if (matBlks) {
        MatBlock* oNext;
        for (MatBlock* mb = matBlks; mb; mb = (MatBlock*)oNext) {
            oNext = (MatBlock*)mb->next;
            delete mb;
        }
        matBlks = 0;
    }
}

// ----------------------------------------------------------------------------
// A grid area enclosing the material block, on either H or E centers.

struct MatArea {
    double3 S0, S1, E1, E0; // start and end ramp from 0 to 1 (inside) to 0
    double3 C;              // cylinder center
    double rmh2, rph2;      // cylinder radius +/- h, squared

    MatArea(double3& Bs, double3& Be, double h, double3& C, double r) {
        // set alpha ramp-up, down limits to half cell either side of edge
        S0 = Bs - h;
        S1 = Bs + h;
        if (Bs.x == 0) S1.x = 0;    // if up against domain edge, don't ramp
        if (Bs.y == 0) S1.y = 0;
        if (Bs.z == 0) S1.z = 0;
        E1 = Be - h;
        E0 = Be + h;
        double3 N = sp->N;
        if (Be.x == N.x) E1.x = N.x;
        if (Be.y == N.y) E1.y = N.y;
        if (Be.z == N.z) E1.z = N.z;
        this->C = C;
        rmh2 = (r-h) * (r-h);
        rph2 = (r+h) * (r+h);
    }
    void cylinder(double3& P, int axis);
    double in(double3& P, int axis);
};

void MatArea::cylinder(double3& P, int axis) {
    // a1 and a2 are the indecies of the other 2 axes; they form a circle
    int a1 = (axis+1) % 3;
    int a2 = (axis+2) % 3;
    // center coords of that circle
    double cy = *(&C.x + a1);
    double cz = *(&C.x + a2);

    double dzc2 = fabs(cz - *(&P.x + a2));
    dzc2 *= dzc2;
    if (dzc2 < rph2) {
        // trim S0,E0 to outer-ramp limit of cylinder
        double dcxz = sqrt(rph2 - dzc2);
        *(&S0.x + a1) = cy - dcxz;
        *(&E0.x + a1) = cy + dcxz;
        // trim S1,E1 to inner-ramp limit of cylinder
        dcxz = sqrt(rmh2 - dzc2);
        *(&S1.x + a1) = cy - dcxz;
        *(&E1.x + a1) = cy + dcxz;
        // other round axis too
        double dyc2 = fabs(cy - *(&P.x + a1));
        dyc2 *= dyc2;
        if (dyc2 < rph2) {
            double dcxy = sqrt(rph2 - dyc2);
            *(&S0.x + a2) = cz - dcxy;
            *(&E0.x + a2) = cz + dcxy;
            dcxy = sqrt(rmh2 - dyc2);
            *(&S1.x + a2) = cz - dcxy;
            *(&E1.x + a2) = cz + dcxy;
        }
    }
}

double MatArea::in(double3& P, int axis) {
    double px = *(&P.x + axis);
    double s0x = *(&S0.x + axis);
    double s1x = *(&S1.x + axis);
    double e1x = *(&E1.x + axis);
    double e0x = *(&E0.x + axis);

    // alpha: 1 when H, E element entirely inside material
    double alpha = 1.;
    if (px <= s0x || px > e0x) {
        alpha = 0.;
#ifdef XINCLUSIVE_ALPHA
    } else if (px <= s1x || px > e1x) {
        // in the ramp-up or down area: inclusive
        alpha = 1.;
    }
#else
    } else if (px <= s1x) {
        // in the ramp-up area: alpha is fraction within
        alpha = (px - s0x) / (s1x - s0x);
    } else if (px > e1x) {
        // ramp-down area
        alpha = (e0x - px) / (e0x - e1x);
    }
#endif
    return alpha;
}

// ----------------------------------------------------------------------------

void MatBlock::alphaPush(int i, int j, int k, double ah, double ae) {
    int3 N = sp->N;
    AStack* astk = aStacks + asz * (i*N.j*N.k + j*N.k + k);
    for (int i = asz-1; i > 0; i--) {
        astk[i] = astk[i-1];
    }
    astk->ah = ah;
    astk->ae = ae;
    astk->mat = material;
}

// ----------------------------------------------------------------------------
// Place MatBlocks in field arrays.

void MatBlock::place() {
    Material& mat = *material;
    printf("MatBlock-%c %s placed %s @ %d:%d, %d:%d, %d:%d\n", mtype, name,
           mat.name, Is.i, Ie.i, Is.j, Ie.j, Is.k, Ie.k);
    double h = sp->dx / 2;
    int cylAxis = mtype - 'X';
    bool cyl = (cylAxis >= 0 && cylAxis < 3);
    double r = 0;
    // cylinder center and radius (+/-h) from untrimmed coordinates
    double3 C = (Bsg + Beg) * 0.5 + sp->goffs - osp->Bsg;
    if (cylAxis == 2) {
        r = (Beg.x - Bsg.x) / 2;
    } else {
        r = (Beg.y - Bsg.y) / 2;
    }
    // slice entry and exit ramp from 0 at h outside to 1 at h inside
    // rise, fall start and end of this axis slice through object
    MatArea mh = MatArea(Bs, Be, h, C, r);
    MatArea me = MatArea(Bs, Be, h, C, r);

    // outside-start and end grid indicies
    int3 Is0, Ie0;
    sp->coord2grid(&mh.S0, &Is0);
    sp->coord2grid(&mh.E0, &Ie0);
    //Ie0 += 1;   // TODO: why???
    int3 N = sp->N;

    // loop through enclosing grid space, building array of alpha-material sets
    for (int i = Is0.i; i < min(Ie0.i, N.i); i++) {
        for (int j = Is0.j; j < min(Ie0.j, N.j); j++) {
            for (int k = Is0.k; k < min(Ie0.k, N.k); k++) {
                // position of E element is on-grid
                double3 Pe = double3(int3(i,j,k)) * sp->dx;
                // position of H element is half-grid-step up
                double3 Ph = Pe + h;
#ifdef H_BEFORE_E_NEG_Y
                Ph = Pe;
                Pe.x += h;
                Pe.y -= h;      // E to left and below H
                Pe.z += h;
#endif
#ifdef H_BEFORE_E
                Ph = Pe;
                Pe += h;
#endif
#ifdef NO_HE_OFFSET_GRID
                Ph = Pe;
#endif
#ifdef NO_HE_OFFSET_HALF
                Pe = Ph;
#endif
                // trim to any cylinder dimensions
                if (cyl) {
                    mh.cylinder(Ph, cylAxis);
                    me.cylinder(Pe, cylAxis);
                }
                // alpha: 1 when H, E element is entirely inside material
                double ah = min(mh.in(Ph, 0), min(mh.in(Ph, 1), mh.in(Ph, 2)));
                double ae = min(me.in(Pe, 0), min(me.in(Pe, 1), me.in(Pe, 2)));
#ifdef INCLUSIVE_ALPHA
                double th = 0.6;
                ah = ah > th ? 1. : 0.;
                ae = ae > th ? 1. : 0.;
#endif
                alphaPush(i, j, k, ah, ae);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Place voxels in field arrays.

void MatBlock::place(int nix, int niy, int niz) {
    Material* mat = material;
    png_uint_32*** vlayerp = voxels;
    int is = Is.i;
    int js = Is.j;
    int ks = Is.k;
    int nx = max(Ie.i - is, 1);
    int ie = is + nx;
    int ny = max(Ie.j - js, 1);
    int je = js + ny;
    int nz = max(Ie.k - ks, 1);
    int ke = ks + nz;

    double3 base = (Bsg + sp->goffs - osp->Bsg - Bs);
    int3 ibase;
    sp->coord2grid(&base, &ibase);
    int3 igstart, igend;
    double3 Bsgp = Bsg + sp->goffs;
    double3 Begp = Beg + sp->goffs;
    sp->coord2grid(&Bsgp, &igstart);
    sp->coord2grid(&Begp, &igend);
    double3 scaleImg;
    int3 nImg(nix, niy, niz);
    scaleImg = double3(igend - igstart) / nImg;
    printf("%s voxels placed %s @ %d:%d, %d:%d, %d:%d, iscale.x=%g\n", name,
            mat->name, is, ie, js, je, ks, ke, scaleImg.x);

    // resample image alphas into a grid-aligned volume
    struct AMerge {
        int   ia;
        int   total;
    };
    int n = nx*ny*nz;
    AMerge* alphas = new AMerge[n];
    for (AMerge* ap = alphas; ap < alphas+n; ap++) {
        ap->ia = 0;
        ap->total = 0;
    }
    int pixPol = (pixShift == 24) ? 1 : -1;
    int pixBase = (pixShift == 24) ? 0 : 255;

    // image is higher resolution than voxels
    for (int ki = 0; ki < niz; ki++) {
        png_uint_32** vcolp = *vlayerp++;
        for (int ji = niy-1; ji >= 0; ji--) { // image y is reversed
            png_uint_32* vrowp = *vcolp++;
            for (int ii = 0; ii < nix; ii++) {
                png_uint_32 abgr = *vrowp++;
                png_byte ia = ((abgr >> pixShift) & 0xff) * pixPol + pixBase;
                if (ia > 0) {
                    int i = ii * scaleImg.x + ibase.i;
                    int j = ji * scaleImg.y + ibase.j;
                    int k = ki * scaleImg.z + ibase.k;
                    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 &&
                        k < nz) {
                        if (scaleImg.z > 1.) {
                            for (int kr = k; kr < k+scaleImg.z; kr++) {
                                AMerge* ap = &alphas[i*ny*nz+j*nz+kr];
                                ap->ia += ia;
                                ap->total += 1;
                            }
                        } else {
                            AMerge* ap = &alphas[i*ny*nz+j*nz+k];
                            ap->ia += ia;
                            ap->total += 1;
                        }
                    }
                    //printf(" [%d,%d,%d] -> [%d,%d,%d]=%d/%d\n",
                    //    ii,ji,ki, i,j,k, ia, ap->total);
                }
            }
        }
    }

    for (AMerge* ap = alphas; ap < alphas+n; ap++) {
        if (ap->total > 0)
            ap->ia /= ap->total;
    }

    // alpha-merge the resampled image into the sim grid
    for (int i = is; i < ie; i++) {
        for (int j = js; j < je; j++) {
            for (int k = ks; k < ke; k++) {
                int ia = alphas[(i-is)*ny*nz+(j-js)*nz+(k-ks)].ia;
                if (ia > 0) {
                    double a = ia / 255.;
                    alphaPush(i, j, k, a, a);
                    // TODO: place voxels on separate H,E grids too
                    //printf("[%d,%d,%d]=%d\n", i,j,k, ia);
                }
            }
        }
    }
    delete[] alphas;
}

// ----------------------------------------------------------------------------
// Place MatBlocks in field arrays.

void MatBlock::placeInit() {
    // Create and initialize alpha stacks array.
    int3 N = sp->N;
    int n = N.i * N.j * N.k;
    aStacks = new AStack[asz*n];
    Material* background = sp->material;
    for (AStack* astk = aStacks; astk < aStacks+asz*n; astk += asz) {
        astk->ae = 1.1;
        astk->ah = 1.1;
        astk->mat = background;
    }
}

void MatBlock::placeAll() {
    for (MatBlock* mb = matBlks; mb; mb = (MatBlock*)mb->next) {
        if ((mb->Ie.k - mb->Is.k) == 0) {
            // "plane" layers sit on top of their copper, so adjust down 1
            mb->Is.k -= 1;
            mb->Ie.k -= 1;
        }
        mb->place();
    }
}

void MatBlock::placeFinish() {
    int3 N = sp->N;
    // loop through enclosing grid space again, blending mats into cells
    for (int i = 0; i < N.i; i++) {
        for (int j = 0; j < N.j; j++) {
            for (int k = 0; k < N.k; k++) {

                // update cell material coeffs, blending in if alpha < 1
                size_t idx = sp->idxcell(i,j,k);
                double ahsum = 0.;
                double aesum = 0.;
                double mur = 1.;
                double epr = 1.;
                double sigh = 0.;
                double sige = 0.;
                AStack* bastk = aStacks + asz * (i*N.j*N.k + j*N.k + k);
                AStack* eastk = bastk + asz;
                for (AStack* astk = bastk; astk < eastk; astk++) {
                    double ah = min(astk->ah, 1. - ahsum);
                    mur -= ah * (1 - astk->mat->mur);
                    sigh += ah * astk->mat->sigh;
                    ahsum += astk->ah;
                    if (ahsum >= 1.)
                        break;
                }
                for (AStack* astk = bastk; astk < eastk; astk++) {
                    double ae = min(astk->ae, 1. - aesum);
                    epr -= ae * (1 - astk->mat->epr);
                    sige += ae * astk->mat->sige;
                    aesum += astk->ae;
                    if (aesum >= 1.)
                        break;
                }
#ifdef XINCLUSIVE_ALPHA
                if (i == 5 && j == 7 && (k > 4 && k < 7))
                {
                    printf("placeFinal (%d,%d,%d) mur=%6.2f epr=%6.2f"
                       " sh=%6.2f se=%6.2f\n", i,j,k, mur, epr, sigh, sige);
                }
#endif
                sp->mur[idx] = mur;
                sp->epr[idx] = epr;
                sp->sigh[idx] = sigh;
                sp->sige[idx] = sige;
            }
        }
    }
    delete[] aStacks;
}


// ============================================================================
// Image-based MatBlock, from a stack of one or more image files.

// Read a stack of images for an IMBlock.

IMBlock::IMBlock(char** args, int argc) : MatBlock(args, argc-2) {
    nix = atoi(args[9]);
    niz = atoi(args[10]);

    char* path = new char[201];
    voxels = new png_uint_32**[niz];
    pixShift = 24;  // use Alpha channel

    for (int k = 0; k < niz; k++) {
        snprintf(path, 200, "cache_dp/%s_%04d/paintmap%04d.png",
                 name, niz, k+1);
        int width, height;
        voxels[k] = readPNG(path, &width, &height);
        if (width != nix || height != nix)
            throw new Err("Expected %dx%d size images for IMBlock %s, got %dx%d",
                nix, nix, name, width, height);
        if (verbose > 0)
            printf("%d: Read %dx%d PNG image\n", k, nix, nix);

    }
    delete[] path;
}

IMBlock::~IMBlock() {
    png_uint_32*** vlp = voxels;
    if (vlp) {
        png_uint_32*** vlend = vlp + niz;
        for ( ; vlp < vlend; vlp++) {
            png_uint_32** vcp = *vlp;
            png_uint_32** vcend = vcp + nix;
            for ( ; vcp < vcend; vcp++) {
                delete[] *vcp;
            }
            delete[] *vlp++;
        }
        delete[] voxels;
    }
}

// Place voxels in field arrays.

void IMBlock::place() {
    MatBlock::place(nix, nix, niz);
}

// ============================================================================
// Image-based MatBlock, from a stack of one or more image files.

// Read a layer's image for a LayerMBlock.

LayerMBlock::LayerMBlock(char** args, int argc) : MatBlock(args, argc-1) {
    char* path = args[9];
    voxels = new png_uint_32**[1];
    voxels[0] = readPNG(path, &nix, &niy);
    pixShift = 0;  // use Red channel
}

LayerMBlock::~LayerMBlock() {
    png_uint_32*** vlp = voxels;
    if (vlp) {
        png_uint_32** vcp = *vlp;
        png_uint_32** vcend = vcp + nix;
        for ( ; vcp < vcend; vcp++) {
            delete[] *vcp;
        }
        delete[] *vlp++;
        delete[] voxels;
    }
}

// Place voxels in field arrays.

void LayerMBlock::place() {
    MatBlock::place(nix, niy, 1);
}


// ============================================================================
// Source block for injecting signals.

Source* Source::sources;

Source::Source(const char* name, double3 Bs, double3 Be) {
    strncpy(this->name, name, maxName);
    Bsg = Bs * unit;
    Beg = Be * unit;
    next = sources;
    sources = this;
}

Source::Source(char** args, int argc) {
    if (argc != 18)
        throw new Err("Source: wrong number of args");
    strncpy(name, args[1], maxName);
    excite =   args[2][0];
    Bsg.x =    atof(args[3]) * unit;
    Beg.x =    atof(args[4]) * unit;
    Bsg.y =    atof(args[5]) * unit;
    Beg.y =    atof(args[6]) * unit;
    Bsg.z =    atof(args[7]) * unit;
    Beg.z =    atof(args[8]) * unit;
    strncpy(func, args[9], maxName);
    isHard =   atoi(args[10]);
    R =        atof(args[11]);
    axis =     atoi(args[12]);
    scale =    atof(args[13]);
    tstart =   atof(args[14]);
    trise =    atof(args[15]);
    duration = atof(args[16]);
    tfall =    atof(args[17]);
    sigma = pi;
    next = sources;
    sources = this;
}

void Source::deleteAll() {
    if (sources) {
        Source* sNext;
        for (Source* s = sources; s; s = sNext) {
            sNext = (Source*)s->next;
            delete s;
        }
        sources = 0;
    }
}

void Source::inject() {
    double val = scale; // default to constant
    double t = sp->step * sp->dt - tstart;
    if (strcmp(func, "Gaussian_Pulse") == 0) {
        double t = sp->step * sp->dt;
        double offset = 0.;
        double sc = scale;
        double tr = trise;
        double ts = tstart;
        double si = sigma;
#ifdef GAUSSIAN_MATCH_LC
        t += sp->dt;
        ts += 2*sp->dt;
        tr -= 2*sp->dt;
#endif
        val = gaussian(t, offset, sc, si, ts, tr, duration, tfall);
#ifdef X_GAUSSIAN_MATCH_LC
        // this fudge corrects testSmall errors when LC src sigma = 3.1416, but
        // isn't needed if LC sigma = 3.14
        double a = 11210452.808;
        double ta = a * (t - ts - tr);
        val -= ta * ta;
#endif

    }
    if (strcmp(func, "Sine") == 0) {
        val = scale * sin(t * 2*pi / duration);
    }
    if (strcmp(func, "Custom") == 0) {
        val = (*customFunc)(this, t);
    }
    if (isHard && (osp->verbose > 0 || verbose))
        printf("%d: src E[%d,%d,%d].%c = %g\n",
               sp->step, Is.i, Is.j, Is.k, 'x'+axis, val);

    // voltage sources (at least?) are inclusive of edges
    int3 Iev = Ie;
    switch (axis) {
        case 0:
            Iev.j += 1;
            Iev.k += 1;
            break;
        case 1:
            Iev.i += 1;
            Iev.k += 1;
            break;
        case 2:
            Iev.i += 1;
            Iev.j += 1;
            break;
    }
    int step = sp->step;
    double dx = sp->dx;
    double dt = sp->dt;

    for (int i = Is.i; i < Iev.i; i++) {
        for (int j = Is.j; j < Iev.j; j++) {
            for (int k = Is.k; k < Iev.k; k++) {
                size_t idx = sp->idxcell(i,j,k);
                switch (excite) {

                    case 'E':
                        if (isHard) {
                            sp->cellType[idx] = HARD_E_SOURCE;
                            switch (axis) {
                                case 0:
                                    sp->E.x[idx] = val;
                                    break;
                                case 1:
                                    sp->E.y[idx] = val;
                                    break;
                                default:
                                    sp->E.z[idx] = val;
                                    break;
                            }
                        } else {
                            if (step == -1)
                                printf("src break\n");
                            // set load and voltage in source's direction
                            // current density (*z0 due to mE2, ~H)
#define SOURCE_PUNCHES_HOLE
#ifdef SOURCE_PUNCHES_HOLE
                            double mur = 1.;
                            double epr = 1.;
                            double sigh = 0.;
                            double sige = 0.;
#else
                            double mur = sp->mur[idx];
                            double epr = sp->epr[idx];
                            double sigh = sp->sigh[idx];
                            double sige = sp->sige[idx];
#endif
                            sigh += 1/(R*dx);
                            sige += 1/(R*dx);
                            double stm = sigh*dt / (mu0*mur*2);
                            double ste = sige*dt / (ep0*epr*2);
                            double mH1 = (1. - stm) / (1. + stm);
                            double mE1 = (1. - ste) / (1. + ste);
                            double mH2 = -dt*z0 / ((1. + stm) *dx*mu0*mur);
                            double mE2 = dt / ((1. + ste) *
                                               dx*z0*ep0*epr);
                            double J = z0 * val / (dx*R);
                            switch (axis) {
                                case 0:
                                    sp->mH1.x[idx] = mH1;
                                    sp->mH2.x[idx] = mH2;
                                    sp->mE1.x[idx] = mE1;
                                    sp->mE2.x[idx] = mE2;
                                    sp->J.x[idx] = J;
                                    break;
                                case 1:
                                    sp->mH1.y[idx] = mH1;
                                    sp->mH2.y[idx] = mH2;
                                    sp->mE1.y[idx] = mE1;
                                    sp->mE2.y[idx] = mE2;
                                    sp->J.y[idx] = J;
                                    break;
                                default:
                                    sp->mH1.z[idx] = mH1;
                                    sp->mH2.z[idx] = mH2;
                                    sp->mE1.z[idx] = mE1;
                                    sp->mE2.z[idx] = mE2;
                                    sp->J.z[idx] = J;
                                    break;
                            }
                        } break;

                    case 'H': {
                        switch (axis) {
                            case 0:  sp->H.x[idx] += val; break;
                            case 1:  sp->H.y[idx] += val; break;
                            default: sp->H.z[idx] += val; break;
                        }
                    } break;
                }
            }
        }
    }
}

void Source::preInjectAll() {
    for (Source* src = sources; src; src = (Source*)src->next) {
        if (!src->isHard)
            src->inject();
    }
}

void Source::postInjectAll() {
    for (Source* src = sources; src; src = (Source*)src->next) {
        if (src->isHard)  // !preRoll duplicates a quirk of LC
            src->inject();
    }
}

// ============================================================================
// Probe block for measuring fields.

Probe* Probe::probes;
bool Probe::printing;

Probe::Probe(char** args, int argc) {
    if (argc != 13)
        throw new Err("Probe: wrong number of args");
    strncpy(name, args[1], maxName);
    update(args, argc);
}

void Probe::update(char** args, int argc) {
    Bsg.x =   atof(args[2]) * unit;
    Beg.x =   atof(args[3]) * unit;
    Bsg.y =   atof(args[4]) * unit;
    Beg.y =   atof(args[5]) * unit;
    Bsg.z =   atof(args[6]) * unit;
    Beg.z =   atof(args[7]) * unit;
    fieldName = args[8][0];
    dispType = args[9][0];
    dispScale = atof(args[10]);
    sfactor = atoi(args[11]);
    verbose = atoi(args[12]);
}

void Probe::deleteAll() {
    if (probes) {
        Probe* pNext;
        for (Probe* p = probes; p; p = pNext) {
            pNext = (Probe*)p->next;
            delete (ProbeGeneric*)p;
        }
        probes = 0;
    }
}

Probe* Probe::find(const char* name) {
    Probe* p = probes;
    for ( ; p; p = (Probe*)p->next)
        if (strcmp(p->name, name) == 0)
            break;
    if (!p)
        throw new Err("probe %s not found", name);
    return p;
}

void Probe::print() {
    printf("Probe %c %g %d %d %d %d %d %d %d\n",
           fieldName, dispScale, sfactor,
           Is.i, Ie.i, Is.j, Ie.j, Is.k, Ie.k);
}

// ----------------------------------------------------------------------------
// Write field array to file fd. Stride sfactor determines
// how many elements are sent.

void Probe::writeCells(int fd) {
    int3 Iep = Ie;
    int pnx = (Iep.i-Is.i+sfactor-1) / sfactor;
    if (pnx == 0) { pnx++; Iep.i++; }
    int pny = (Iep.j-Is.j+sfactor-1) / sfactor;
    if (pny == 0) { pny++; Iep.j++; }
    int pnz = (Iep.k-Is.k+sfactor-1) / sfactor;
    if (pnz == 0) { pnz++; Iep.k++; }
    size_t nbytes;
    int nP = pnx*pny*pnz;
    size_t _ = 0;
    printedProbe = (mode != PAUSE);
    printing = true;
    wrInit(nP);

    if (dispType == 'S') {
        // sum all elements in the volume
        S.x = S.y = S.z = 0.;
        for (int i = Is.i; i < Iep.i; i++) {
            for (int j = Is.j; j < Iep.j; j++) {
                for (int k = Is.k; k < Iep.k ; k++) {
                    sumElem(sp->idxcell(i,j,k));
                }
            }
        }

        // send byte count followed by sum back to client
        float Sf[3];
        Sf[0] = S.x;
        Sf[1] = S.y;
        Sf[2] = S.z;
        nbytes = sizeof(Sf);
        char s[7];
        snprintf(s, 7, "%06ld", nbytes);
        _ = write(fd, s, 6);
        _ = write(fd, (char*)&Sf, nbytes);

    } else if (dispType == 'L') {
        // line integrate border of plane
        int linesum = 0;
#ifdef TODO
        if (Iep.i - Is.i == 1) {
            int j = Is.j;
            Cell* cpl = &sp->cell(Is.i,j,Is.k  );
            Cell* cph = &sp->cell(Is.i,j,Iep.k-1);
            S.y = 0.;
            for ( ; j < Iep.j ; j++) {
                sumElem(cpl++);
                sumElem(cph++, -1);
            }
            int k = Is.k;
            cpl = &sp->cell(Is.i,Is.j,  k);
            cph = &sp->cell(Is.i,Iep.j-1,k);
            S.z = S.y;
            for ( ; k < Iep.k; k++) {
                sumElem(cpl++, -1);
                sumElem(cph++);
            }
            linesum = S.z;
        } else if (Iep.j - Is.j == 1) {
            int i = Is.i;
            Cell* cpl = &sp->cell(i,Is.j,Is.k  );
            Cell* cph = &sp->cell(i,Is.j,Iep.k-1);
            S.x = 0.;
            for ( ; i < Iep.i ; i++) {
                sumElem(cpl++, -1);
                sumElem(cph++);
            }
            int k = Is.k;
            cpl = &sp->cell(Is.i,  Is.j,k);
            cph = &sp->cell(Iep.i-1,Is.j,k);
            S.z = S.x;
            for ( ; k < Iep.k; k++) {
                sumElem(cpl++);
                sumElem(cph++, -1);
            }
            linesum = S.z;
        } else {
            int i = Is.i;
            Cell* cpl = &sp->cell(i,Is.j,  Is.k);
            Cell* cph = &sp->cell(i,Iep.j-1,Is.k);
            S.x = 0.;
            for ( ; i < Iep.i ; i++) {
                sumElem(cpl++);
                sumElem(cph++, -1);
            }
            int j = Is.j;
            cpl = &sp->cell(Is.i,  j,Is.k);
            cph = &sp->cell(Iep.i-1,j,Is.k);
            S.y = S.x;
            for ( ; j < Iep.j ; j++) {
                sumElem(cpl++, -1);
                sumElem(cph++);
            }
            linesum = S.y;
        }
#endif

        // send byte count followed by sum, total back to client
        S.x = linesum;
        S.y = 0.;
        S.z = 0.;
        uint32_t count = (uint32_t)wrCount() / sizeof(float);
        nbytes = sizeof(S) + sizeof(count);
        char s[7];
        snprintf(s, 7, "%06ld", nbytes);
        _ = write(fd, s, 6);
        _ = write(fd, (char*)&S, sizeof(S));
        _ = write(fd, (char*)&count, sizeof(count));

    } else {
        // write all elements in volume, stepping by sfactor
        for (int i = Is.i; i < Iep.i; i += sfactor) {
            for (int j = Is.j; j < Iep.j; j += sfactor) {
                for (int k = Is.k; k < Iep.k; k += sfactor) {
                    wrElem(sp->idxcell(i,j,k));
                }
            }
        }
        // send byte count followed by data back to client
        nbytes = wrCount();
        char s[7];
        snprintf(s, 7, "%06ld", nbytes);
        _ = write(fd, s, 6);
        _ = write(fd, Pchar(), nbytes);
    }

    if (verbose)
        printf("writeCells: %dx%dx%d, wrote %ld bytes\n", pnx, pny, pnz,
               nbytes);
}

// ----------------------------------------------------------------------------

void Probe::writeCells(int fd, char* name) {
    Probe* p = Probe::find(name);
    if (p->verbose)
        p->print();
    p->writeCells(fd);
}

// ----------------------------------------------------------------------------
// virtual classes for each data type

struct Probe3I16: Probe {
    int16_t* p;
    int16_t* pip;

    Probe3I16(char** args, int argc): Probe(args, argc) { }
    void wrInit(int nP) { pip = p = new int16_t[3*nP]; }
    virtual void wrElem(size_t idx) = 0;
    virtual void sumElem(size_t idx, double sign=1.) = 0;
    void wrZero() {
        *pip++ = 0;
        *pip++ = 0;
        *pip++ = 0;
    }
    size_t wrCount() { return (pip-p)*sizeof(int16_t); }
    char* Pchar() { return (char*)p; }
};

struct ProbeI32: Probe {
    uint32_t* p;
    uint32_t* pip;

    ProbeI32(char** args, int argc): Probe(args, argc) { }
    void wrInit(int nP) { pip = p = new uint32_t[nP]; }
    virtual void wrElem(size_t idx) = 0;
    void sumElem(size_t idx, double sign=1.) { }
    void wrZero() { *pip++ = 0.; }
    size_t wrCount() { return (pip-p)*sizeof(uint32_t); }
    char* Pchar() { return (char*)p; }
};

struct ProbeFloat: Probe {
    float* pf;
    float* pfp;

    ProbeFloat(char** args, int argc): Probe(args, argc) { }
    void wrInit(int nP) { pfp = pf = new float[nP]; }
    virtual void wrElem(size_t idx) = 0;
    virtual void sumElem(size_t idx, double sign=1.) = 0;
    void wrZero() { *pfp++ = 0.; }
    size_t wrCount() { return (pfp-pf)*sizeof(float); }
    char* Pchar() { return (char*)pf; }
};

struct Probe3Float: Probe {
    float* pf;
    float* pfp;

    Probe3Float(char** args, int argc): Probe(args, argc) { }
    void wrInit(int nP) { pfp = pf = new float[3*nP]; }
    virtual void wrElem(size_t idx) = 0;
    virtual void sumElem(size_t idx, double sign=1.) = 0;
    void wrZero() {
        *pfp++ = 0.;
        *pfp++ = 0.;
        *pfp++ = 0.;
    }
    size_t wrCount() { return (pfp-pf)*sizeof(float); }
    char* Pchar() { return (char*)pf; }
};

// ----------------------------------------------------------------------------
// probe classes for each combination

/*
struct PrVecI16H: Probe3I16 {
    const double packScale = 1e5; // multiplier for packed int16_t values
    void wrElem(size_t idx) {
        double3 H = double3(sp->H, idx);
        *pip++ = clamp16(H.x/zd * packScale);
        *pip++ = clamp16(H.y/zd * packScale);
        *pip++ = clamp16(H.z/zd * packScale);
    }
    PrVecI16H(char** args, int argc): Probe3I16(args, argc) { }
};
*/
struct PrVecFloatH: Probe3Float {
    void wrElem(size_t idx) {
        double3 H = double3(sp->H, idx) * dispScale;
        *pfp++ = H.x/zd;
        *pfp++ = H.y/zd;
        *pfp++ = H.z/zd;
        if (!printedProbe)
            printCell(idx);
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 H = double3(sp->H, idx) * dispScale;
        S.x += sign * H.x/zd;
        S.y += sign * H.y/zd;
        S.z += sign * H.z/zd;
        pfp += 3;
    }
    PrVecFloatH(char** args, int argc): Probe3Float(args, argc) { }
};

struct PrVecFloatE: Probe3Float {
    void wrElem(size_t idx) {
        double3 E = double3(sp->E, idx) * dispScale;
        *pfp++ = E.x;
        *pfp++ = E.y;
        *pfp++ = E.z;
        if (!printedProbe)
            printCell(idx);
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 E = double3(sp->E, idx) * dispScale;
        S.x += sign * E.x;
        S.y += sign * E.y;
        S.z += sign * E.z;
        pfp += 3;
    }
    PrVecFloatE(char** args, int argc): Probe3Float(args, argc) { }
};

struct PrVecFloatJ: Probe3Float {
    void wrElem(size_t idx) {
        double3 J = double3(sp->J, idx) * dispScale;
        *pfp++ = J.x;
        *pfp++ = J.y;
        *pfp++ = J.z;
        if (!printedProbe)
            printCell(idx);
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 J = double3(sp->J, idx) * dispScale;
        S.x += sign * J.x;
        S.y += sign * J.y;
        S.z += sign * J.z;
        pfp += 3;
    }
    PrVecFloatJ(char** args, int argc): Probe3Float(args, argc) { }
};

struct PrVecFloatME2: Probe3Float {
    void wrElem(size_t idx) {
        double3 mE2 = double3(sp->mE2, idx) * dispScale;
        *pfp++ = mE2.x;
        *pfp++ = mE2.y;
        *pfp++ = mE2.z;
        if (!printedProbe)
            printCell(idx);
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 mE2 = double3(sp->mE2, idx) * dispScale;
        S.x += sign * mE2.x;
        S.y += sign * mE2.y;
        S.z += sign * mE2.z;
        pfp += 3;
    }
    PrVecFloatME2(char** args, int argc): Probe3Float(args, argc) { }
};

struct PrMagFloatH: ProbeFloat {
    void wrElem(size_t idx) {
        double3 H = double3(sp->H, idx) * dispScale;
        H.x /= zd;
        H.y /= zd;
        H.z /= zd;
        *pfp++ = H.length();
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 H = double3(sp->H, idx) * dispScale;
        H.x /= zd;
        H.y /= zd;
        H.z /= zd;
        S.x += H.length();
        pfp++;
    }
    PrMagFloatH(char** args, int argc): ProbeFloat(args, argc) { }
};

struct PrMagFloatE: ProbeFloat {
    void wrElem(size_t idx) {
        double3 E = double3(sp->E, idx) * dispScale;
        *pfp++ = E.length();
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 E = double3(sp->E, idx) * dispScale;
        S.x += E.length();
        pfp++;
    }
    PrMagFloatE(char** args, int argc): ProbeFloat(args, argc) { }
};

struct PrFloatME2: ProbeFloat {
    void wrElem(size_t idx) {
        *pfp++ = sp->mE2.x[idx] * dispScale;
        //*pfp++ = sp->mE2.y[idx];
        //*pfp++ = sp->mE2.z[idx];
        //*pfp++ = sp->mE1.x[idx];
        //*pfp++ = sp->mE1.y[idx];
        //*pfp++ = sp->mE1.z[idx];
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 mE2 = double3(sp->mE2, idx) * dispScale;
        S.x += mE2.x;
        S.y += mE2.y;
        S.z += mE2.z;
        pfp++;
    }
    PrFloatME2(char** args, int argc): ProbeFloat(args, argc) { }
};

class PrFloatSigE: public ProbeFloat {
public:
    void wrElem(size_t idx) {
        *pfp++ = sp->sige[idx] * 1e-8 * dispScale;
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 mH1 = double3(sp->mH1, idx) * dispScale;
        S.x += mH1.x;
        S.y += mH1.y;
        S.z += mH1.z;
        pfp++;
    }
    PrFloatSigE(char** args, int argc): ProbeFloat(args, argc) { }
};

struct PrFloatSigH: ProbeFloat {
    void wrElem(size_t idx) {
        *pfp++ = sp->sigh[idx] * 1e-8 * dispScale;
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 J = double3(sp->J, idx) * dispScale;
        S.x += J.x;
        S.y += J.y;
        S.z += J.z;
        pfp++;
    }
    PrFloatSigH(char** args, int argc): ProbeFloat(args, argc) { }
};

class PrFloatJ: public ProbeFloat {
public:
    void wrElem(size_t idx) {
        double3 J = double3(sp->J, idx) * dispScale;
        *pfp++ = J.length();
    }
    void sumElem(size_t idx, double sign=1.) {
        double3 J = double3(sp->J, idx) * dispScale;
        S.x += J.x;
        S.y += J.y;
        S.z += J.z;
        pfp++;
    }
    PrFloatJ(char** args, int argc): ProbeFloat(args, argc) { }
};

struct PrRGBH: ProbeI32 {
    void wrElem(size_t idx) {
        double3 H = double3(sp->H, idx) * (dispScale / zd);
        uint8_t red = clampu8s(H.x);
        uint8_t gr = clampu8s(H.y);
        uint8_t bl = clampu8s(H.z);
        uint32_t abgr = (255 << 24) + (bl << 16) + (gr <<  8) + red;
        //printf("H=(%g,%g,%g) abgr=0x%08x\n", H.x, H.y, H.z, abgr);
        *pip++ = abgr;
    }
    PrRGBH(char** args, int argc): ProbeI32(args, argc) { }
};

struct PrRGBE: ProbeI32 {
    void wrElem(size_t idx) {
        double3 E = double3(sp->E, idx);
        uint8_t red = clampu8s(E.x*dispScale);
        uint8_t gr = clampu8s(E.y*dispScale);
        uint8_t bl = clampu8s(E.z*dispScale);
        uint32_t abgr = (255 << 24) + (bl << 16) + (gr <<  8) + red;
        *pip++ = abgr;
    }
    PrRGBE(char** args, int argc): ProbeI32(args, argc) { }
};


struct PrFloat1: ProbeFloat {
    virtual double getElem(size_t idx) = 0;
    void wrElem(size_t idx) {
        *pfp++ = getElem(idx) * dispScale;
    }
    void sumElem(size_t idx, double sign=1.) { ; }
    PrFloat1(char** args, int argc): ProbeFloat(args, argc) { }
};

struct PrHx: PrFloat1 {
    double getElem(size_t idx) { return sp->H.x[idx]/zd; }
    PrHx(char** args, int argc): PrFloat1(args, argc) { }
};
struct PrHy: PrFloat1 {
    double getElem(size_t idx) { return sp->H.y[idx]/zd; }
    PrHy(char** args, int argc): PrFloat1(args, argc) { }
};
struct PrHz: PrFloat1 {
    double getElem(size_t idx) { return sp->H.z[idx]/zd; }
    PrHz(char** args, int argc): PrFloat1(args, argc) { }
};
struct PrEx: PrFloat1 {
    double getElem(size_t idx) { return sp->E.x[idx]; }
    PrEx(char** args, int argc): PrFloat1(args, argc) { }
};
struct PrEy: PrFloat1 {
    double getElem(size_t idx) { return sp->E.y[idx]; }
    PrEy(char** args, int argc): PrFloat1(args, argc) { }
};
struct PrEz: PrFloat1 {
    double getElem(size_t idx) { return sp->E.z[idx]; }
    PrEz(char** args, int argc): PrFloat1(args, argc) { }
};

// ----------------------------------------------------------------------------

void Probe::add(char** args, int argc) {
    Probe* p = (Probe*) new ProbeGeneric(args, argc);
    Probe* p2 = NULL;
    Err* fieldNameErr = new Err("Probe: unknown fieldName %c", p->fieldName);

    switch (p->dispType) {
        case 'S':   // summed vector
        case 'L':   // border-line-integrated vector
        case 'V':   // vector
            switch (p->fieldName) {
                case 'H': p2 = new PrVecFloatH(args, argc); break;
                case 'E': p2 = new PrVecFloatE(args, argc); break;
                case 'J': p2 = new PrVecFloatJ(args, argc); break;
                case 'M': p2 = new PrVecFloatME2(args, argc); break;
                //case 'N': p2 = new PrVecFloatMH2(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        case 'M':   // magnitude
            switch (p->fieldName) {
                case 'H': p2 = new PrMagFloatH(args, argc); break;
                case 'E': p2 = new PrMagFloatE(args, argc); break;
                case 'M': p2 = new PrFloatME2(args, argc); break;
                //case 'N': p2 = new PrFloatMH2(args, argc); break;
                case 'S': p2 = new PrFloatSigE(args, argc); break;
                case 'T': p2 = new PrFloatSigH(args, argc); break;
                case 'J': p2 = new PrFloatJ(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        case 'R':   // RGB
            switch (p->fieldName) {
                case 'H': p2 = new PrRGBH(args, argc); break;
                case 'E': p2 = new PrRGBE(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        case 'X':   // X only
            switch (p->fieldName) {
                case 'H': p2 = new PrHx(args, argc); break;
                case 'E': p2 = new PrEx(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        case 'Y':   // Y only
            switch (p->fieldName) {
                case 'H': p2 = new PrHy(args, argc); break;
                case 'E': p2 = new PrEy(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        case 'Z':   // Z only
            switch (p->fieldName) {
                case 'H': p2 = new PrHz(args, argc); break;
                case 'E': p2 = new PrEz(args, argc); break;
                default: throw fieldNameErr;
            }
            break;
        default:
            throw new Err("Probe: unknown dispType %c", p->dispType);
    }

    p2->next = probes;
    probes = p2;
    delete (ProbeGeneric*)p;
    delete fieldNameErr;
}

// ----------------------------------------------------------------------------
// Initialize coefficients in one PML area.
//
// x0 - xn, etc. are cells indices bounding PML area
// Id is loc of adjacent inner cell, where ==ba means same row or col

void OuterSpace::initPMLarea(int x0, int xn, int y0, int yn, int z0, int zn,
                             int3 Id) {

    // pre-calc PML gradients (sH are shifted over 0.5)
    double* sHg = new double[nb];
    double* sEg = new double[nb];
    for (int i = 0; i < nb; i++) {
        double r = double(i) / nb;
        sEg[i] = r*r*r / (4*dt);
        r += 0.5 / nb;
        sHg[i] = r*r*r / (4*dt);
    }

    for (int i = x0; i < xn; i++) {
        // (iac,jac,kac) are adjacent center-cell indices
        int iac = (Id.i == ba) ? i : Id.i;
        // sigma_x for Hx (without the "ep0/2" factor)
        double sHx1 = 0.;
        double sEx1 = 0.;
        if (i < 0) {
            sHx1 = sHg[-i-1];
            sEx1 = sEg[-i-1];
        } else if (i >= N.i) {
            sHx1 = sHg[i-N.i];
            sEx1 = sEg[i-N.i];
        }
        for (int j = y0; j < yn; j++) {
            int jac = (Id.j == ba) ? j : Id.j;
            double sHy1 = 0.;
            double sEy1 = 0.;
            if (j < 0) {
                sHy1 = sHg[-j-1];
                sEy1 = sEg[-j-1];
            } else if (j >= N.j) {
                sHy1 = sHg[j-N.j];
                sEy1 = sEg[j-N.j];
            }
            for (int k = z0; k < zn; k++) {
                int kac = (Id.k == ba) ? k : Id.k;
                double sHz = 0.;
                double sEz = 0.;
                if (k < 0) {
                    sHz = sHg[-k-1];
                    sEz = sEg[-k-1];
                } else if (k >= N.k) {
                    sHz = sHg[k-N.k];
                    sEz = sEg[k-N.k];
                }
                // get material parms from adjacent center-cell
                size_t idxa = idxcell(iac,jac,kac);
                double sHx = sHx1;
                double sHy = sHy1;
                if (sigh[idxa] > 0) {
                    double stma = sigh[idxa] / (2*mu0*mur[idxa]);
                    // sHx corresponds to allocate()'s stm/dt
                    sHx += stma;
                    sHy += stma;
                    sHz += stma;
                }
                double sEx = sEx1;
                double sEy = sEy1;
                if (sige[idxa] > 0) {
                    double stea = sige[idxa] / (2*ep0*epr[idxa]);
                    sEx += stea;
                    sEy += stea;
                    sEz += stea;
                }

                // H update coefficients, mH2 includes "/dx"
                size_t idx = idxcell(i,j,k);
                double A = sHy + sHz;
                double B = sHy * sHz * dt;
                double R = 1 / (1/dt + A + B);
                mH1.x[idx] = R * (1/dt - A - B);
                mH2.x[idx] = -R * c0 / (dx * mur[idxa]);

                A = sHx + sHz;
                B = sHx * sHz * dt;
                R = 1 / (1/dt + A + B);
                mH1.y[idx] = R * (1/dt - A - B);
                mH2.y[idx] = -R * c0 / (dx * mur[idxa]);

                A = sHx + sHy;
                B = sHx * sHy * dt;
                R = 1 / (1/dt + A + B);
                mH1.z[idx] = R * (1/dt - A - B);
                mH2.z[idx] = -R * c0 / (dx * mur[idxa]);

                // E update coefficients (= D with 1/ep on H terms, 2 & 3)
                A = sEy + sEz;
                B = sEy * sEz * dt;
                R = 1 / (1/dt + A + B);
                mE1.x[idx] = R * (1/dt - A - B);
                mE2.x[idx] = R * c0 / (dx * epr[idxa]);

                A = sEx + sEz;
                B = sEx * sEz * dt;
                R = 1 / (1/dt + A + B);
                mE1.y[idx] = R * (1/dt - A - B);
                mE2.y[idx] = R * c0 / (dx * epr[idxa]);

                A = sEx + sEy;
                B = sEx * sEy * dt;
                R = 1 / (1/dt + A + B);
                mE1.z[idx] = R * (1/dt - A - B);
                mE2.z[idx] = R * c0 / (dx * epr[idxa]);
            }
        }
    }
    delete[] sHg;
    delete[] sEg;
}

// ----------------------------------------------------------------------------
// Integrate E field to get voltage relative to origin. Also prints error.
void Space::integVolts() {
    volt[0] = 0.;
    for (int k = 0; k < N.k; k++) {
        for (int j = 0; j < N.j; j++) {
            for (int i = 0; i < N.i; i++) {
                size_t idx = idxcell(i,j,k);
                double v = volt[idx];
                double dvx = volt[idx+ 1] - v - E.x[idx]*dx;
                double dvy = volt[idx+J1] - v - E.y[idx]*dx;
                double dvz = volt[idx+k1] - v - E.z[idx]*dx;
                printf("[%d,%d,%d].volt=% 11.4g dvx=% 11.4g dvy=% 11.4g"
                       " dvz=% 11.4g\n", i,j,k, v, dvx, dvy, dvz);
                volt[idx+ 1] = v + E.x[idx]*dx;
                volt[idx+J1] = v + E.y[idx]*dx;
                volt[idx+k1] = v + E.z[idx]*dx;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Print E & H values, good for use with mobile point probe while paused.

void Probe::printCell(size_t idx) {
    printedProbe = true;
    double Hm = double3(sp->H, idx).length();
    double Em = double3(sp->E, idx).length();
    if (Hm == 0. && Em == 0.) {
        if (printing) {
            printing = false;
            printf("...\n");
        }
    } else {
        printing = true;
        sp->printCell(idx);
    }
}

// ----------------------------------------------------------------------------

void Space::printCell(size_t idx) {
    double3 Hi = double3(H, idx);
    double3 Ei = double3(E, idx);
    double Hm = Hi.length();
    double Em = Ei.length();
    int nx = Ncb.i;
    int k = (int)(idx / k1);
    int j = (int)((idx - k*k1) / nx);
    int i = (int)(idx - (k*k1+j*nx));
    printf("H[%2d,%2d,%2d]=(% 7.3f,% 7.3f,% 7.3f)=% 7.3f "
        "E=(% 7.3f,% 7.3f,% 7.3f)=% 7.3f\n",
        i, j, k, Hi.x/zd, Hi.y/zd, Hi.z/zd, Hm/zd, Ei.x, Ei.y, Ei.z, Em);
}

// ----------------------------------------------------------------------------
// Print H and E arrays, skipping all-zero lines.

void Space::dumpHE(int vstep) {
    bool printing = true;
    for (int i = 0; i < N.i; i++) {
        for (int j = 0; j < N.j; j++) {
            for (int k = zo0; k < zo0+nzo; k++) {
                double3 H = double3(sp->H, sp->idxcell(i,j,k));
                double3 E = double3(sp->E, sp->idxcell(i,j,k));
                if (H.length() == 0 && E.length() == 0) {
                    if (printing) {
                        printing = false;
                        printf("...\n");
                    }
                } else {
                    printing = true;
                    int f = 6;  // fraction digits to display
                    int w = f+7;
                    printf("%d H[%d,%d,%d]=[% *.*g, % *.*g, % *.*g]"
                       " E=[% *.*g, % *.*g, % *.*g]\n", vstep, i, j, k,
                       w, f, H.x/zd, w, f, H.y/zd, w, f, H.z/zd,
                       w, f, E.x,    w, f, E.y,    w, f, E.z);
                }
            }
        }
    }

    // dump any subspaces
    for (SubSpace* c = children; c; c = c->nextSib)
        c->dumpHE(vstep);
}

// ----------------------------------------------------------------------------
// Simulation space ctors and dtors.

Space* Space::spaces;

void Space::insert() {
    // default values
    zo0 = 12;
    nzo = 1;
    children = NULL;
    next = spaces;
    spaces = this;
    verbose = 0;
}

SubSpace::SubSpace(char** args, int argc): Space(args, argc-1) {
    parent = find(args[9]);
    dx = parent->dx / 2;
    nextSib = parent->children;
    parent->children = this;
}

OuterSpace::OuterSpace(char** args, int argc): Space(args, argc-3) {
    dx =      atof(args[9]) * unit;
    //conductBorder = atoi(args[10]);
    conductBorder = 0;
    ncb =       atoi(args[11]) + 1;
}

Space::~Space() {
    for (SubSpace* c = children; c; c = c->nextSib)
        delete c;
    delete[] mur;
    delete[] epr;
    delete[] sige;
    delete[] sigh;
}

void Space::deleteAll() {
    if (spaces) {
        Space* sNext;
        for (Space* s = spaces; s; s = sNext) {
            sNext = (Space*)s->next;
            delete s;
        }
        spaces = 0;
    }
}

Space* Space::find(const char* name) {
    Space* s = spaces;
    for ( ; s; s = s->next)
        if (strcmp(s->name, name) == 0)
            break;
    if (!s)
        throw new Err("space %s not found", name);
    return s;
}

void SubSpace::initICoords() {
    // trim block to fit within this space on all sides,
    // and also compute grid coordinates in this space
    for (Space* s = spaces; s; s = s->next)
        s->initICoords1(0);
}

// ----------------------------------------------------------------------------
// Initialize before setting up a FDTD simulation.

void setupInit() {
    // initialize lists
    Material::materials = NULL;
    MatBlock::matBlks = NULL;
    Source::sources = NULL;
    Probe::probes = NULL;
    Space::spaces = NULL;
}

// ----------------------------------------------------------------------------
// Initialize for a FDTD simulation run.

void OuterSpace::runInit() {
    printf("runInit start\n");

    // how far objects may go outside Fields box, m
    extend = (ncb-1) * dx;
    //goffs = -dx / 2;
    goffs = 0;
    sp = this;
    // clip and translate to Fields box, and determine grid coordinates
    //MatBlock::initICoords(extend);
    MatBlock::initICoords(0);
    Source::initICoords();
    Probe::initICoords(extend);
    SubSpace::initICoords();

    allocate();
    sp = this;

    // pre-roll source currents, conductions
    Source::preInjectAll();

    printf("runInit done.\n");
}

// ----------------------------------------------------------------------------

void Space::allocate() {

    coord2grid(&Bs, &Is);
    coord2grid(&Be, &Ie);
    N = Ie - Is;
    if (N.i < 1 || N.j < 1 || N.k < 1)
        throw new Err("%s bounds too small?", name);
    if (N.i > 1000 || N.j > 1000 || N.k > 1000)
        throw new Err("block space size exceeded");
    
    // original grid dimensions
    Norig = N;

    // round up the fastest axis width so that the total including PML (nb)
    // is an even multiple of the vector width (wv)
    N.i = (N.i + 2*nb + wv - 1) / wv * wv - 2*nb;

    // Ncb includes the PML boundary
    Ncb = N + 2*ncb;
    ncbv = ncb - 1 + wv;

    // entire fastest axis includes at least one PML vector on either side
    Nv = Ncb;
    Nv.i = max(N.i+2*wv, Nv.i);
    nCells = Nv.k * Nv.j * Nv.i;

    // k and j strides
    k1 = Nv.j * Nv.i;
    J1 = Nv.i;

    // initialize field cells
    E.init(nCells);
    H.init(nCells);
    mur = new double[nCells];
    epr = new double[nCells];
    sige = new double[nCells];
    sigh = new double[nCells];
    mE1.init(nCells);
    mE2.init(nCells);
    mH1.init(nCells);
    mH2.init(nCells);
    J.init(nCells);
    volt = new double[nCells];
    cellType = new CellType[nCells];

    // vectorized versions
    vk1 = k1 / wv;
    vJ1 = J1 / wv;
    vE = E;
    vH = H;
    vmE1 = mE1;
    vmE2 = mE2;
    vmH1 = mH1;
    vmH2 = mH2;
    vJ = J;

    for (size_t i = 0; i < nCells; i++) {
        E.x[i] = E.y[i] = E.z[i] = 0.;
        H.x[i] = H.y[i] = H.z[i] = 0.;
        mur[i] = 1.;
        epr[i] = 1.;
        sigh[i] = 0.;
        sige[i] = 0.;
        J.x[i] = J.y[i] = J.z[i] = 0.;
        cellType[i] = NORMAL;
    }

    placeInit();
    place();
    placeAll();
    placeFinish();

    // compute coefficients
    c0 = 1./sqrt(mu0*ep0);
    z0 = sqrt(mu0/ep0);
#ifdef DISPLAY_H_V_PER_M
    zd = 1.;    // 1 for displaying H in V/m
#else
    zd = z0;    // z0 for displaying H in F/m
#endif
    dt = 0.5 * dx / c0;
    //dt = (1./sqrt(3.)) * dx / c0;   // Gedney recommendation, but unstable?
    //dt = (1./4.) * dx / c0;        // test
    double K = dt/dx;
    for (size_t i = 0; i < nCells; i++) {
        double stm = sigh[i]*dt / (2*mu0*mur[i]);
        double ste = sige[i]*dt / (2*ep0*epr[i]);
        mH1.x[i] = mH1.y[i] = mH1.z[i] = (1. - stm) / (1. + stm);
        mE1.x[i] = mE1.y[i] = mE1.z[i] = (1. - ste) / (1. + ste);
        mH2.x[i] = mH2.y[i] = mH2.z[i] = -K*z0 / ((1. + stm) * mu0*mur[i]);
        mE2.x[i] = mE2.y[i] = mE2.z[i] = K / ((1. + ste) * z0*ep0*epr[i]);
    }

    // allocate any subspaces
    for (SubSpace* c = children; c; c = c->nextSib)
        c->allocate();
}

// ----------------------------------------------------------------------------

void OuterSpace::allocate() {
    Space::allocate();
    printf("FDTD %dx%dx%d, (%dx%dx%d), ncb=%d dx=%g, dt=%g\n",
        Ncb.i, Ncb.j, Ncb.k, N.i, N.j, N.k, ncb, dx, dt);

    nb = ncb - 1; // allow for 1 cell Dirichlet boundary on corresp Cells
    ba = -nb;
    Nb = Ncb - 2;
    M = Ncb - ncb - 1;
    int3 K = N - 1;
    if (ncb > 1) {
        // allocate PML areas and compute their coefficients
        initPMLarea(ba, 0,    ba, M.j, ba, M.k, int3(0,  ba, ba));
        initPMLarea(N.i, M.i, ba, M.j, ba, M.k, int3(K.i,ba, ba));
        initPMLarea(0, N.i, ba,    0,  ba, M.k, int3(ba, 0,  ba));
        initPMLarea(0, N.i, N.j, M.j,  ba, M.k, int3(ba, K.j,ba));
        initPMLarea(0, N.i, 0,   N.j,  ba,   0, int3(ba, ba, 0));
        initPMLarea(0, N.i, 0,   N.j, N.k, M.k, int3(ba, ba, K.k));
    }
}

// ----------------------------------------------------------------------------
// Step H field of FDTD simulation.

void Space::stepH() {
    // step H field
    for (int k = ba; k < M.k; k++) {
        for (int j = ba; j < M.j; j++) {
            size_t idx = idxcell(ba,j,k);
            size_t ic = idx / wv;
            int i = ba;
            int ivEnd = M.i - wv + 1;
            // main H loop, vectorized
            for ( ; i < ivEnd; i += wv) {
                vH.x[ic] = vmH1.x[ic] * vH.x[ic] +
                           vmH2.x[ic] * ((vE.z[ic+vJ1] - vE.z[ic]) -
                                         (vE.y[ic+vk1] - vE.y[ic]));
                vH.y[ic] = vmH1.y[ic] * vH.y[ic] +
                           vmH2.y[ic] * ((vE.x[ic+vk1] - vE.x[ic]) -
                                         (v(E.z,idx+1) - vE.z[ic]));
                vH.z[ic] = vmH1.z[ic] * vH.z[ic] +
                           vmH2.z[ic] * ((v(E.y,idx+1) - vE.y[ic]) -
                                         (vE.x[ic+vJ1] - vE.x[ic]));
                idx += wv;
                ic++;
            }
            // do remainder as scalars
            for ( ; i < M.i; i++) {
                H.x[idx] = mH1.x[idx] * H.x[idx] +
                           mH2.x[idx] * ((E.z[idx+J1] - E.z[idx]) -
                                         (E.y[idx+k1] - E.y[idx]));
                H.y[idx] = mH1.y[idx] * H.y[idx] +
                           mH2.y[idx] * ((E.x[idx+k1] - E.x[idx]) -
                                         (E.z[idx+1] - E.z[idx]));
                H.z[idx] = mH1.z[idx] * H.z[idx] +
                           mH2.z[idx] * ((E.y[idx+1] - E.y[idx]) -
                                         (E.x[idx+J1] - E.x[idx]));
                idx++;
            }
        }
    }
 
    // optionally print field states
    if (verbose >= 2) {
        const char* probeName = "9E_mobile";
        Probe* probe = Probe::find(probeName);
        if (probe) {
            int i = probe->Is.i;
            int j = probe->Is.j;
            int k = probe->Is.k;
            size_t idx1 = idxcell(i, j, k);
            printf("%d %s: idx=%ld H[%d,%d,%d]=[% 9.3g, % 9.3g, % 9.3g],",
                   step, probeName, idx1, i, j, k,
                   H.x[idx1]/zd, H.y[idx1]/zd, H.z[idx1]/zd);
            printf(" E=[% 9.3g, % 9.3g, % 9.3g]\n",
                   E.x[idx1], E.y[idx1], E.z[idx1]);
        } else {
            printf("(StepH verbose 2: probe %s not found)\n", probeName);
        }
    }
    if (verbose == 4) {
        // (-2.5, -1.4, -0.2) fields=(-2.2, -1.5, -0.0) --> (-3,1,-2)
        size_t idx = idxcell(2,6,6);
        printf("VERBOSE4: %g\n", step*dt);
        printCell(idx);
    }

    // step any subspaces
    for (SubSpace* c = children; c; c = c->nextSib)
        c->stepH();

    if (verbose == 3) {
        printf("%d, post H: ------------------------------------------\n",
               step+1);
        dumpHE(step+1);
    }
}

// ----------------------------------------------------------------------------
// Step border H field of FDTD simulation.

void OuterSpace::stepH() {
    Space::stepH();

    // conductive outer 1-cell border: tangential E and normal H are zero
    if (conductBorder) {
        for (int j = -ncb; j < M.j; j++) {
            size_t idx = idxcell(-ncb,j,-ncb);
            size_t ic = idx / wv;
            int i = -ncb;
            int ivEnd = M.i - wv + 1;
            if (step == 9 /* && (idx > 2922-65 && idx < 2922+65) */) {
                printf("");
            }
            // step all H[i,j,0], but with z terms removed, vectorized
            for ( ; i < ivEnd; i += wv) {
                vH.x[ic] -= vmH2.x[ic] * (vE.y[ic+vk1] - vE.y[ic]);
                vH.y[ic] += vmH2.y[ic] * (vE.x[ic+vk1] - vE.x[ic]);
                // want H[i,j,M.k], but it reaches beyond border
                idx += wv;
                ic++;
            }
            // remainder as scalars
            for ( ; i < M.i; i++) {
                H.x[idx] -= mH2.x[idx] * (E.y[idx+k1] - E.y[idx]);
                H.y[idx] += mH2.y[idx] * (E.x[idx+k1] - E.x[idx]);
                idx++;
            }
        }
        for (int k = -ncb; k < M.k; k++) {
            size_t idx = idxcell(ba,-ncb,k);
            size_t ic = idx / wv;
            // step all H[i,0,k], but with y terms removed, vectorized
            int i = ba;
            int ivEnd = M.i - wv + 1;
            for ( ; i < ivEnd; i += wv) {
                vH.x[ic] += vmH2.x[ic] * (vE.z[ic+vJ1] - vE.z[ic]);
                vH.z[ic] -= vmH2.z[ic] * (vE.x[ic+vJ1] - vE.x[ic]);
                idx += wv;
                ic++;
            }
            // remainder as scalars
            for ( ; i < M.i; i++) {
                H.x[idx] += mH2.x[idx] * (E.z[idx+J1] - E.z[idx]);
                H.z[idx] -= mH2.z[idx] * (E.x[idx+J1] - E.x[idx]);
                idx++;
            }
        }
        for (int j = ba; j < M.j; j++) {
            size_t idx = idxcell(-ncb,j,ba);
            for (int k = ba; k < M.k; k++) {
                // step all H[0,j,k], but with x terms removed
                H.y[idx] -= mH2.y[idx] * (E.z[idx+1] - E.z[idx]);
                H.z[idx] += mH2.z[idx] * (E.y[idx+1] - E.y[idx]);
                idx += k1;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Step E field of FDTD simulation.

void Space::stepE() {
    // pre-Estep update source currents, conductions
    Source::preInjectAll();

    // step E field
     for (int k = ba; k < M.k; k++) {
        for (int j = ba; j < M.j; j++) {
            size_t idx = idxcell(ba,j,k);
            size_t ic = idx / wv;
            int i = ba;
            int ivEnd = M.i - wv + 1;
            // main E loop, vectorized
            for ( ; i < ivEnd; i += wv) {
                vE.x[ic] = vmE1.x[ic] * vE.x[ic] +
                           vmE2.x[ic] * ((vH.z[ic] - vH.z[ic-vJ1]) -
                                         (vH.y[ic] - vH.y[ic-vk1]) - vJ.x[ic]);
                vE.y[ic] = vmE1.y[ic] * vE.y[ic] +
                           vmE2.y[ic] * ((vH.x[ic] - vH.x[ic-vk1]) -
                                         (vH.z[ic] - v(H.z,idx-1)) - vJ.y[ic]);
                vE.z[ic] = vmE1.z[ic] * vE.z[ic] +
                           vmE2.z[ic] * ((vH.y[ic] - v(H.y,idx-1)) -
                                         (vH.x[ic] - vH.x[ic-vJ1]) - vJ.z[ic]);
                idx += wv;
                ic++;
            }
            // do remainder as scalars
            for ( ; i < M.i; i++) {
                E.x[idx] = mE1.x[idx] * E.x[idx] +
                           mE2.x[idx] * ((H.z[idx] - H.z[idx-J1]) -
                                         (H.y[idx] - H.y[idx-k1]) - J.x[idx]);
                E.y[idx] = mE1.y[idx] * E.y[idx] +
                           mE2.y[idx] * ((H.x[idx] - H.x[idx-k1]) -
                                         (H.z[idx] - H.z[idx-1]) - J.y[idx]);
                E.z[idx] = mE1.z[idx] * E.z[idx] +
                           mE2.z[idx] * ((H.y[idx] - H.y[idx-1]) -
                                         (H.x[idx] - H.x[idx-J1]) - J.z[idx]);
                idx++;
            }
        }
    }
    // post-Estep update source currents, conductions
    Source::postInjectAll();

    if (verbose == 4) {
        printf("%d, post E & src:\n", step);
        dumpHE(step);
    }

    // step any subspaces
    for (SubSpace* c = children; c; c = c->nextSib)
        c->stepE();
}

// ----------------------------------------------------------------------------
// Step border E field of FDTD simulation.

void OuterSpace::stepE() {
    Space::stepE();

    // conductive outer 1-cell border: tangential E and normal H are zero
    if (conductBorder) {
        int3 K = Ncb - ncb;
        for (int j = ba; j < K.j; j++) {
            size_t idx = idxcell(ba,j,-ncb);
            size_t ic = idx / wv;
            size_t idxh = idx + (Ncb.k-1)*vk1;
            size_t ich = idxh / wv;
            // step all E[i,j,0] and E[i,j,h], but with x,y terms removed
            int i = ba;
            int ivEnd = K.i - wv + 1;
            for ( ; i < ivEnd; i += wv) {
                vE.z[ic] += vmE2.z[ic] * ((vH.y[ic] - *(vdbl*)&H.y[idx-1]) -
                                          (vH.x[ic] - vH.x[ic-vJ1]));
                vE.z[ich] += vmE2.z[ich] * ((vH.y[ich] -*(vdbl*)&H.y[idxh-1]) -
                                            (vH.x[ich] - vH.x[ich-vJ1]));
                idx += wv;
                ic++;
                idxh += wv;
                ich++;
            }
            // remainder as scalars
            for ( ; i < K.i; i++) {
                E.z[idx] += mE2.z[idx] * ((H.y[idx] - H.y[idx-1]) -
                                          (H.x[idx] - H.x[idx-J1]));
                E.z[idxh] += mE2.z[idxh] * ((H.y[idxh] - H.y[idxh-1]) -
                                            (H.x[idxh] - H.x[idxh-J1]));
                idx++;
                idxh++;
            }
       }
        for (int k = ba; k < M.k; k++) {
            size_t idx = idxcell(ba,-ncb,k);
            size_t ic = idx / wv;
            size_t idxh = idx + (Ncb.j-1)*vJ1;
            size_t ich = idxh / wv;
            // step all E[i,0,k] and E[i,h,k], but with x,z terms removed
            int i = ba;
            int ivEnd = K.i - wv + 1;
            for ( ; i < ivEnd; i += wv) {
                vE.y[ic] += vmE2.y[ic] * ((vH.x[ic] - vH.x[ic-vk1]) -
                                          (vH.z[ic] - *(vdbl*)&H.z[idx-1]));
                vE.y[ich] += vmE2.y[ich] * ((vH.x[ich] - vH.x[ich-vk1]) -
                                            (vH.z[ich] -*(vdbl*)&H.z[idxh-1]));
                idx += wv;
                ic++;
                idxh += wv;
                ich++;
            }
            // remainder as scalars
            for ( ; i < K.i; i++) {
                E.y[idx] += mE2.y[idx] * ((H.x[idx] - H.x[idx-k1]) -
                                          (H.z[idx] - H.z[idx-1]));
                E.y[idxh] += mE2.y[idxh] * ((H.x[idxh] - H.x[idxh-k1]) -
                                            (H.z[idxh] - H.z[idxh-1]));
                idx++;
                idxh++;
            }
        }
        for (int j = ba; j < M.j; j++) {
            size_t idx = idxcell(-ncb,j,ba);
            for (int k = ba; k < M.k; k++) {
                // step all E[0,j,k] and E[h,j,k], but with y,z terms removed
                size_t idxh = idx + Ncb.i-1;
                E.x[idx] += mE2.x[idx] * ((H.z[idx] - H.z[idx-J1]) -
                                          (H.y[idx] - H.y[idx-k1]));
                E.x[idxh] += mE2.x[idxh] * ((H.z[idxh] - H.z[idxh-J1]) -
                                            (H.y[idxh] - H.y[idxh-k1]));
                idx += k1;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// End simulation run-- free memory.

void fieldEnd() {
    printf("fieldEnd: freeing memory\n");
    Material::deleteAll();
    MatBlock::deleteAll();
    Source::deleteAll();
    Probe::deleteAll();
    Space::deleteAll();
}
