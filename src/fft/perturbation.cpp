//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file perturbation.cpp
//  \brief implementation of functions in class PerturbationGenerator

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>     // mt19937, normal_distribution, uniform_real_distribution
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"
#include "athena_fft.hpp"
#include "perturbation.hpp"

PerturbationBlock::PerturbationBlock(MeshBlock *pmb) :
    vec(3, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    ptbvar(pmb, &vec, nullptr, empty_flux, false) {
  ptbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ptbvar);
  scal.InitWithShallowSlice(vec,4,0,1);
}

// destructor
PerturbationBlock::~PerturbationBlock() {
  vec.DeleteAthenaArray();
}

//----------------------------------------------------------------------------------------
//! \fn PerturbationGenerator::PerturbationGenerator(Mesh *pm, ParameterInput *pin)
//! \brief PerturbationGenerator constructor
//!
//! \note
//! By default, this generates a vector field

PerturbationGenerator::PerturbationGenerator(Mesh *pm, ParameterInput *pin) :
    FFTDriver(pm, pin),
    rseed(pin->GetOrAddInteger("perturbation", "rseed", -1)), // seed for RNG
    // cut-off wavenumbers, low and high:
    nlow(pin->GetOrAddInteger("perturbation", "nlow", 0)),
    nhigh(pin->GetOrAddInteger("perturbation", "nhigh", pm->mesh_size.nx1/2)),
    f_shear(pin->GetOrAddReal("perturbation", "f_shear", -1)), // ratio of shear component
    expo(pin->GetOrAddReal("perturbation", "expo", 2)) {// power-law exponent
  if (f_shear > 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PerturbationGenerator" << std::endl
        << "The ratio between shear and compressible components should be less than one"
        << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
#ifndef FFT
  std::stringstream msg;
  msg << "### FATAL ERROR in PerturbationGenerator" << std::endl
      << "non zero PerturbationGenerator is called without FFT!" << std::endl;
  ATHENA_ERROR(msg);
  return;
#endif

  InitializeFFTBlock(true);
  // note, pmy_fb won't be defined until InitializeFFTBlock is called:
  dvol = pmy_fb->dx1*pmy_fb->dx2*pmy_fb->dx3;
  QuickCreatePlan();

  fv_ = new std::complex<Real>*[3];
  fv_sh_ = new std::complex<Real>*[3];
  fv_co_ = new std::complex<Real>*[3];
  for (int nv=0; nv<3; nv++) {
    fv_[nv] = new std::complex<Real>[pmy_fb->cnt_];
    fv_sh_[nv] = new std::complex<Real>[pmy_fb->cnt_];
    fv_co_[nv] = new std::complex<Real>[pmy_fb->cnt_];
  }

  // initialize MT19937 random number generator
  if (rseed < 0) {
    std::random_device device;
    rseed = static_cast<std::int64_t>(device());
  } else {
    // If rseed is specified with a non-negative value,
    // PS is generated with a global random number sequence.
    // This would make perturbation identical irrespective of number of MPI ranks,
    // but the cost of the PowerSpectrum() function call is huge.
    // Not recommended with turb_flag = 3 or turb_flag = with small dtdrive
    global_ps_ = true;
  }
  rng_generator.seed(rseed);

  my_ptblocks.NewAthenaArray(pm->nblocal);

  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    my_ptblocks(nb) = new PerturbationBlock(pmb);
  }
}

// destructor
PerturbationGenerator::~PerturbationGenerator() {
  for (int b=0; b<pmy_mesh_->nblocal; ++b)
    delete my_ptblocks(b);

  for (int nv=0; nv<3; nv++) {
    delete [] fv_[nv];
    delete [] fv_sh_[nv];
    delete [] fv_co_[nv];
  }
  delete [] fv_;
  delete [] fv_sh_;
  delete [] fv_co_;
}

//----------------------------------------------------------------------------------------
//! \fn PerturbationGenerator::SetBoundary(Mesh *pm, ParameterInput *pin)
//! \brief populate ghost zones with proper boundary conditions

void PerturbationGenerator::SetBoundary() {
  Mesh *pm = pmy_mesh_;

  for (int nb=0; nb<pm->nblocal; ++nb) {
    PerturbationBlock *ptblock = my_ptblocks(nb);
    ptblock->ptbvar.SetupPersistentMPI();
  }

  for (int nb=0; nb<pm->nblocal; ++nb) {
    PerturbationBlock *ptblock = my_ptblocks(nb);
    ptblock->ptbvar.StartReceiving(BoundaryCommSubset::mesh_init);
  }

  for (int nb=0; nb<pm->nblocal; ++nb) {
    PerturbationBlock *ptblock = my_ptblocks(nb);
    ptblock->ptbvar.SendBoundaryBuffers();
  }

  for (int nb=0; nb<pm->nblocal; ++nb) {
    PerturbationBlock *ptblock = my_ptblocks(nb);
    ptblock->ptbvar.ReceiveAndSetBoundariesWithWait();
  }

  for (int nb=0; nb<pm->nblocal; ++nb) {
    PerturbationBlock *ptblock = my_ptblocks(nb);
    ptblock->ptbvar.ClearBoundary(BoundaryCommSubset::mesh_init);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::Generate()
//! \brief Generate an isotropic vector pertubation.

void PerturbationGenerator::GenerateVector() {
  GenerateVector(fv_);
}

void PerturbationGenerator::GenerateVector(std::complex<Real> **fv) {
  Mesh *pm = pmy_mesh_;
  FFTBlock *pfb = pmy_fb;

  for (int nv=0; nv<3; nv++) {
    std::complex<Real> *fv1 = fv[nv];
    PowerSpectrum(fv1);
  }
  if (f_shear >= 0) Project(fv, f_shear);
}

void PerturbationGenerator::AssignVector() {
  AssignVector(fv_);
}

void PerturbationGenerator::AssignVector(std::complex<Real> **fv) {
  Mesh *pm = pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_;

  for (int nv=0; nv<3; nv++) {
    for (int kidx=0; kidx<pfb->cnt_; kidx++) pfb->in_[kidx] = fv[nv][kidx];
    pfb->Execute(plan);

    for (int nb=0; nb<pm->nblocal; ++nb) {
      MeshBlock *pmb = pm->my_blocks(nb);
      PerturbationBlock *ptb = my_ptblocks(nb);
      AthenaArray<Real> vec1;
      vec1.InitWithShallowSlice(ptb->vec, 4, nv, 1);
      pfb->RetrieveResult(vec1, 0, NGHOST, pmb->loc, pmb->block_size);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::Generate()
//! \brief Generate a scalar perturbation
void PerturbationGenerator::GenerateScalar() {
  GenerateScalar(fv_[0]);
}

void PerturbationGenerator::GenerateScalar(std::complex<Real> *fv) {
  PowerSpectrum(fv);
}

void PerturbationGenerator::AssignScalar() {
  AssignScalar(fv_[0]);
}

void PerturbationGenerator::AssignScalar(std::complex<Real> *fv) {
  Mesh *pm = pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_;

  for (int kidx=0; kidx<pfb->cnt_; kidx++) pfb->in_[kidx] = fv[kidx];
  pfb->Execute(plan);

  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    PerturbationBlock *ptb = my_ptblocks(nb);
    pfb->RetrieveResult(ptb->scal, 0, NGHOST, pmb->loc, pmb->block_size);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::GetVector()
//! \brief accessor

AthenaArray<Real> PerturbationGenerator::GetVector(int nb) {
  return my_ptblocks(nb)->vec;
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::GetScalar(Real dt)
//! \brief accessor

AthenaArray<Real> PerturbationGenerator::GetScalar(int nb) {
  return my_ptblocks(nb)->scal;
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::PowerSpectrum(std::complex<Real> *amp)
//! \brief Generate Power spectrum in Fourier space with power-law

void PerturbationGenerator::PowerSpectrum(std::complex<Real> *amp) {
  Real pcoeff;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int kNx1 = pfb->kNx[0], kNx2 = pfb->kNx[1], kNx3 = pfb->kNx[2];
  int knx1 = pfb->knx[0], knx2 = pfb->knx[1], knx3 = pfb->knx[2];
  int kdisp1 = pfb->kdisp[0], kdisp2 = pfb->kdisp[1], kdisp3 = pfb->kdisp[2];

  std::normal_distribution<Real> ndist(0.0,1.0); // standard normal distribution
  std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)

  // set random amplitudes with gaussian deviation
  // loop over entire Mesh
  if (global_ps_) {
    for (int gk=0; gk<kNx3; gk++) {
      for (int gj=0; gj<kNx2; gj++) {
        for (int gi=0; gi<kNx1; gi++) {
          std::int64_t nx = GetKcomp(gi, 0, kNx1);
          std::int64_t ny = GetKcomp(gj, 0, kNx2);
          std::int64_t nz = GetKcomp(gk, 0, kNx3);
          Real nmag = std::sqrt(nx*nx+ny*ny+nz*nz);
          int k = gk - kdisp3;
          int j = gj - kdisp2;
          int i = gi - kdisp1;
          // Draw random number only in the cutoff range.
          // This ensures the velocity field realization is independent
          // of the grid resolution.
          if ((nmag > nlow) && (nmag < nhigh)) {
            if ((k >= 0) && (k < knx3) &&
                (j >= 0) && (j < knx2) &&
                (i >= 0) && (i < knx1)) {
              std::int64_t kidx = pfb->GetIndex(i,j,k,idx);
              Real A = ndist(rng_generator);
              Real ph = udist(rng_generator)*TWO_PI;
              amp[kidx] = A*std::complex<Real>(std::cos(ph), std::sin(ph));
            } else { // if it is not in FFTBlock, just burn unused random numbers
              Real A = ndist(rng_generator);
              Real ph = udist(rng_generator)*TWO_PI;
            }
          }
        }
      }
    }
  }

  // set power spectrum: only power-law

  // find the reference 2PI/L along the longest axis
  Real dkx = pfb->dkx[0];
  if (knx2>1) dkx = dkx < pfb->dkx[1] ? dkx : pfb->dkx[1];
  if (knx3>2) dkx = dkx < pfb->dkx[2] ? dkx : pfb->dkx[2];

  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        std::int64_t nx = GetKcomp(i,pfb->kdisp[0],pfb->kNx[0]);
        std::int64_t ny = GetKcomp(j,pfb->kdisp[1],pfb->kNx[1]);
        std::int64_t nz = GetKcomp(k,pfb->kdisp[2],pfb->kNx[2]);
        Real nmag = std::sqrt(nx*nx+ny*ny+nz*nz);
        Real kx = nx*pfb->dkx[0];
        Real ky = ny*pfb->dkx[1];
        Real kz = nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);

        std::int64_t gidx = pfb->GetGlobalIndex(i,j,k);

        if (gidx == 0) {
          pcoeff = 0.0;
        } else {
          if ((kmag/dkx > nlow) && (kmag/dkx < nhigh)) {
            pcoeff = 1.0/std::pow(kmag,(expo+2.0)/2.0);
          } else {
            pcoeff = 0.0;
          }
        }
        std::int64_t kidx=pfb->GetIndex(i,j,k,idx);

        if (global_ps_) {
          amp[kidx] *= pcoeff;
        } else {
          Real A = ndist(rng_generator);
          Real ph = udist(rng_generator)*TWO_PI;
          amp[kidx] = pcoeff*A*std::complex<Real>(std::cos(ph), std::sin(ph));
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::Project(std::complex<Real> **fv, Real f_shear)
//! \brief calculate velocity field with a given ratio of shear to comp.

void PerturbationGenerator::Project(std::complex<Real> **fv, Real f_shear) {
  FFTBlock *pfb = pmy_fb;
  Project(fv, fv_sh_, fv_co_);
  for (int nv=0; nv<3; nv++) {
    for (int kidx=0; kidx<pfb->cnt_; kidx++) {
      fv[nv][kidx] = (1-f_shear)*fv_co_[nv][kidx] + f_shear*fv_sh_[nv][kidx];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::Project(std::complex<Real> **fv,
//!                                    std::complex<Real> **fv_sh,
//!                                    std::complex<Real> **fv_co)
//! \brief calculates shear and compressible components
void PerturbationGenerator::Project(std::complex<Real> **fv, std::complex<Real> **fv_sh,
                               std::complex<Real> **fv_co) {
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int knx1 = pfb->knx[0], knx2 = pfb->knx[1], knx3 = pfb->knx[2];

  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        // Get khat
        std::int64_t nx = GetKcomp(i, pfb->kdisp[0], pfb->kNx[0]);
        std::int64_t ny = GetKcomp(j, pfb->kdisp[1], pfb->kNx[1]);
        std::int64_t nz = GetKcomp(k, pfb->kdisp[2], pfb->kNx[2]);
        Real kx = nx*pfb->dkx[0];
        Real ky = ny*pfb->dkx[1];
        Real kz = nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);

        std::int64_t kidx = pfb->GetIndex(i, j, k, idx);
        std::int64_t gidx = pfb->GetGlobalIndex(i,j,k);
        if (gidx == 0.0) {
          fv_co[0][kidx] = std::complex<Real>(0,0);
          fv_co[1][kidx] = std::complex<Real>(0,0);
          fv_co[2][kidx] = std::complex<Real>(0,0);

          fv_sh[0][kidx] = std::complex<Real>(0,0);
          fv_sh[1][kidx] = std::complex<Real>(0,0);
          fv_sh[2][kidx] = std::complex<Real>(0,0);
        } else {
          kx /= kmag;
          ky /= kmag;
          kz /= kmag;
          // Form (khat.f)
          std::complex<Real> kdotf = kx*fv[0][kidx] + ky*fv[1][kidx] + kz*fv[2][kidx];

          fv_co[0][kidx] = kdotf * kx;
          fv_co[1][kidx] = kdotf * ky;
          fv_co[2][kidx] = kdotf * kz;

          fv_sh[0][kidx] = fv[0][kidx] - fv_co[0][kidx];
          fv_sh[1][kidx] = fv[1][kidx] - fv_co[1][kidx];
          fv_sh[2][kidx] = fv[2][kidx] - fv_co[2][kidx];
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PerturbationGenerator::GetKcomp(int idx, int disp, int Nx)
//! \brief Get k index, which runs from 0, 1, ... Nx/2-1, -Nx/2, -Nx/2+1, ..., -1.

std::int64_t PerturbationGenerator::GetKcomp(int idx, int disp, int Nx) {
  return ((idx+disp) - static_cast<std::int64_t>(2*(idx+disp)/Nx)*Nx);
}
