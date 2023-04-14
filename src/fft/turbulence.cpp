//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turbulence.cpp
//  \brief implementation of functions in class Turbulence

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

//----------------------------------------------------------------------------------------
//! \fn TurbulenceDriver::TurbulenceDriver(Mesh *pm, ParameterInput *pin)
//! \brief TurbulenceDriver constructor
//!
//! \note
//! turb_flag is initialzed in the Mesh constructor to 0 by default;
//! - turb_flag = 1 for decaying turbulence
//! - turb_flag = 2 for impulsively driven turbulence
//! - turb_flag = 3 for continuously driven turbulence

TurbulenceDriver::TurbulenceDriver(Mesh *pm, ParameterInput *pin) :
    PerturbationGenerator(pm, pin),
    turb_flag(pin->GetOrAddInteger("problem","turb_flag",0)),
    tdrive(pm->time),
    // driving interval must be set manually:
    dtdrive(turb_flag == 2 ? pin->GetReal("problem", "dtdrive") : 0.0),
    // correlation time scales for OU smoothing:
    tcorr(turb_flag > 1 ? pin->GetReal("problem", "tcorr") : 0.0),
    dedt(pin->GetReal("problem", "dedt")), // turbulence amplitude
    fv_new_(nullptr) {
}

// destructor
TurbulenceDriver::~TurbulenceDriver() {
  for (int nv=0; nv<3; nv++) {
    if (fv_new_ != nullptr) delete [] fv_new_[nv];
  }
  if (fv_new_ != nullptr) delete [] fv_new_;
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Driving()
//! \brief Generate and Perturb the velocity field

void TurbulenceDriver::Driving() {
  Mesh *pm = pmy_mesh_;

  // check driving time interval to generate new perturbation
  switch(turb_flag) {
    case 1: // turb_flag == 1 : decaying turbulence
      Generate();
      Perturb(0);
      break;
    case 2: // turb_flag == 2 : impulsively driven turbulence with OU smoothing
      if (pm->time >= tdrive) {
        tdrive = pm->time + dtdrive;
        Generate();
        Perturb(dtdrive);
      }
      break;
    case 3: // turb_flag == 3 : continuously driven turbulence with OU smoothing
      Generate();
      Perturb(pm->dt);
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in TurbulenceDriver::Driving" << std::endl
          << "Turbulence flag " << turb_flag << " is not supported!" << std::endl;
      ATHENA_ERROR(msg);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Generate()
//! \brief Generate velocity pertubation.
//! Ornstein–Uhlenbeck (OU) process based on Eq. 26 in Lynn et al. (2012)
//! original formalism for
//! \f$ f=exp(-dt/tcorr) \f$
//! \f[ dv_k(t+dt) = f*dv_k(t) + sqrt(1-f^2)*dv_k' \f]

void TurbulenceDriver::Generate() {
  Mesh *pm = pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_;

  // For driven turbulence (turb_flag == 2 or 3),
  // Ornstein-Uhlenbeck (OU) process is implemented.
  // fv_ are set initially (or in restart) and kept
  // unless tcorr == 0
  if (fv_new_ == nullptr) {
    GenerateVector();
    if (turb_flag > 1) {
      fv_new_ = new std::complex<Real>*[3];
      for (int nv=0; nv<3; nv++) fv_new_[nv] = new std::complex<Real>[pfb->cnt_];
    }
  } else {
    Real OUdt = pm->dt;
    if (turb_flag == 2) OUdt=dtdrive;
    GenerateVector(fv_new_);

    // apply OU smoothing

    Real factor = std::exp(-OUdt/tcorr);
    Real sqrt_factor = std::sqrt(1 - factor*factor);

    for (int nv=0; nv<3; nv++) {
      for (int k=0; k<pfb->cnt_; k++) {
        fv_[nv][k] = factor * fv_[nv][k] + sqrt_factor * fv_new_[nv][k];
      }
    }
  }

  AssignVector();
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Perturb(Real dt)
//! \brief Add velocity perturbation to the hydro variables

void TurbulenceDriver::Perturb(Real dt) {
  Mesh *pm = pmy_mesh_;

  int il = pm->my_blocks(0)->is, iu = pm->my_blocks(0)->ie;
  int jl = pm->my_blocks(0)->js, ju = pm->my_blocks(0)->je;
  int kl = pm->my_blocks(0)->ks, ku = pm->my_blocks(0)->ke;

  Real aa, b, c, s, de, v1, v2, v3, den, M1, M2, M3;
  Real m[4] = {0};

  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    PerturbationBlock *ptb = my_ptblocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          den = pmb->phydro->u(IDN,k,j,i);
          m[0] += den;
          m[1] += den*ptb->vec(0,k,j,i);
          m[2] += den*ptb->vec(1,k,j,i);
          m[3] += den*ptb->vec(2,k,j,i);
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  int mpierr;
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = " << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL

  for (int nb=0; nb<nmb; nb++) {
    PerturbationBlock *ptb = my_ptblocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          ptb->vec(0,k,j,i) -= m[1]/m[0];
          ptb->vec(1,k,j,i) -= m[2]/m[0];
          ptb->vec(2,k,j,i) -= m[3]/m[0];
        }
      }
    }
  }

  // Calculate unscaled energy of perturbations
  m[0] = 0.0;
  m[1] = 0.0;
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    PerturbationBlock *ptb = my_ptblocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          v1 = ptb->vec(0,k,j,i);
          v2 = ptb->vec(1,k,j,i);
          v3 = ptb->vec(2,k,j,i);
          den = pmb->phydro->u(IDN,k,j,i);
          M1 = pmb->phydro->u(IM1,k,j,i);
          M2 = pmb->phydro->u(IM2,k,j,i);
          M3 = pmb->phydro->u(IM3,k,j,i);
          m[0] += den*(SQR(v1) + SQR(v2) + SQR(v3));
          m[1] += M1*v1 + M2*v2 + M3*v3;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = "
        << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL

  // Rescale to give the correct energy injection rate
  if (turb_flag > 1) {
    // driven turbulence
    de = dedt*dt;
  } else {
    // decaying turbulence (all in one shot)
    de = dedt;
  }
  aa = 0.5*m[0];
  aa = std::max(aa,static_cast<Real>(1.0e-20));
  b = m[1];
  c = -de/dvol;
  if (b >= 0.0)
    s = (-2.0*c)/(b + std::sqrt(b*b - 4.0*aa*c));
  else
    s = (-b + std::sqrt(b*b - 4.0*aa*c))/(2.0*aa);

  if (std::isnan(s)) std::cout << "[perturb]: s is NaN!" << std::endl;

  // Apply momentum pertubations
  for (int nb=0; nb<pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    PerturbationBlock *ptb = my_ptblocks(nb);
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          v1 = ptb->vec(0,k,j,i);
          v2 = ptb->vec(1,k,j,i);
          v3 = ptb->vec(2,k,j,i);
          den = pmb->phydro->u(IDN,k,j,i);
          M1 = pmb->phydro->u(IM1,k,j,i);
          M2 = pmb->phydro->u(IM2,k,j,i);
          M3 = pmb->phydro->u(IM3,k,j,i);

          if (NON_BAROTROPIC_EOS) {
            pmb->phydro->u(IEN,k,j,i) += s*(M1*v1 + M2*v2+M3*v3)
                                         + 0.5*s*s*den*(SQR(v1) + SQR(v2) + SQR(v3));
          }
          pmb->phydro->u(IM1,k,j,i) += s*den*v1;
          pmb->phydro->u(IM2,k,j,i) += s*den*v2;
          pmb->phydro->u(IM3,k,j,i) += s*den*v3;
        }
      }
    }
  }
  return;
}
