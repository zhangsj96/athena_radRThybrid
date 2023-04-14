//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particle_mesh.cpp
//! \brief implements ParticleMesh class used for operations involved in particle-mesh
//!        methods.

// Standard library
#include <algorithm>
#include <cstring>
#include <sstream>

// Athena++ classes headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/buffer_utils.hpp"
#include "particle_mesh.hpp"
#include "particles.hpp"

// Local function prototypes.
static Real _WeightFunction(Real dxi);

//--------------------------------------------------------------------------------------
//! \fn ParticleMesh::ParticleMesh(Particles *ppar, MeshBlock *pmb)
//! \brief constructs a new ParticleMesh instance.

ParticleMesh::ParticleMesh(Particles *ppar, MeshBlock *pmb) : updated(false),
  imom1(-1), imom2(-1), imom3(-1), idens(-1), nmeshaux_(0),
  is(pmb->is), ie(pmb->ie), js(pmb->js), je(pmb->je), ks(pmb->ks), ke(pmb->ke),
  active1_(ppar->active1_), active2_(ppar->active2_), active3_(ppar->active3_),
  dxi1_(active1_ ? RINF : 0),
  dxi2_(active2_ ? RINF : 0),
  dxi3_(active3_ ? RINF : 0),
  nx1_(pmb->ncells1), nx2_(pmb->ncells2), nx3_(pmb->ncells3),
  ncells_(nx1_ * nx2_ * nx3_),
  npc1_(active1_ ? NPC : 1), npc2_(active2_ ? NPC : 1), npc3_(active3_ ? NPC : 1),
  ppar_(ppar), pmb_(pmb), pmesh_(ppar->pmy_mesh_) {
  // Add density and momentum in meshaux.
  idens = AddMeshAux();
  imom1 = AddMeshAux();
  imom2 = AddMeshAux();
  imom3 = AddMeshAux();

  meshaux_.NewAthenaArray(nmeshaux_, nx3_, nx2_, nx1_);
  coarse_meshaux_.NewAthenaArray(nmeshaux_, pmb->ncc3, pmb->ncc2, pmb->ncc1);

  // Get a shorthand to density.
  dens_.InitWithShallowSlice(meshaux_, 4, idens, 1);
  mom1_.InitWithShallowSlice(meshaux_, 4, imom1, 1);
  mom2_.InitWithShallowSlice(meshaux_, 4, imom2, 1);
  mom3_.InitWithShallowSlice(meshaux_, 4, imom3, 1);

  // Enroll CellCenteredBoundaryVariable object
  AthenaArray<Real> empty_flux[3]=
    {AthenaArray<Real>(),AthenaArray<Real>(), AthenaArray<Real>()};

  pmbvar = new ParticleMeshBoundaryVariable(pmb, &meshaux_, &coarse_meshaux_,
                                            empty_flux, this);
  pmbvar->bvar_index = pmb_->pbval->bvars.size();
  pmb_->pbval->bvars.push_back(pmbvar);
  // Add particle mesh boundary variable to the list for main integrator
  // if that particle exert gravity. Otherwise, add it to the list for
  // outputs.
  pmb_->pbval->bvars_pm.push_back(pmbvar);
  if (ppar_->IsGravity()) pmb_->pbval->bvars_pm_grav.push_back(pmbvar);
}

//--------------------------------------------------------------------------------------
//! \fn ParticleMesh::~ParticleMesh()
//! \brief destructs a ParticleMesh instance.

ParticleMesh::~ParticleMesh() {
  // Destroy the particle meshblock.
  meshaux_.DeleteAthenaArray();
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::ComputePMDensity(bool include_momentum)
//! \brief finds the mass and momentum density of particles on the mesh.

void ParticleMesh::ComputePMDensity(bool include_momentum) {
  Coordinates *pc(pmb_->pcoord);

  if (include_momentum) {
    AthenaArray<Real> parprop, mom1, mom2, mom3, mpar;
    parprop.NewAthenaArray(4, ppar_->npar_);
    mpar.InitWithShallowSlice(parprop, 2, 0, 1);
    mom1.InitWithShallowSlice(parprop, 2, 1, 1);
    mom2.InitWithShallowSlice(parprop, 2, 2, 1);
    mom3.InitWithShallowSlice(parprop, 2, 3, 1);
    for (int k = 0; k < ppar_->npar_; ++k) {
      pc->CartesianToMeshCoordsVector(ppar_->xp(k), ppar_->yp(k), ppar_->zp(k),
        ppar_->mass(k)*ppar_->vpx(k), ppar_->mass(k)*ppar_->vpy(k),
        ppar_->mass(k)*ppar_->vpz(k), mom1(k), mom2(k), mom3(k));
      mpar(k) = ppar_->mass(k);
    }
    DepositParticlesToMeshAux(parprop, 0, idens, 4);
  } else {
    DepositParticlesToMeshAux(ppar_->mass, 0, idens, 1);
  }

  // set flag to trigger PM communications
  updated = true;
  pmbvar->var_buf.ZeroClear();
}

//--------------------------------------------------------------------------------------
//! \fn ParticleMesh::DepositPMtoMesh()
//! \brief deposit PM momentum to hydro vars
//!
//! this has to be tested
void ParticleMesh::DepositPMtoMesh(int stage) {
  // Deposit ParticleMesh meshaux to MeshBlock.
  Hydro *phydro = pmb_->phydro;
  Real t = pmesh_->time, dt = pmesh_->dt;

  switch (stage) {
  case 1:
    dt = 0.5 * dt;
    break;

  case 2:
    t += 0.5 * dt;
    break;
  }

  ppar_->DepositToMesh(t, dt, phydro->w, phydro->u);
}

//--------------------------------------------------------------------------------------
//! \fn Real ParticleMesh::FindMaximumDensity()
//! \brief returns the maximum density in the meshblock.

Real ParticleMesh::FindMaximumDensity() const {
  Real dmax = 0.0;
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i)
        dmax = std::max(dmax, dens_(k,j,i));
  return dmax;
}

//--------------------------------------------------------------------------------------
//! \fn AthenaArray<Real> ParticleMesh::GetVelocityField()
//! \brief returns the mass-weighted mean particle velocity at each cell
//!
//! \note
//!   Precondition:
//!   The particle properties on mesh must be assigned using the class method
//!   Particles::ComputePMDensityAndCommunicate().
// TODO(SMOON) can't it be controled using the variable "updated"?
// Need to look into time integrator task list and main.

AthenaArray<Real> ParticleMesh::GetVelocityField() const {
  AthenaArray<Real> vel(3, nx3_, nx2_, nx1_);
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real rho = dens_(k,j,i);
        rho = (rho > 0.0) ? rho : 1.0;
        vel(0,k,j,i) = mom1_(k,j,i) / rho;
        vel(1,k,j,i) = mom2_(k,j,i) / rho;
        vel(2,k,j,i) = mom3_(k,j,i) / rho;
      }
    }
  }
  return vel;
}

//--------------------------------------------------------------------------------------
//! \fn int ParticleMesh::AddMeshAux()
//! \brief adds one auxiliary to the mesh and returns the index.

int ParticleMesh::AddMeshAux() {
  return nmeshaux_++;
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::InterpolateMeshToParticles(
//!              const AthenaArray<Real>& meshsrc, int ms1,
//!              AthenaArray<Real>& par, int p1, int nprop)
//! \brief interpolates meshsrc from property index ms1 to ms1+nprop-1 onto particle
//!     array par (realprop, auxprop, or work in Particles class) from property index p1
//!     to p1+nprop-1.

void ParticleMesh::InterpolateMeshToParticles(
         const AthenaArray<Real>& meshsrc, int ms1,
         AthenaArray<Real>& par, int p1, int nprop) {
  // Transpose meshsrc.
  int nx1 = meshsrc.GetDim1(), nx2 = meshsrc.GetDim2(), nx3 = meshsrc.GetDim3();
  AthenaArray<Real> u;
  u.NewAthenaArray(nx3,nx2,nx1,nprop);
  for (int n = 0; n < nprop; ++n)
    for (int k = 0; k < nx3; ++k)
      for (int j = 0; j < nx2; ++j)
        for (int i = 0; i < nx1; ++i)
          u(k,j,i,n) = meshsrc(ms1+n,k,j,i);

  // Allocate space for SIMD.
  Real **w1 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc1_];
  Real **w2 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc2_];
  Real **w3 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc3_];
  for (int i = 0; i < npc1_; ++i)
    w1[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc2_; ++i)
    w2[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc3_; ++i)
    w3[i] = new Real[SIMD_WIDTH];
  int imb1v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int imb2v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int imb3v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  // Loop over each particle.
  int npar = ppar_->npar_;
  for (int k = 0; k < npar; k += SIMD_WIDTH) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar-k); ++kk) {
      int kkk = k + kk;

      // Find the domain the particle influences.
      Real xi1 = ppar_->xi1_(kkk), xi2 = ppar_->xi2_(kkk), xi3 = ppar_->xi3_(kkk);
      int imb1 = static_cast<int>(xi1 - dxi1_),
          imb2 = static_cast<int>(xi2 - dxi2_),
          imb3 = static_cast<int>(xi3 - dxi3_);
      xi1 = imb1 + 0.5 - xi1;
      xi2 = imb2 + 0.5 - xi2;
      xi3 = imb3 + 0.5 - xi3;

      imb1v[kk] = imb1;
      imb2v[kk] = imb2;
      imb3v[kk] = imb3;

      // Weigh each cell.
#pragma loop count (NPC)
      for (int i = 0; i < npc1_; ++i)
        w1[i][kk] = active1_ ? _WeightFunction(xi1 + i) : 1.0;
#pragma loop count (NPC)
      for (int i = 0; i < npc2_; ++i)
        w2[i][kk] = active2_ ? _WeightFunction(xi2 + i) : 1.0;
#pragma loop count (NPC)
      for (int i = 0; i < npc3_; ++i)
        w3[i][kk] = active3_ ? _WeightFunction(xi3 + i) : 1.0;
    }

#pragma ivdep
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar-k); ++kk) {
      int kkk = k + kk;

      // Initiate interpolation.
      Real *pd = new Real[nprop];
      for (int i = 0; i < nprop; ++i)
        pd[i] = 0.0;

      int imb1 = imb1v[kk], imb2 = imb2v[kk], imb3 = imb3v[kk];

#pragma loop count (NPC)
      for (int ipc3 = 0; ipc3 < npc3_; ++ipc3) {
#pragma loop count (NPC)
        for (int ipc2 = 0; ipc2 < npc2_; ++ipc2) {
#pragma loop count (NPC)
          for (int ipc1 = 0; ipc1 < npc1_; ++ipc1) {
            Real w = w1[ipc1][kk] * w2[ipc2][kk] * w3[ipc3][kk];

            // Interpolate meshsrc to particles.
            for (int n = 0; n < nprop; ++n)
              pd[n] += w * u(imb3+ipc3,imb2+ipc2,imb1+ipc1,n);
          }
        }
      }

      // Record the final interpolated properties.
      for (int n = 0; n < nprop; ++n)
        par(p1+n,kkk) = pd[n];

      delete [] pd;
    }
  }

  // Release working arrays.
  u.DeleteAthenaArray();
  for (int i = 0; i < npc1_; ++i)
    delete [] w1[i];
  for (int i = 0; i < npc2_; ++i)
    delete [] w2[i];
  for (int i = 0; i < npc3_; ++i)
    delete [] w3[i];
  delete [] w1;
  delete [] w2;
  delete [] w3;
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::DepositParticlesToMeshAux(
//!              const AthenaArray<Real>& par, int p1, int ma1, int nprop)
//! \brief Deposit particle properties into Mesh.
//!
//!        Assigns par (realprop, auxprop, or work in Particles class) from property
//!        index p1 to p1+nprop-1 onto meshaux from property index ma1 and up.
void ParticleMesh::DepositParticlesToMeshAux(
         const AthenaArray<Real>& par, int p1, int ma1, int nprop) {
  // Zero out meshaux.
  Real *pfirst = &meshaux_(ma1,0,0,0);
  int size = nprop*meshaux_.GetDim3()*meshaux_.GetDim2()*meshaux_.GetDim1();
  std::fill(pfirst, pfirst + size, 0.0);
  // Allocate space for SIMD.
  Real **w1 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc1_];
  Real **w2 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc2_];
  Real **w3 __attribute__((aligned(CACHELINE_BYTES))) = new Real*[npc3_];
  for (int i = 0; i < npc1_; ++i)
    w1[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc2_; ++i)
    w2[i] = new Real[SIMD_WIDTH];
  for (int i = 0; i < npc3_; ++i)
    w3[i] = new Real[SIMD_WIDTH];
  int imb1v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int imb2v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int imb3v[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Coordinates *pc = pmb_->pcoord;

  // Loop over each particle.
  int npar = ppar_->npar_;
  for (int k = 0; k < npar; k += SIMD_WIDTH) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar-k); ++kk) {
      int kkk = k + kk;

      // Find the domain the particle influences.
      Real xi1 = ppar_->xi1_(kkk), xi2 = ppar_->xi2_(kkk), xi3 = ppar_->xi3_(kkk);
      int imb1 = static_cast<int>(xi1 - dxi1_),
          imb2 = static_cast<int>(xi2 - dxi2_),
          imb3 = static_cast<int>(xi3 - dxi3_);
      xi1 = imb1 + 0.5 - xi1;
      xi2 = imb2 + 0.5 - xi2;
      xi3 = imb3 + 0.5 - xi3;

      imb1v[kk] = imb1;
      imb2v[kk] = imb2;
      imb3v[kk] = imb3;

      // Weigh each cell.
#pragma loop count (NPC)
      for (int i = 0; i < npc1_; ++i)
        w1[i][kk] = active1_ ? _WeightFunction(xi1 + i) : 1.0;
#pragma loop count (NPC)
      for (int i = 0; i < npc2_; ++i)
        w2[i][kk] = active2_ ? _WeightFunction(xi2 + i) : 1.0;
#pragma loop count (NPC)
      for (int i = 0; i < npc3_; ++i)
        w3[i][kk] = active3_ ? _WeightFunction(xi3 + i) : 1.0;
    }

#pragma ivdep
    for (int kk = 0; kk < std::min(SIMD_WIDTH, npar-k); ++kk) {
      int kkk = k + kk;

      // Fetch particle properties.
      Real *ps = new Real[nprop];
      for (int i = 0; i < nprop; ++i)
        ps[i] = par(p1+i,kkk);

      int imb1 = imb1v[kk], imb2 = imb2v[kk], imb3 = imb3v[kk];

#pragma loop count (NPC)
      for (int ipc3 = 0; ipc3 < npc3_; ++ipc3) {
#pragma loop count (NPC)
        for (int ipc2 = 0; ipc2 < npc2_; ++ipc2) {
#pragma loop count (NPC)
          for (int ipc1 = 0; ipc1 < npc1_; ++ipc1) {
            Real w = w1[ipc1][kk] * w2[ipc2][kk] * w3[ipc3][kk];
            Real vol = pc->GetCellVolume(imb3+ipc3,imb2+ipc2,imb1+ipc1);
            // Assign particles to meshaux.
            for (int n = 0; n < nprop; ++n)
              meshaux_(ma1+n,imb3+ipc3,imb2+ipc2,imb1+ipc1) += w * ps[n] / vol;
          }
        }
      }
      delete [] ps;
    }
  }

  // Release working array.
  for (int i = 0; i < npc1_; ++i)
    delete [] w1[i];
  for (int i = 0; i < npc2_; ++i)
    delete [] w2[i];
  for (int i = 0; i < npc3_; ++i)
    delete [] w3[i];
  delete [] w1;
  delete [] w2;
  delete [] w3;
}

//--------------------------------------------------------------------------------------
//! \fn void ParticleMesh::DepositMeshAux(AthenaArray<Real>& u,
//!                                       int ma1, int mb1, int nprop)
//! \brief deposits data in meshaux from property index ma1 to ma1+nprop-1 to meshblock
//!        data u from property index mb1 and mb1+nprop-1, divided by cell volume.

void ParticleMesh::DepositMeshAux(AthenaArray<Real>& u, int ma1, int mb1, int nprop) {
  Coordinates *pc = pmb_->pcoord;

#pragma ivdep
  for (int n = 0; n < nprop; ++n)
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
          u(mb1+n,k,j,i) += meshaux_(ma1+n,k,j,i);
}

//--------------------------------------------------------------------------------------
//! \fn Real _WeightFunction(Real dxi)
//! \brief evaluates the weight function given index distance.

Real _WeightFunction(Real dxi) {
  dxi = std::min(std::abs(dxi), static_cast<Real>(1.5));
  return dxi < 0.5 ? 0.75 - dxi * dxi : 0.5 * ((1.5 - dxi) * (1.5 - dxi));
}
