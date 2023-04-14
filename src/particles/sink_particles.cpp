//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file sink_particles.cpp
//! \brief implements functions in the SinkParticles class

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "particles.hpp"

int sgn(int val) {
  return (0 < val) - (val < 0);
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a SinkParticles instance.

SinkParticles::SinkParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : StarParticles(pmb, pin, pp) {
  if (rctrl > pp->max_rinfl) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SinkParticles constructor" << std::endl
      << "Control volume radius = " << rctrl << " is larger than the maximum radius of "
      << "influence = " << pp->max_rinfl << std::endl;
    ATHENA_ERROR(msg);
  }
  if (NGHOST < req_nghost_ + rctrl + 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SinkParticles constructor" << std::endl
      << req_nghost_ << " ghost cells are required for hydro/MHD, "
      << "but sink particle needs more ghost cells to fill the control volume"
      << " overlaping with the required ghost cells." << std::endl
      << "Reconfigure with --nghost=XXX with XXX >= " << req_nghost_ + rctrl + 1
      << std::endl;
    ATHENA_ERROR(msg);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::~SinkParticles()
//! \brief destroys a SinkParticles instance.

SinkParticles::~SinkParticles() {
  // nothing to do
  return;
}


//--------------------------------------------------------------------------------------
//! \fn SinkParticles::CreateInLoop()
//! \brief Create particles on-the-fly based on LP threshold

void SinkParticles::CreateInLoop() {
  // Set Larson-Penston threshold (Eqn. (7) of Kim & Ostriker 2017, ApJ, 846)
  // rhoLP changes when mesh is refined.
  Coordinates *pcoord = pmy_block->pcoord;
  Real four_pi_G = pmy_block->pgrav->four_pi_G;
  Real dx = pcoord->dx1f(0);
  Real cs = pmy_block->peos->GetIsoSoundSpeed();
  Real rhoLP = 8.86*SQR(cs)/four_pi_G/SQR(dx/2.);

  for (int k=pmy_block->ks; k<=pmy_block->ke; ++k) {
    for (int j=pmy_block->js; j<=pmy_block->je; ++j) {
      for (int i=pmy_block->is; i<=pmy_block->ie; ++i) {
        if (pmy_block->phydro->u(IDN,k,j,i) > rhoLP) {
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          // Create a particle with zero mass and momentum
          // Mass and momentum will be set based on the control volume information in
          // subsequent INTERACT task.
          int n = AddOneParticle(0, x, y, z, 0, 0, 0);
          if (n==-1) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::CreateInLoop]"
                << ": LP threshold is satisfied, but failed to create a particle"
                << std::endl;
            ATHENA_ERROR(msg);
          }
          UpdatePositionIndices(n);
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::InteractWithMesh()
//! \brief Interact with Mesh variables (e.g., feedback, accretion)
//!
//! Must use conservative variables instead of primitive variables, because this will be
//! called before ConservedToPrimitive in the time integrator task list.

void SinkParticles::InteractWithMesh() {
  Merge();
  Accrete();
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SetControlVolume()
//! \brief Interface to set current control volume of all particles

void SinkParticles::SetControlVolume() {
  AthenaArray<Real> &cons = pmy_block->phydro->u;
  // loop over all active plus ghost particles
  for (int idx=0; idx<npar_+npar_gh_; ++idx) {
    // find the indices of the particle-containing cell.
    int ip, jp, kp;
    MeshBlockIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    SetControlVolume(cons, ip, jp, kp);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::IsControlVolumeOverlap(Real x1, Real y1, Real z1,
//                                          Real x2, Real y2, Real z2, bool &flag_strong)
//! \brief Check if control volumes of particle 1 and 2 overlap

bool SinkParticles::IsControlVolumeOverlap(Real x1, Real y1, Real z1,
                                           Real x2, Real y2, Real z2, bool &flag_strong) {
  int ip1, jp1, kp1, ip2, jp2, kp2;
  MeshBlockIndex(x1, y1, z1, ip1, jp1, kp1);
  MeshBlockIndex(x2, y2, z2, ip2, jp2, kp2);

  // find indices for overapping region
  int idst = std::min(ip1+rctrl, ip2+rctrl) - std::max(ip1-rctrl, ip2-rctrl);
  int jdst = std::min(jp1+rctrl, jp2+rctrl) - std::max(jp1-rctrl, jp2-rctrl);
  int kdst = std::min(kp1+rctrl, kp2+rctrl) - std::max(kp1-rctrl, kp2-rctrl);

  if ((idst >= 0)&&(jdst >= 0)&&(kdst >= 0)) {
    // "Strongly overlap" if one particle is inside the control volume of the other.
    // "Weakly overlap" if control volumes overlap each other, but each particle
    // does not invade the control volume of the other particle.
    flag_strong = ((idst >= rctrl)&&(jdst >= rctrl)&&(kdst >= rctrl)) ? true : false;
    return true;
  } else {
    // no overlap
    return false;
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::TestMerger(int &par1_index, int &par2_index, bool &strong_overlap)
//! \brief Test if there is any merger and return the indices of merging particles

bool SinkParticles::TestMerger(int &par1_index, int &par2_index, bool &strong_overlap) {
  for (int i=0; i<npar_+npar_gh_; ++i) {
    if (pid(i) == DEL)
      continue;
    for (int j=i+1; j<npar_+npar_gh_; ++j) {
      if (pid(j) == DEL)
        continue;
      if (IsControlVolumeOverlap(xp(i), yp(i), zp(i), xp(j), yp(j), zp(j),
                                 strong_overlap)) {
        par1_index = i;
        par2_index = j;
        return true;
      }
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::MergeTwoParticles(int i, int j, bool strong_overlap)
//! \brief Merge particle i and particle j.

void SinkParticles::MergeTwoParticles(int i, int j, bool strong_overlap) {
  std::stringstream header;
  header << "[particle] Message from rank = " << Globals::my_rank
         << ", gid = " << pmy_block->gid << ", lid = " << pmy_block->lid
         << " at  t = " << pmy_mesh_->time << ", ncycle = " << pmy_mesh_->ncycle << ": ";

  if ((pid(i) == NEW)&&(pid(j) == NEW)) {
    // If both are new particles, merge them to the midpoint
    pid(i) = pid(j) = DEL;
    Real xmid = 0.5*(xp(i) + xp(j));
    Real ymid = 0.5*(yp(i) + yp(j));
    Real zmid = 0.5*(zp(i) + zp(j));
    int k = AddOneParticle(0, xmid, ymid, zmid, 0, 0, 0, true);
    if (k==-1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::Merge]" << std::endl
          << "Two new particles have merged, but failed to create particle at the"
          << " midpoint.";
      ATHENA_ERROR(msg);
    }
    UpdatePositionIndices(k);
    std::cerr << header.str() << "Two new particles have merged" << std::endl;
  } else if ((pid(i) == NEW)&&(strong_overlap)) {
    pid(i) = DEL;
    std::cerr << header.str() << "A new particle have formed inside the control"
     << "volume of existing particle. Cancel the creation" << std::endl;
  } else if ((pid(j) == NEW)&&(strong_overlap)) {
    pid(j) = DEL;
    std::cerr << header.str() << "A new particle have formed inside the control"
     << "volume of existing particle. Cancel the creation" << std::endl;
  } else {
    // If one of the two particles is new, assign mass and velocity and reset
    // its control volume before merge.
    if (pid(i) == NEW)
      Accrete(i);
    else if (pid(j) == NEW)
      Accrete(j);

    // Inherit pid of the more massive particle.
    int newpid = mass(i) >= mass(j) ? pid(i) : pid(j);
    pid(i) = pid(j) = DEL;

    // Add a particle at the COM/position/velocity
    Real mtot = mass(i) + mass(j);
    Real mtot_inv = 1.0/mtot;
    Real xcom = (mass(i)*xp(i) + mass(j)*xp(j))*mtot_inv;
    Real ycom = (mass(i)*yp(i) + mass(j)*yp(j))*mtot_inv;
    Real zcom = (mass(i)*zp(i) + mass(j)*zp(j))*mtot_inv;
    Real vxcom = (mass(i)*vpx(i) + mass(j)*vpx(j))*mtot_inv;
    Real vycom = (mass(i)*vpy(i) + mass(j)*vpy(j))*mtot_inv;
    Real vzcom = (mass(i)*vpz(i) + mass(j)*vpz(j))*mtot_inv;
    int k = AddOneParticle(mtot, xcom, ycom, zcom, vxcom, vycom, vzcom, true);
    if ((mass(i) < 0)||(mass(j) < 0)||(mtot <= 0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::Merge]" << std::endl
          << "Zero or Negative particle mass: mass(" << i << ") = " << mass(i)
          << ", mass(" << j << ") = " << mass(j) << std::endl << "t = "
          << pmy_mesh_->time << ", ncycle = " << pmy_mesh_->ncycle << std::endl;
      ATHENA_ERROR(msg);
    }
    if (k==-1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::Merge]" << std::endl
          << "Two existing particles have merged, but failed to create a particle"
          << " at the COM position/velocity" << std::endl;
      ATHENA_ERROR(msg);
    }
    pid(k) = newpid;
    UpdatePositionIndices(k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::Merge()
//! \brief Merge particles when their control volume overlap with each other.
//!
//! \note Head-on collision leads to merge, regardless of their reletive velocity or
//!       boundness.
//!
//!       The deletion of particles would have been much simpler if the data structure
//!       for particles were dictionary-like, e.g., {pid: property}, such that we can
//!       index particles by their unique particle IDs. In the current design, every time
//!       a particle is deleted, the array index of the other particle is affected.
//!       Therefore, special care is needed when calling RemoveOneParticle in sequence.
//!       Because of this reason, MergeTwoParticles only tag particles to be removed,
//!       and the actual removal is done seperately.

void SinkParticles::Merge() {
  int nmerger = 0;
  int par1_index, par2_index;
  bool strong_overlap;

  // SMOON: Particle merging is iterative process, because a merged particle can again
  // merge with yet another particle (e.g., 3 particles trying to merge). This is likely
  // to be rare, though.
  while (TestMerger(par1_index, par2_index, strong_overlap)) {
    MergeTwoParticles(par1_index, par2_index, strong_overlap);
    nmerger++;
  }

  // Remove particles tagged for deletion.
  for (int k=0; k<npar_+npar_gh_;) {
    if (pid(k) == DEL) {
      RemoveOneParticle(k);
      // Note that the loop index is not incremented.
    } else {
      k++;
    }
  }
  if (nmerger > 0) {
    std::cerr << "[particle] Message from rank = " << Globals::my_rank
              << ", gid = " << pmy_block->gid << ", lid = " << pmy_block->lid
              << " at  t = " << pmy_mesh_->time
              << ", ncycle = " << pmy_mesh_->ncycle << ": "
              << nmerger << " sink particles have formed via merger."
              << std::endl;
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::Accrete()
//! \brief accrete gas from neighboring cells for all particles
//!
//! \note Always accrete; No converging flow check.

void SinkParticles::Accrete() {
  int ip, jp, kp, ip0, jp0, kp0;

  // loop over all active particles
  for (int idx=0; idx<npar_; ++idx) {
    Accrete(idx);
    // If the particle has crossed the grid boundaries, accrete from the previous control
    // volume as well
    MeshBlockIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    MeshBlockIndex(xp0(idx), yp0(idx), zp0(idx), ip0, jp0, kp0);
    if ((ip != ip0) || (jp != jp0) || (kp != kp0)) {
      Accrete(idx, true);
    }
  } // end of the loop over particles

  // loop over all ghost particles and reset the control volume
  // Note that ghost particles don't need to accrete; they only need to set their control
  // volume. In fact, in current implementation, ghost particles should not accrete to
  // avoid segfault, because they will try to access the cells outside the ghost zones.
  AthenaArray<Real> &cons = pmy_block->phydro->u;
  for (int idx=npar_; idx<npar_+npar_gh_; ++idx) {
    // find the indices of the particle-containing cell.
    int ip, jp, kp, ip0, jp0, kp0;
    MeshBlockIndex(xp(idx), yp(idx), zp(idx), ip, jp, kp);
    MeshBlockIndex(xp0(idx), yp0(idx), zp0(idx), ip0, jp0, kp0);
    SetControlVolume(cons, ip, jp, kp);
    if ((ip != ip0) || (jp != jp0) || (kp != kp0))
      SetControlVolume(cons, ip0, jp0, kp0);
  }
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::Accrete(int n, bool old_pos)
//! \brief accrete gas from neighboring cells for a selected particle of array index n
//!
//! If old_pos = True, accrete from the old control volume around xp0, yp0, zp0.
//!
//! The mass dM_flux that has flown into the control volume during dt must be equal
//! to the dM_sink + dM_{ctrl,reset} to conserve mass:
//!   dM_sink = dM_flux - dM_{ctrl,reset}       -- (1)
//! Meanwhile, hydro integrator will update M^n_{ctrl,reset} at time t^n (which is
//! already reset to the extrapolated value) to M^{n+1}_ctrl, which will be subsequently
//! reset to the extrapolated value M^{n+1}_{ctrl,reset}. This is done by
//!   M^{n+1}_ctrl = M^n_{ctrl,reset} + dM_flux    -- (2)
//! Therefore, instead using Riemann fluxes directly to calculate dM_flux in eq. (1), we
//! can use eq. (2) to substitute dM_flux in eq. (1) with M^{n+1}_ctrl - M^n_{ctrl,reset},
//! yielding
//!   dM_sink = M^{n+1}_ctrl - M^{n+1}_{ctrl,reset}  -- (3)
//! TODO(SMOON) AMR compatibility?

int SinkParticles::Accrete(int n, bool old_pos) {
#ifdef DEBUG
  if ((n < 0)||(n >= npar_+npar_gh_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [SinkParticles::Accrete]" << std::endl
        << "The index n = " << n << "is outside of the allowed range [0,npar+npar_gh)"
        << std::endl << "npar = " << npar_ << ", npar_gh = " << npar_gh_ << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  AthenaArray<Real> &cons = pmy_block->phydro->u;
  AthenaArray<Real> cons_ctrl0(4, 2*rctrl+1, 2*rctrl+1, 2*rctrl+1);

  // find the indices of the particle-containing cell.
  int ip, jp, kp;
  if (old_pos)
    MeshBlockIndex(xp0(n), yp0(n), zp0(n), ip, jp, kp);
  else
    MeshBlockIndex(xp(n), yp(n), zp(n), ip, jp, kp);

#ifdef DEBUG
  if ((ip-rctrl < 0)||(ip+rctrl >= pmy_block->ncells1)||
      (jp-rctrl < 0)||(jp+rctrl >= pmy_block->ncells2)||
      (kp-rctrl < 0)||(kp+rctrl >= pmy_block->ncells3)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [SinkParticles::Accrete]" << std::endl
        << "The MeshBlock index of this particle can cause segfault" << std::endl
        << " : ip = " << ip << ", jp = " << jp << ", kp = " << kp << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  // Step 1. Calculate total mass and momentum in the control volume M^{n+1}
  // that are updated by hydro integrator, before applying extrapolation.
  Real m{0.}, M1{0.}, M2{0.}, M3{0.};
  for (int k=kp-rctrl, kk=0; k<=kp+rctrl; ++k, ++kk) {
    for (int j=jp-rctrl, jj=0; j<=jp+rctrl; ++j, ++jj) {
      for (int i=ip-rctrl, ii=0; i<=ip+rctrl; ++i, ++ii) {
        Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
        m += cons(IDN,k,j,i)*dV;
        cons_ctrl0(IDN,kk,jj,ii) = cons(IDN,k,j,i);
        M1 += cons(IM1,k,j,i)*dV;
        cons_ctrl0(IM1,kk,jj,ii) = cons(IM1,k,j,i);
        M2 += cons(IM2,k,j,i)*dV;
        cons_ctrl0(IM2,kk,jj,ii) = cons(IM2,k,j,i);
        M3 += cons(IM3,k,j,i)*dV;
        cons_ctrl0(IM3,kk,jj,ii) = cons(IM3,k,j,i);
      }
    }
  }
  // Step 2. Reset the density and momentum inside the control volume by extrapolation
  SetControlVolume(cons, ip, jp, kp);
  // Step 3. Calculate M^{n+1}_ctrl
  Real mext{0.}, M1ext{0.}, M2ext{0.}, M3ext{0.};
  for (int k=kp-rctrl; k<=kp+rctrl; ++k) {
    for (int j=jp-rctrl; j<=jp+rctrl; ++j) {
      for (int i=ip-rctrl; i<=ip+rctrl; ++i) {
        Real dV = pmy_block->pcoord->GetCellVolume(k,j,i);
        mext += cons(IDN,k,j,i)*dV;
        M1ext += cons(IM1,k,j,i)*dV;
        M2ext += cons(IM2,k,j,i)*dV;
        M3ext += cons(IM3,k,j,i)*dV;
      }
    }
  }
  // Step 4. Calculate dM_sink by subtracting M^{n+1}_ctrl from M^{n+1}
  Real dm = m - mext;
  Real dM1 = M1 - M1ext;
  Real dM2 = M2 - M2ext;
  Real dM3 = M3 - M3ext;

  // Step 5. Update mass and velocity of the particle
  if (dm < 0) {
    // Restore conservative variables inside control volume.
    for (int k=kp-rctrl, kk=0; k<=kp+rctrl; ++k, ++kk) {
      for (int j=jp-rctrl, jj=0; j<=jp+rctrl; ++j, ++jj) {
        for (int i=ip-rctrl, ii=0; i<=ip+rctrl; ++i, ++ii) {
          cons(IDN,k,j,i) = cons_ctrl0(IDN,kk,jj,ii);
          cons(IM1,k,j,i) = cons_ctrl0(IM1,kk,jj,ii);
          cons(IM2,k,j,i) = cons_ctrl0(IM2,kk,jj,ii);
          cons(IM3,k,j,i) = cons_ctrl0(IM3,kk,jj,ii);
        }
      }
    }
    std::cerr << "[particle] Message from rank = " << Globals::my_rank
              << ", gid = " << pmy_block->gid << ", lid = " << pmy_block->lid
              << " at  t = " << pmy_mesh_->time << ", ncycle = " << pmy_mesh_->ncycle
              << ": Negative mass flux into particle pid(" << n << ") = " << pid(n)
              << ". Cancel the accretion." << std::endl;
    if (pid(n) == NEW) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::Accrete]" << std::endl
          << "New particle has formed, but the accretion rate is negative. "
          << "Cannot assign positive particle mass.";
      ATHENA_ERROR(msg);
    }
  } else {
    Real minv = 1.0 / (mass(n) + dm);
    vpx(n) = (mass(n)*vpx(n) + dM1)*minv;
    vpy(n) = (mass(n)*vpy(n) + dM2)*minv;
    vpz(n) = (mass(n)*vpz(n) + dM3)*minv;
    mass(n) += dm;
    if ((pid(n) == NEW)&&(dm == 0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SinkParticles::Accrete]" << std::endl
          << "New particle has formed, but the accretion rate is zero. "
          << "Cannot assign positive particle mass.";
      ATHENA_ERROR(msg);
    }
  }

  return 0;
}

//--------------------------------------------------------------------------------------
//! \fn SinkParticles::SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp)
//! \brief set control volume quantities by extrapolating from neighboring active cells.

void SinkParticles::SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp) {
  // Do extrapolation using "face-neighbors", that is, neighboring cells
  // that are outside the control volume and share a cell face with the
  // cell being extrapolated.
  // In Athena-TIGRESS, larger stencil that include edge- and corner-neighbors
  // were not compatible with shearing box. Is that true also in athena++?
  // TODO(SMOON) AMR compatibility?

  const int is(pmy_block->is), ie(pmy_block->ie);
  const int js(pmy_block->js), je(pmy_block->je);
  const int ks(pmy_block->ks), ke(pmy_block->ke);
  const int il(is - req_nghost_), iu(ie + req_nghost_);
  const int jl(js - req_nghost_), ju(je + req_nghost_);
  const int kl(ks - req_nghost_), ku(ke + req_nghost_);
  int cil, ciu, cjl, cju, ckl, cku;

  // Start extrapolation from the outermost shell, marching inward
  for (int s=rctrl; s>=1; --s) {
    // 6 front faces
    // Each has one neighbor. Do simple copy.

    // (SMOON) The loop index limits are carefully chosen in order to
    //         1. Avoid segfault,
    //         2. Make active and corresponding (required) ghost cells consistent
    //            between neighboring MeshBlocks,
    //         fully considering the possibility of a particle crossing the cell boundary.
    //         The logic is faily complicated to describe without visual aid.
    //         This function requires NGHOST=4 for rctrl=1.
    // x1-face
    ckl = std::max(kp-s+1, kl);
    cjl = std::max(jp-s+1, jl);
    cku = std::min(kp+s-1, ku);
    cju = std::min(jp+s-1, ju);
    for (int k=ckl; k<=cku; ++k) {
      for (int j=cjl; j<=cju; ++j) {
        for (int i=ip-s; i<=ip+s; i+=2*s) {
          if ((i < il-1)||(i > iu+1))
            continue;
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif

          int ioff = sgn(i-ip);
          cons(IDN,k,j,i) = cons(IDN,k,j,i+ioff);
          cons(IM1,k,j,i) = cons(IM1,k,j,i+ioff);
          cons(IM2,k,j,i) = cons(IM2,k,j,i+ioff);
          cons(IM3,k,j,i) = cons(IM3,k,j,i+ioff);
        }
      }
    }
    // x2-face
    ckl = std::max(kp-s+1, kl);
    cil = std::max(ip-s+1, il);
    cku = std::min(kp+s-1, ku);
    ciu = std::min(ip+s-1, iu);
    for (int k=ckl; k<=cku; ++k) {
      for (int j=jp-s; j<=jp+s; j+=2*s) {
        for (int i=cil; i<=ciu; ++i) {
          if ((j < jl-1)||(j > ju+1))
            continue;
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif

          int joff = sgn(j-jp);
          cons(IDN,k,j,i) = cons(IDN,k,j+joff,i);
          cons(IM1,k,j,i) = cons(IM1,k,j+joff,i);
          cons(IM2,k,j,i) = cons(IM2,k,j+joff,i);
          cons(IM3,k,j,i) = cons(IM3,k,j+joff,i);
        }
      }
    }
    // x3-face
    cjl = std::max(jp-s+1, jl);
    cil = std::max(ip-s+1, il);
    cju = std::min(jp+s-1, ju);
    ciu = std::min(ip+s-1, iu);
    for (int k=kp-s; k<=kp+s; k+=2*s) {
      for (int j=cjl; j<=cju; ++j) {
        for (int i=cil; i<=ciu; ++i) {
          if ((k < kl-1)||(k > ku+1))
            continue;
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif

          int koff = sgn(k-kp);
          cons(IDN,k,j,i) = cons(IDN,k+koff,j,i);
          cons(IM1,k,j,i) = cons(IM1,k+koff,j,i);
          cons(IM2,k,j,i) = cons(IM2,k+koff,j,i);
          cons(IM3,k,j,i) = cons(IM3,k+koff,j,i);
        }
      }
    }

    // 8 corners
    // Each has three neighbors. Average them.
    for (int k=kp-s; k<=kp+s; k+=2*s) {
      for (int j=jp-s; j<=jp+s; j+=2*s) {
        for (int i=ip-s; i<=ip+s; i+=2*s) {
          if ((i < il)||(i > iu)||(j < jl)||(j > ju)||(k < kl)||(k > ku))
            continue;
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif

          int koff = sgn(k-kp);
          int joff = sgn(j-jp);
          int ioff = sgn(i-ip);
          Real davg = ONE_3RD*(cons(IDN,k     ,j     ,i+ioff) +
                               cons(IDN,k     ,j+joff,i     ) +
                               cons(IDN,k+koff,j     ,i     ));
          Real M1avg = ONE_3RD*(cons(IM1,k     ,j     ,i+ioff) +
                                cons(IM1,k     ,j+joff,i     ) +
                                cons(IM1,k+koff,j     ,i     ));
          Real M2avg = ONE_3RD*(cons(IM2,k     ,j     ,i+ioff) +
                                cons(IM2,k     ,j+joff,i     ) +
                                cons(IM2,k+koff,j     ,i     ));
          Real M3avg = ONE_3RD*(cons(IM3,k     ,j     ,i+ioff) +
                                cons(IM3,k     ,j+joff,i     ) +
                                cons(IM3,k+koff,j     ,i     ));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }

    // 4 sides for 3 middle-slices
    // Each has two neighbors. Average them.

    // x1-slice
    cil = std::max(ip-s+1, il);
    ciu = std::min(ip+s-1, iu);
    for (int k=kp-s; k<=kp+s; k+=2*s) {
      for (int j=jp-s; j<=jp+s; j+=2*s) {
        if ((j < jl)||(j > ju)||(k < kl)||(k > ku))
          continue;
        for (int i=cil; i<=ciu; ++i) {
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif
          int koff = sgn(k-kp);
          int joff = sgn(j-jp);
          Real davg = 0.5*(cons(IDN,k+koff,j,i) + cons(IDN,k,j+joff,i));
          Real M1avg = 0.5*(cons(IM1,k+koff,j,i) + cons(IM1,k,j+joff,i));
          Real M2avg = 0.5*(cons(IM2,k+koff,j,i) + cons(IM2,k,j+joff,i));
          Real M3avg = 0.5*(cons(IM3,k+koff,j,i) + cons(IM3,k,j+joff,i));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
    // x2-slice
    cjl = std::max(jp-s+1, jl);
    cju = std::min(jp+s-1, ju);
    for (int k=kp-s; k<=kp+s; k+=2*s) {
      for (int i=ip-s; i<=ip+s; i+=2*s) {
        if ((i < il)||(i > iu)||(k < kl)||(k > ku))
          continue;
        for (int j=cjl; j<=cju; ++j) {
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif
          int koff = sgn(k-kp);
          int ioff = sgn(i-ip);
          Real davg = 0.5*(cons(IDN,k+koff,j,i) + cons(IDN,k,j,i+ioff));
          Real M1avg = 0.5*(cons(IM1,k+koff,j,i) + cons(IM1,k,j,i+ioff));
          Real M2avg = 0.5*(cons(IM2,k+koff,j,i) + cons(IM2,k,j,i+ioff));
          Real M3avg = 0.5*(cons(IM3,k+koff,j,i) + cons(IM3,k,j,i+ioff));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
    // x3-slice
    ckl = std::max(kp-s+1, kl);
    cku = std::min(kp+s-1, ku);
    for (int j=jp-s; j<=jp+s; j+=2*s) {
      for (int i=ip-s; i<=ip+s; i+=2*s) {
        if ((i < il)||(i > iu)||(j < jl)||(j > ju))
          continue;
        for (int k=ckl; k<=cku; ++k) {
#ifdef DEBUG
          if ((i < 0)||(i >= pmy_block->ncells1)||
              (j < 0)||(j >= pmy_block->ncells2)||
              (k < 0)||(k >= pmy_block->ncells3)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [SinkParticles::SetControlVolume]"
                << std::endl << "This can cause segfault" << std::endl
                << " : i = " << i << ", j = " << j << ", k = " << k << std::endl;
            ATHENA_ERROR(msg);
          }
#endif
          int joff = sgn(j-jp);
          int ioff = sgn(i-ip);
          Real davg = 0.5*(cons(IDN,k,j,i+ioff) + cons(IDN,k,j+joff,i));
          Real M1avg = 0.5*(cons(IM1,k,j,i+ioff) + cons(IM1,k,j+joff,i));
          Real M2avg = 0.5*(cons(IM2,k,j,i+ioff) + cons(IM2,k,j+joff,i));
          Real M3avg = 0.5*(cons(IM3,k,j,i+ioff) + cons(IM3,k,j+joff,i));
          cons(IDN,k,j,i) = davg;
          cons(IM1,k,j,i) = M1avg;
          cons(IM2,k,j,i) = M2avg;
          cons(IM3,k,j,i) = M3avg;
        }
      }
    }
  }
  if ((kp>=kl)&&(kp<=ku)&&(jp>=jl)&&(jp<=ju)&&(ip>=il)&&(ip<=iu)) {
    // finally, fill the central cell containing the particle
    Real davg = (cons(IDN,kp,jp,ip-1) + cons(IDN,kp,jp,ip+1) +
                 cons(IDN,kp,jp-1,ip) + cons(IDN,kp,jp+1,ip) +
                 cons(IDN,kp-1,jp,ip) + cons(IDN,kp+1,jp,ip))/6.;
    Real M1avg = (cons(IM1,kp,jp,ip-1) + cons(IM1,kp,jp,ip+1) +
                  cons(IM1,kp,jp-1,ip) + cons(IM1,kp,jp+1,ip) +
                  cons(IM1,kp-1,jp,ip) + cons(IM1,kp+1,jp,ip))/6.;
    Real M2avg = (cons(IM2,kp,jp,ip-1) + cons(IM2,kp,jp,ip+1) +
                  cons(IM2,kp,jp-1,ip) + cons(IM2,kp,jp+1,ip) +
                  cons(IM2,kp-1,jp,ip) + cons(IM2,kp+1,jp,ip))/6.;
    Real M3avg = (cons(IM3,kp,jp,ip-1) + cons(IM3,kp,jp,ip+1) +
                  cons(IM3,kp,jp-1,ip) + cons(IM3,kp,jp+1,ip) +
                  cons(IM3,kp-1,jp,ip) + cons(IM3,kp+1,jp,ip))/6.;
    cons(IDN,kp,jp,ip) = davg;
    cons(IM1,kp,jp,ip) = M1avg;
    cons(IM2,kp,jp,ip) = M2avg;
    cons(IM3,kp,jp,ip) = M3avg;
  }
}
