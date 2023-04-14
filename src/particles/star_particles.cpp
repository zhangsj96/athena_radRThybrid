//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file star_particles.cpp
//! \brief implements functions in the StarParticles class

// C++ headers
#include <algorithm>  // min()

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "particles.hpp"

//--------------------------------------------------------------------------------------
//! \fn StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a StarParticles instance.

StarParticles::StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp)
  : Particles(pmb, pin, pp), imetal(-1), iage(-1), ifgas(-1),
    hydro_integrator(pin->GetOrAddString("time", "integrator", "vl2")) {
  // Add metal mass
  imetal = AddRealProperty();
  realpropname.push_back("metal");

  // Add particle age
  iage = AddRealProperty();
  realpropname.push_back("age");

  // Add gas fraction as aux peroperty
  ifgas = AddAuxProperty();
  auxpropname.push_back("fgas");

  // Allocate memory and assign shorthands (shallow slices).
  // Every derived Particles need to call these two functions.
  AllocateMemory();
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn StarParticles::~StarParticles()
//! \brief destroys a StarParticles instance.

StarParticles::~StarParticles() {
  // nothing to do
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::AssignShorthandsForDerived()
//! \brief assigns shorthands by shallow coping slices of the data.

void StarParticles::AssignShorthandsForDerived() {
  metal.InitWithShallowSlice(realprop, 2, imetal, 1);
  age.InitWithShallowSlice(realprop, 2, iage, 1);

  fgas.InitWithShallowSlice(auxprop, 2, ifgas, 1);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Integrate(int step)
//! \brief updates all particle positions and velocities from t to t + dt.
//!

void StarParticles::Integrate(int stage) {
  if (hydro_integrator == "vl2") {
    VL2DKD(stage);
  } else if (hydro_integrator == "rk2") {
//    RK2(stage);
    RK2KDK(stage);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in [StarParticles::Integrate]" << std::endl
        << "integrator=" << hydro_integrator << " does not work with star particles."
        << std::endl;
    ATHENA_ERROR(msg);
  }
}


//--------------------------------------------------------------------------------------
//! \fn void StarParticles::VL2DKD(int step)
//! \brief DKD leapfrog integrator.
//!
//! Satisfying Newton's 3rd law when coupled with VL2 hydro integrator (gas->particle
//! = particle->gas).

void StarParticles::VL2DKD(int stage) {
  Real t = pmy_mesh_->time;
  Real dt = pmy_mesh_->dt; // t^(n+1) - t^n;
  Real hdt = 0.5*dt;

  switch (stage) {
  case 1:
    // Save position x^n and velocity v^n at time t^n
    SaveStatus();
    // Step 1. Drift from x^n to x^(n+1/2)
    Drift(t, hdt);
    // Update the position index to be used in Poisson solver
    UpdatePositionIndices();
    break;
  case 2:
    // Calculate the acceleration g(u^(n+1/2), x^(n+1/2))
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // Step 2. Kick from v^n to v^(n+1)
    // Boris algorithm splits the full kick into three consecutive kicks.
    // v^n -> v^(-) -> v^(+) -> v^(n+1)
    // See Appendix of Moon et al. (2021) for notations.

    // kick from v^n to v^(-) : gravity
    Kick(t, hdt);
    // rotation from v^(-) to v^(+) : Coriolis
    if (pmy_mesh_->shear_periodic) BorisKick(t, dt);
    // kick from v^(+) to v^(n+1) : gravity
    Kick(t, hdt);

    // Step 3. Drift from x^(n+1/2) to x^n
    Drift(t, hdt);
    // Update the position index to be used in Poisson solver
    UpdatePositionIndices();

    // Update the age of the star particle
    Age(t, dt);
    break;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::RK2KDK(int step)
//! \brief KDK particle integrator
//!
//! Satisfying Newton's 3rd law when coupled with RK2 hydro integrator (gas->particle
//! = particle->gas).
//! Need to do the force interpolation in both stage (twice as many compared to VL2DKD).

void StarParticles::RK2KDK(int stage) {
  Real t = pmy_mesh_->time;
  Real dt = pmy_mesh_->dt; // t^(n+1) - t^n;
  Real hdt = 0.5*dt;

  switch (stage) {
  case 1:
    // Save position x^n and velocity v^n at time t^n
    SaveStatus();

    // Calculate the acceleration g(u^n, x^n)
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // Step 1. Opening kick from v^n to v^(n+1/2)
    // Order is important!
    if (pmy_mesh_->shear_periodic) BorisKick(t, hdt);
    Kick(t, hdt);

    // Step 2. Drift from x^n to x^(n+1)
    Drift(t, dt);

    // Update the position index to be used in Poisson solver
    UpdatePositionIndices();
    break;
  case 2:
    // Calculate the acceleration g(u', x^(n+1))
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // Step 3. Closing kick from v^(n+1/2) to v^(n+1)
    // Order is important!
    Kick(t, hdt);
    if (pmy_mesh_->shear_periodic) BorisKick(t, hdt);

    // Update the age of the star particle
    Age(t, dt);
    break;
  }
}


//--------------------------------------------------------------------------------------
//! \fn void StarParticles::RK2(int step)
//! \brief RK2 particle integrator
//!
//! Satisfying Newton's 3rd law when coupled with RK2 hydro integrator (gas->particle
//! = particle->gas).
//!
//! Let q = (x, v) and u = (hydro conserved variable)
//! q'  = q^n + dt*g(u^n, q^n)
//! q'' = q'  + dt*g(u' , q' )
//! q^(n+1) = 0.5*(q^n + q'') = q^n + dt*0.5*(g(u^n, q^n) + g(u', q'))

void StarParticles::RK2(int stage) {
  Real t = pmy_mesh_->time;
  Real dt = pmy_mesh_->dt; // t^(n+1) - t^n;
  Real hdt = 0.5*dt;

  switch (stage) {
  case 1:
    // Save position x^n and velocity v^n at time t^n
    SaveStatus();

    // Calculate the acceleration g(u^n, x^n)
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // q^n -> q'
    Drift(t, dt);

    // Kick from v^n to v'
    // Boris algorithm splits the full kick into three consecutive kicks.
    // v^n -> v^(-) -> v^(+) -> v'
    // See Appendix of Moon et al. (2021) for notations.
    // Note that for RK2, the gravity is forward Euler whereas the Coriolis
    // is semi-implicit.

    // kick from v^n to v^(-) : gravity
    Kick(t, hdt);
    // rotation from v^(-) to v^(+) : Coriolis
    if (pmy_mesh_->shear_periodic) BorisKick(t, dt);
    // kick from v^(+) to v' : gravity
    Kick(t, hdt);

    // Update the position index to be used in Poisson solver
    UpdatePositionIndices();
    break;
  case 2:
    // Calculate the acceleration g(u', x')
    if (SELF_GRAVITY_ENABLED) {
      ppgrav->FindGravitationalForce(pmy_block->pgrav->phi);
      ppgrav->InterpolateGravitationalForce();
    }

    // q' -> q''
    Drift(t, dt);
    Kick(t, hdt);
    if (pmy_mesh_->shear_periodic) BorisKick(t, dt);
    Kick(t, hdt);

    // q^(n+1) = 0.5*(q^n + q'')
    for (int k = 0; k < npar_; ++k) {
      xp(k) = 0.5*(xp0(k) + xp(k));
      yp(k) = 0.5*(yp0(k) + yp(k));
      zp(k) = 0.5*(zp0(k) + zp(k));
      vpx(k) = 0.5*(vpx0(k) + vpx(k));
      vpy(k) = 0.5*(vpy0(k) + vpy(k));
      vpz(k) = 0.5*(vpz0(k) + vpz(k));
    }

    // Update the position index to be used in Poisson solver
    UpdatePositionIndices();

    // Update the age of the star particle
    Age(t, dt);
    break;
  }
}


//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Age(Real t, Real dt)
//! \brief aging particles

void StarParticles::Age(Real t, Real dt) {
  // aging particles
  for (int k = 0; k < npar_; ++k) age(k) += dt;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Drift(Real t, Real dt)
//! \brief drift particles

void StarParticles::Drift(Real t, Real dt) {
  // drift position
  for (int k = 0; k < npar_; ++k) {
    xp(k) += dt * vpx(k);
    yp(k) += dt * vpy(k);
    zp(k) += dt * vpz(k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::Kick(Real t, Real dt)
//! \brief kick particles
//!
//! forces from self gravity, external gravity, and tidal potential
//! Coriolis force is treated by BorisKick
//! dt must be the half dt

void StarParticles::Kick(Real t, Real dt) {
  // Integrate the source terms (e.g., acceleration).
  AthenaArray<Real> emptyarray;
  SourceTerms(t, dt, emptyarray);
  UserSourceTerms(t, dt, emptyarray);
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::BorisKick(Real t, Real dt)
//! \brief Coriolis force with Boris algorithm
//!
//! symmetric force application using the Boris algorithm
//! velocity must be updated to the midpoint before this call
//! dt must be the full dt (t^n+1/2 - t^n-1/2)

void StarParticles::BorisKick(Real t, Real dt) {
  Real Omdt = Omega_0_*dt, hOmdt = 0.5*Omdt;
  Real Omdt2 = SQR(Omdt), hOmdt2 = 0.25*Omdt2;
  Real f1 = (1-Omdt2)/(1+Omdt2), f2 = 2*Omdt/(1+Omdt2);
  Real hf1 = (1-hOmdt2)/(1+hOmdt2), hf2 = 2*hOmdt/(1+hOmdt2);
  for (int k = 0; k < npar_; ++k) {
    Real vpxm = vpx(k), vpym = vpy(k);
    if (age(k) == 0) { // for the new particles
      vpx(k) = hf1*vpxm + hf2*vpym;
      vpy(k) = -hf2*vpxm + hf1*vpym;
    } else {
      vpx(k) = f1*vpxm + f2*vpym;
      vpy(k) = -f2*vpxm + f1*vpym;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ExertTidalForce(Real t, Real dt)
//! \brief force from tidal potential (qshear != 0)
//!
//! Phi_tidal = - q Omega^2 x^2
//! g_x = 2 q Omega^2 x xhat

void StarParticles::ExertTidalForce(Real t, Real dt) {
  // Negative spring constant. Particles are pushed away from x=0.
  Real kspring = -2*qshear_*SQR(Omega_0_);
  for (int k = 0; k < npar_; ++k) {
    vpx(k) -= kspring*xp(k)*dt;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::PointMass(Real t, Real dt)
//! \brief force from a point mass at origin
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::PointMass(Real t, Real dt, Real gm) {
  const Coordinates *pc = pmy_block->pcoord;
  for (int k = 0; k < npar_; ++k) {
    if (age(k) > 0) {
      Real x1, x2, x3;
      pc->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

      Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3); // m0 is at (0,0,0)
      Real acc = -gm/(r*r); // G=1
      Real ax = acc*x1/r, ay = acc*x2/r, az = acc*x3/r;

      vpx(k) += dt*ax;
      vpy(k) += dt*ay;
      vpz(k) += dt*az;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ConstantAcceleration(Real t, Real dt)
//! \brief constant acceleration

void StarParticles::ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3) {
  for (int k = 0; k < npar_; ++k) {
    if (age(k) > 0) { // first kick (from n-1/2 to n) is skipped for the new particles
      vpx(k) += dt*g1;
      vpy(k) += dt*g2;
      vpz(k) += dt*g3;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::SourceTerms()
//! \brief adds acceleration to particles.
//!
//! star particles will feel all forces that gas feels
//! \note first kick (from n-1/2 to n) is skipped for the new particles

void StarParticles::SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  Hydro *ph = pmy_block->phydro;
  // accleration due to point mass (MUST BE AT ORIGIN)
  Real gm = ph->hsrc.GetGM();
  if (gm != 0) PointMass(t,dt,gm);

  // constant acceleration (e.g. for RT instability)
  Real g1 = ph->hsrc.GetG1(), g2 = ph->hsrc.GetG2(), g3 = ph->hsrc.GetG3();
  if (g1 != 0.0 || g2 != 0.0 || g3 != 0.0)
    ConstantAcceleration(t,dt,g1,g2,g3);

  if (pmy_mesh_->shear_periodic) ExertTidalForce(t,dt);
  if (SELF_GRAVITY_ENABLED) ppgrav->ExertGravitationalForce(dt);
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::UserSourceTerms(Real t, Real dt,
//!                                         const AthenaArray<Real>& meshsrc)
//! \brief adds additional source terms to particles, overloaded by the user.

void __attribute__((weak)) StarParticles::UserSourceTerms(
    Real t, Real dt, const AthenaArray<Real>& meshsrc) {
}

//--------------------------------------------------------------------------------------
//! \fn void StarParticles::ReactToMeshAux(
//!              Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief Reacts to meshaux before boundary communications.

void StarParticles::ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Nothing to do for stars
  return;
}
