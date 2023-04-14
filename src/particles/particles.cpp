//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particles.cpp
//! \brief implements functions in particle classes

// C++ Standard Libraries
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "particles.hpp"

// Class variable initialization
bool Particles::initialized = false;
int Particles::num_particles = 0;
int Particles::num_particles_grav = 0;
int Particles::num_particles_output = 0;
ParameterInput* Particles::pinput = NULL;
std::vector<int> Particles::idmax;
#ifdef MPI_PARALLEL
MPI_Comm Particles::my_comm = MPI_COMM_NULL;
#endif

//--------------------------------------------------------------------------------------
//! \fn ComputeReqNGHOST(int xorder, int rinfl)
//! \brief helpfer function to initialize const member req_nghost_

int ComputeReqNGHOST(int xorder) {
  int req_nghost(0); // required number of ghost cells for hydro/MHD
  switch (xorder) {
    case 1:
      req_nghost = 1;
      break;
    case 2:
      req_nghost = 2;
      break;
    case 3:
      req_nghost = 3;
      break;
    case 4:
      req_nghost = 4;
      if (MAGNETIC_FIELDS_ENABLED)
        req_nghost += 2;
      break;
  }
  return req_nghost;
}

//--------------------------------------------------------------------------------------
//! \fn ComputeOverlap(int xorder, int rinfl)
//! \brief helpfer function to initialize const member noverlap_

int ComputeOverlap(int xorder, int rinfl) {
  // Set the thickness of the overlap region for ghost particle exchange.
  // rinfl = -1 means that the particle does not modify the fluid variable at all
  // (e.g., tracer), and therefore does not require ghost particles.
  // See the comments in the header file for more information.
  int noverlap = rinfl >= 0 ? ComputeReqNGHOST(xorder) + rinfl : 0;
  return noverlap;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::Particles(MeshBlock *pmb, ParameterInput *pin)
//! \brief constructs a Particles instance.

Particles::Particles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp) :
  ipar(pp->ipar), input_block_name(pp->block_name), partype(pp->partype),
  npar_(0), npar_gh_(0), nparmax_(1), req_nghost_(ComputeReqNGHOST(pmb->precon->xorder)),
  noverlap_(ComputeOverlap(pmb->precon->xorder, pp->max_rinfl)),
  nint(0), nreal(0), naux(0), nwork(0),
  ipid(-1), ish(-1),
  imass(-1), ixp(-1), iyp(-1), izp(-1), ivpx(-1), ivpy(-1), ivpz(-1),
  ixp0(-1), iyp0(-1), izp0(-1), ivpx0(-1), ivpy0(-1), ivpz0(-1),
  ixi1(-1), ixi2(-1), ixi3(-1), igx(-1), igy(-1), igz(-1),
  parhstout_(false), isgravity_(pp->gravity) {
  // integer properties

  // Add particle ID.
  ipid = AddIntProperty();
  intpropname.push_back("pid");

  // real properties

  // Add particle mass
  imass = AddRealProperty();
  realpropname.push_back("mass");

  // Add particle position.
  ixp = AddRealProperty();
  iyp = AddRealProperty();
  izp = AddRealProperty();
  realpropname.push_back("x1");
  realpropname.push_back("x2");
  realpropname.push_back("x3");

  // Add particle velocity.
  ivpx = AddRealProperty();
  ivpy = AddRealProperty();
  ivpz = AddRealProperty();
  realpropname.push_back("v1");
  realpropname.push_back("v2");
  realpropname.push_back("v3");

  // aux properties

  // Add old particle position.
  ixp0 = AddAuxProperty();
  iyp0 = AddAuxProperty();
  izp0 = AddAuxProperty();
  auxpropname.push_back("x10");
  auxpropname.push_back("x20");
  auxpropname.push_back("x30");

  // Add old particle velocity.
  ivpx0 = AddAuxProperty();
  ivpy0 = AddAuxProperty();
  ivpz0 = AddAuxProperty();
  auxpropname.push_back("v10");
  auxpropname.push_back("v20");
  auxpropname.push_back("v30");

  // Add particle position indices.
  ixi1 = AddWorkingArray();
  ixi2 = AddWorkingArray();
  ixi3 = AddWorkingArray();

  // Point to the calling MeshBlock.
  pmy_block = pmb;
  pmy_mesh_ = pmb->pmy_mesh;
  pbval_ = pmb->pbval;


  // Get the CFL number for particles.
  cfl_par_ = pin->GetOrAddReal(input_block_name, "cfl_par", 1);

  // Check active dimensions.
  active1_ = pmy_mesh_->mesh_size.nx1 > 1;
  active2_ = pmy_mesh_->mesh_size.nx2 > 1;
  active3_ = pmy_mesh_->mesh_size.nx3 > 1;

  // read shearing box parameters from input block
  if (pmy_mesh_->shear_periodic) {
    bool orbital_advection_defined_
           = (pin->GetOrAddInteger("orbital_advection","OAorder",0)!=0)?
             true : false;
    Omega_0_ = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
    qshear_  = pin->GetOrAddReal("orbital_advection","qshear",0.0);
    ShBoxCoord_ = pin->GetOrAddInteger("orbital_advection","shboxcoord",1);
    if (orbital_advection_defined_) { // orbital advection source terms
      std::stringstream msg;
      msg << "### FATAL ERROR in Particle constructor" << std::endl
          << "OrbitalAdvection is not yet implemented for particles" << std::endl
          << std::endl;
      ATHENA_ERROR(msg);
    }

    if (ShBoxCoord_ != 1) {
      // to relax this contrain, modify ApplyBoundaryConditions
      std::stringstream msg;
      msg << "### FATAL ERROR in Particle constructor" << std::endl
          << "only orbital_advection/shboxcoord=1 is supported" << std::endl
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // q*Omega*Lx
    qomL_ = qshear_*Omega_0_*pmy_mesh_->mesh_size.x1len;

    // aux array for shear boundary flag
    ish = AddIntProperty();
    intpropname.push_back("ish");
  }

  // Actual memory allocation and shorthand assignment will be done in the derived class
  // Initialization of ParticleBuffer, ParticleGravity
  // has moved to the derived class
}

//--------------------------------------------------------------------------------------
//! \fn Particles::~Particles()
//! \brief destroys a Particles instance.

Particles::~Particles() {
  // Delete integer properties.
  intprop.DeleteAthenaArray();
  intpropname.clear();

  // Delete real properties.
  realprop.DeleteAthenaArray();
  realpropname.clear();

  // Delete auxiliary properties.
  if (naux > 0) {
    auxprop.DeleteAthenaArray();
    auxpropname.clear();
  }

  // Delete working arrays.
  if (nwork > 0) work.DeleteAthenaArray();

  // Clear links to neighbors.
  ClearNeighbors();

  // Delete mesh auxiliaries.
  delete ppm;

  // Delete particle gravity.
  if (isgravity_) delete ppgrav;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddOneParticle()
//! \brief add one particle if position is within the mesh block
//!
//! if the particle falls in the ghost zones, create it if allow_ghost=true, otherwise
//! dismiss.

int Particles::AddOneParticle(Real mp, Real x1, Real x2, Real x3,
                              Real v1, Real v2, Real v3, bool allow_ghost) {
  int npartot = npar_ + npar_gh_;
  // We are adding an active particle
  bool active_flag = CheckInMeshBlock(x1,x2,x3);
  // We are adding a ghost particle
  bool ghost_flag = (allow_ghost)&&(!active_flag)&&(CheckInMeshBlock(x1,x2,x3,true));
  int& numpar = ghost_flag ? npar_gh_ : npar_;
  int k = ghost_flag ? npartot : npar_; // index of the particle to be added

  if (active_flag||ghost_flag) {
    // Update capacity if full.
    if (npartot == nparmax_)
      UpdateCapacity(npartot*2);

    if ((active_flag)&&(npar_gh_ > 0))
      // To add an active particle when there are ghost particles,
      // move the first ghost particle to the end to make a room and
      // initialize all properties at k=npar_ to zero.
      ReindexOneParticleAndClear(npar_, npartot);
    else
      // Initialize all properties to zero to avoid garbage values.
      ReindexOneParticleAndClear(k, k);

    pid(k) = NEW;
    mass(k) = mp;
    xp(k) = xp0(k) = x1;
    yp(k) = yp0(k) = x2;
    zp(k) = zp0(k) = x3;
    vpx(k) = vpx0(k) = v1;
    vpy(k) = vpy0(k) = v2;
    vpz(k) = vpz0(k) = v3;

    numpar++; // increment npar_ (npar_gh_) if we are adding active (ghost) particle.

    return k; // return the array index of newly added particle
  } else {
    return -1;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::RemoveOneParticle(int k)
//! \brief removes particle k in the block.

void Particles::RemoveOneParticle(int k) {
#ifdef DEBUG
  if ((k < 0) || (k >= npar_+npar_gh_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::RemoveOneParticle]" << std::endl
        << "Index k = " << k << " is ouside the allowed range [0, npar_+npar_gh_)."
        << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  if (k < npar_) {
    npar_--;
    // Overwrite the last active particle at k=npar_ to k=k and clear k=npar_
    ReindexOneParticleAndClear(npar_, k);
    // If there are ghost particles, rearrange the last one to fill the vacancy
    if (npar_gh_ > 0)
      ReindexOneParticleAndClear(npar_+npar_gh_, npar_);
  } else {
    npar_gh_--;
    ReindexOneParticleAndClear(npar_+npar_gh_, k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::Integrate(int step)
//! \brief updates all particle positions and velocities from t to t + dt.

void Particles::Integrate(int stage) {
  Real t = 0, dt = 0;

  // Determine the integration cofficients.
  switch (stage) {
  case 1:
    t = pmy_mesh_->time;
    dt = 0.5 * pmy_mesh_->dt;
    SaveStatus();
    break;

  case 2:
    t = pmy_mesh_->time + 0.5 * pmy_mesh_->dt;
    dt = pmy_mesh_->dt;
    break;
  }

  // Conduct one stage of the integration.
  EulerStep(t, dt, pmy_block->phydro->w);
  ReactToMeshAux(t, dt, pmy_block->phydro->w);

  // Update the position index.
  UpdatePositionIndices();
}

//--------------------------------------------------------------------------------------
//! \fn Real Particles::NewBlockTimeStep();
//! \brief returns the time step required by particles in the block.
//!
//! \note
//! The default timestep for particle integration is the cell crossing time. Additional
//! timestep restrictions can be imposed in NewDtForDerived, which can be overrided in
//! child classes. Final timestep is further reduced by multiplying cfl_par_.

Real Particles::NewBlockTimeStep() {
  Coordinates *pc = pmy_block->pcoord;

  // Find the maximum coordinate speed.
  Real dt_inv2_max = 0.0;
  for (int k = 0; k < npar_; ++k) {
    Real dt_inv2 = 0.0, vpx1, vpx2, vpx3;
    pc->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k), vpx(k), vpy(k), vpz(k),
                                    vpx1, vpx2, vpx3);
    dt_inv2 += active1_ ? std::pow(vpx1 / pc->dx1f(static_cast<int>(xi1_(k))), 2) : 0;
    dt_inv2 += active2_ ? std::pow(vpx2 / pc->dx2f(static_cast<int>(xi2_(k))), 2) : 0;
    dt_inv2 += active3_ ? std::pow(vpx3 / pc->dx3f(static_cast<int>(xi3_(k))), 2) : 0;
    dt_inv2_max = std::max(dt_inv2_max, dt_inv2);
  }

  // Return the time step constrained by the coordinate speed (cell crossing time).
  Real dt = dt_inv2_max > 0.0 ? 1.0 / std::sqrt(dt_inv2_max)
                              : std::numeric_limits<Real>::max();

  // Additional time step constraints for derived Particles
  dt = std::min(dt, NewDtForDerived());
  return cfl_par_*dt;
}

//--------------------------------------------------------------------------------------
//! \fn std::size_t Particles::GetSizeInBytes()
//! \brief returns the data size in bytes in the meshblock.

std::size_t Particles::GetSizeInBytes() const {
  std::size_t size = sizeof(npar_);
  if (npar_ > 0) size += npar_ * (nint * sizeof(int) + nreal * sizeof(Real));
  return size;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::MeshBlockIndex()
//! \brief returns the local meshblock indices of the particle-containing cell.

// TODO(SMOON) Remove code duplication with UpdatePositionIndices.
void Particles::MeshBlockIndex(Real xp, Real yp, Real zp,
  int &ip, int &jp, int &kp) const {
  // Convert to the Mesh coordinates.
  Real x1, x2, x3;
  pmy_block->pcoord->CartesianToMeshCoords(xp, yp, zp, x1, x2, x3);
  // Convert to the index space.
  Real x1i, x2i, x3i;
  pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, x1i, x2i, x3i);
  // Convert to the integer indices
  ip = static_cast<int>(std::floor(x1i));
  jp = static_cast<int>(std::floor(x2i));
  kp = static_cast<int>(std::floor(x3i));
}

//--------------------------------------------------------------------------------------
//! \fn Particles::UnpackParticlesForRestart()
//! \brief reads the particle data from the restart file.

void Particles::UnpackParticlesForRestart(char *mbdata, std::size_t &os) {
  // Read number of particles.
  std::memcpy(&npar_, &(mbdata[os]), sizeof(npar_));
  os += sizeof(npar_);
  if (nparmax_ < npar_)
    UpdateCapacity(npar_);

  if (npar_ > 0) {
    // Read integer properties.
    std::size_t size = npar_ * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(&(intprop(k,0)), &(mbdata[os]), size);
      os += size;
    }

    // Read real properties.
    size = npar_ * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(&(realprop(k,0)), &(mbdata[os]), size);
      os += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::PackParticlesForRestart()
//! \brief pack the particle data for restart dump.

void Particles::PackParticlesForRestart(char *&pdata) {
  // Write number of particles.
  std::memcpy(pdata, &npar_, sizeof(npar_));
  pdata += sizeof(npar_);

  if (npar_ > 0) {
    // Write integer properties.
    std::size_t size = npar_ * sizeof(int);
    for (int k = 0; k < nint; ++k) {
      std::memcpy(pdata, &(intprop(k,0)), size);
      pdata += size;
    }
    // Write real properties.
    size = npar_ * sizeof(Real);
    for (int k = 0; k < nreal; ++k) {
      std::memcpy(pdata, &(realprop(k,0)), size);
      pdata += size;
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AddHistoryOutput(Real data_sum[], int pos)
//! \brief finds the data sums of history output from particles in my process and assign
//!   them to data_sum beginning at index pos.

void Particles::AddHistoryOutput(Real data_sum[], int pos) {
  const int NSUM = NHISTORY - 1;

  // Initiate the summations.
  std::int64_t np = 0;
  std::vector<Real> sum(NSUM, 0.0);

  Real vp1, vp2, vp3;
  np += npar_;

  for (int k = 0; k < npar_; ++k) {
    pmy_block->pcoord->CartesianToMeshCoordsVector(xp(k), yp(k), zp(k),
        vpx(k), vpy(k), vpz(k), vp1, vp2, vp3);
    sum[0] += vp1;
    sum[1] += vp2;
    sum[2] += vp3;
    sum[3] += vp1 * vp1;
    sum[4] += vp2 * vp2;
    sum[5] += vp3 * vp3;
    sum[6] += mass(k);
  }

  // Assign the values to output variables.
  data_sum[pos] += static_cast<Real>(np);
  for (int i = 0; i < NSUM; ++i)
    data_sum[pos+i+1] += sum[i];
}

//--------------------------------------------------------------------------------------
//! \fn Particles::OutputParticles()
//! \brief outputs the particle data in tabulated format.

void Particles::OutputParticles() {
  // TODO(SMOON) currently, OutputParticles is called in Mesh::UserWorkInLoop, which is
  // called before the mesh time is updated in main.cpp. Therefore, the "time" in the
  // individual particle output is incorrect; they must be shifted by dt.
  // Call this in MakeOutputs will solve this problem.
  if (!parhstout_)
    return;
  std::stringstream fname, msg;
  std::ofstream os;
  std::string file_basename = pinput->GetString("job","problem_id");

  for (int k = 0; k < npar_; ++k) {
    // Create the filename.
    fname << file_basename << ".par" << pid(k) << ".csv";

    // Open the file for write.
    os.open(fname.str().data(), std::ofstream::app);

    if (!os.is_open()) {
      msg << "### FATAL ERROR in function [Particles::OutputParticles]"
          << std::endl << "Output file '" << fname.str() << "' could not be opened"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // If the file is newly created, write header.
    bool header;
    std::streampos fsize = os.tellp();
    if (fsize == 0) {
      // The file is new, so write a header.
      header = true;
    } else {
      // The file is already there. Don't write header.
      header = false;
    }

    OutputOneParticle(os, k, header);

    // Close the file
    os.close();
    // clear filename
    fname.str("");
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::ToggleParHstOutFlag()
//! \brief turn on individual particle history outputs
void Particles::ToggleParHstOutFlag() {
  if (npar_ < 100) {
    parhstout_ = true;
  } else {
    std::cout << "Warning [Particles]: npar = " << npar_ << " is too large to output"
      << "all individual particles' history automatically."
      << " Particle history output is turned off." << std::endl;
    parhstout_ = false;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AssignShorthands()
//! \brief assigns shorthands by shallow copying slices of the data.

void Particles::AssignShorthands() {
  // public shorthands
  // integer property
  pid.InitWithShallowSlice(intprop, 2, ipid, 1);
  if (pmy_mesh_->shear_periodic)
    sh.InitWithShallowSlice(intprop, 2, ish, 1);
  // real property
  mass.InitWithShallowSlice(realprop, 2, imass, 1);

  xp.InitWithShallowSlice(realprop, 2, ixp, 1);
  yp.InitWithShallowSlice(realprop, 2, iyp, 1);
  zp.InitWithShallowSlice(realprop, 2, izp, 1);
  vpx.InitWithShallowSlice(realprop, 2, ivpx, 1);
  vpy.InitWithShallowSlice(realprop, 2, ivpy, 1);
  vpz.InitWithShallowSlice(realprop, 2, ivpz, 1);

  // aux property
  xp0.InitWithShallowSlice(auxprop, 2, ixp0, 1);
  yp0.InitWithShallowSlice(auxprop, 2, iyp0, 1);
  zp0.InitWithShallowSlice(auxprop, 2, izp0, 1);
  vpx0.InitWithShallowSlice(auxprop, 2, ivpx0, 1);
  vpy0.InitWithShallowSlice(auxprop, 2, ivpy0, 1);
  vpz0.InitWithShallowSlice(auxprop, 2, ivpz0, 1);

  // private shorthands
  // work
  xi1_.InitWithShallowSlice(work, 2, ixi1, 1);
  xi2_.InitWithShallowSlice(work, 2, ixi2, 1);
  xi3_.InitWithShallowSlice(work, 2, ixi3, 1);

  // Assign remaining shorthands for derived particles
  AssignShorthandsForDerived();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::AllocateMemory()
//! \brief memory allocation will be done at the end of derived class initialization
void Particles::AllocateMemory() {
  // Initiate ParticleBuffer class.
  nint_buf = nint;
  nreal_buf = nreal + naux;

  // Allocate mesh auxiliaries.
  ppm = new ParticleMesh(this, pmy_block);

  // Allocate particle gravity
  if (isgravity_) {
    // Add working arrays for gravity forces
    igx = AddWorkingArray();
    igy = AddWorkingArray();
    igz = AddWorkingArray();
    // Activate particle gravity.
    ppgrav = new ParticleGravity(this);
  }

  // Allocate integer properties.
  intprop.NewAthenaArray(nint,nparmax_);

  // Allocate integer properties.
  realprop.NewAthenaArray(nreal,nparmax_);

  // Allocate auxiliary properties.
  if (naux > 0) auxprop.NewAthenaArray(naux,nparmax_);

  // Allocate working arrays.
  if (nwork > 0) work.NewAthenaArray(nwork,nparmax_);
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddIntProperty()
//! \brief adds one integer property to the particles and returns the index.

int Particles::AddIntProperty() {
  return nint++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddRealProperty()
//! \brief adds one real property to the particles and returns the index.

int Particles::AddRealProperty() {
  return nreal++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddAuxProperty()
//! \brief adds one auxiliary property to the particles and returns the index.

int Particles::AddAuxProperty() {
  return naux++;
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::AddWorkingArray()
//! \brief adds one working array to the particles and returns the index.

int Particles::AddWorkingArray() {
  return nwork++;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::UpdateCapacity(int new_nparmax)
//! \brief changes the capacity of particle arrays while preserving existing data.

void Particles::UpdateCapacity(int new_nparmax) {
  // (changgoo) new_nparmax must be smaller than INT_MAX
  if (new_nparmax >= INT_MAX) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::UpdateCapacity]"
        << "Cannot update capacity for " << new_nparmax
        << " that exceeds INT_MAX=" << INT_MAX
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // Increase size of property arrays
  nparmax_ = new_nparmax;
  intprop.ResizeLastDimension(nparmax_);
  realprop.ResizeLastDimension(nparmax_);
  if (naux > 0) auxprop.ResizeLastDimension(nparmax_);
  if (nwork > 0) work.ResizeLastDimension(nparmax_);

  // Reassign the shorthands.
  AssignShorthands();
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::CheckInMeshBlock()
//! \brief check whether given position is within the meshblock assuming Cartesian

bool Particles::CheckInMeshBlock(Real x1, Real x2, Real x3, bool ghost) {
  RegionSize& bsize = pmy_block->block_size;
  Real x1min(bsize.x1min), x1max(bsize.x1max);
  Real x2min(bsize.x2min), x2max(bsize.x2max);
  Real x3min(bsize.x3min), x3max(bsize.x3max);
  if (ghost) {
    x1min -= noverlap_;
    x2min -= noverlap_;
    x3min -= noverlap_;
    x1max += noverlap_;
    x2max += noverlap_;
    x3max += noverlap_;
  }

  if ((x1>=x1min) && (x1<x1max) &&
      (x2>=x2min) && (x2<x2max) &&
      (x3>=x3min) && (x3<x3max)) {
    return true;
  } else {
    return false;
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SaveStatus()
//! \brief saves the current positions and velocities for later use.

void Particles::SaveStatus() {
  for (int k = 0; k < npar_; ++k) {
    // Save current positions.
    xp0(k) = xp(k);
    yp0(k) = yp(k);
    zp0(k) = zp(k);

    // Save current velocities.
    vpx0(k) = vpx(k);
    vpy0(k) = vpy(k);
    vpz0(k) = vpz(k);
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::UpdatePositionIndices()
//! \brief updates position indices of all particles.

void Particles::UpdatePositionIndices() {
  UpdatePositionIndices(0, npar_);
}


//--------------------------------------------------------------------------------------
//! \fn void Particles::UpdatePositionIndices(int k)
//! \brief Updates position indices of particle k

void Particles::UpdatePositionIndices(int k) {
#ifdef DEBUG
  if ((k < 0) || (k >= npar_+npar_gh_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::UpdatePositionIndices]" << std::endl
        << "Index k = " << k << " is ouside the allowed range [0, npar_+npar_gh_)."
        << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  // Convert to the Mesh coordinates.
  Real x1, x2, x3;
  pmy_block->pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

  // Convert to the index space.
  pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1_(k), xi2_(k), xi3_(k));
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::UpdatePositionIndices(int ks, int npar)
//! \brief Updates position indices of particles in the range [ks, ks+npar)

void Particles::UpdatePositionIndices(int ks, int npar) {
#ifdef DEBUG
  if ((ks < 0) || (ks+npar-1 >= npar_+npar_gh_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::UpdatePositionIndices]" << std::endl
        << "Index is ouside the allowed range [0, npar_+npar_gh_)." << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  for (int k = ks; k < ks + npar; ++k) {
    // Convert to the Mesh coordinates.
    Real x1, x2, x3;
    pmy_block->pcoord->CartesianToMeshCoords(xp(k), yp(k), zp(k), x1, x2, x3);

    // Convert to the index space.
    pmy_block->pcoord->MeshCoordsToIndices(x1, x2, x3, xi1_(k), xi2_(k), xi3_(k));
  }
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::ReindexOneParticleAndClear()
//! \brief Reindex a particle at index k=src to k=dst and clear k=src.

void Particles::ReindexOneParticleAndClear(int src, int dst) {
#ifdef DEBUG
  if ((src < 0) || (src >= nparmax_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::ReindexOneParticleAndClear]"
        << std::endl << "src index is ouside the allowed range [0, nparmax_)."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if ((dst < 0) || (dst >= nparmax_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Particles::ReindexOneParticleAndClear]"
        << std::endl << "dst index is ouside the allowed range [0, nparmax_)."
        << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  for (int j = 0; j < nint; ++j) {
    intprop(j,dst) = intprop(j,src);
    intprop(j,src) = 0;
  }
  for (int j = 0; j < nreal; ++j) {
    realprop(j,dst) = realprop(j,src);
    realprop(j,src) = 0.0;
  }
  for (int j = 0; j < naux; ++j) {
    auxprop(j,dst) = auxprop(j,src);
    auxprop(j,src) = 0.0;
  }
  for (int j = 0; j < nwork; ++j) {
    work(j,dst) = work(j,src);
    work(j,src) = 0.0;
  }
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::CountNewParticles()
//! \brief counts new particles in the block.

int Particles::CountNewParticles() const {
  int n = 0;
  for (int i = 0; i < npar_; ++i)
    if (pid(i) == NEW) ++n;
  return n;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::SetNewParticleID(int id0)
//! \brief searches for new particles and assigns ID, beginning at id + 1.

void Particles::SetNewParticleID(int id) {
  for (int i = 0; i < npar_; ++i)
    if (pid(i) == NEW) pid(i) = ++id;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc)
//! \brief evolves the particle positions and velocities by one Euler step.

void Particles::EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc) {
  // Update positions.
  for (int k = 0; k < npar_; ++k) {
    //! \todo (ccyang):
    //! - This is a temporary hack.
    Real tmpx = xp(k), tmpy = yp(k), tmpz = zp(k);
    xp(k) = xp0(k) + dt * vpx(k);
    yp(k) = yp0(k) + dt * vpy(k);
    zp(k) = zp0(k) + dt * vpz(k);
    xp0(k) = tmpx;
    yp0(k) = tmpy;
    zp0(k) = tmpz;
  }

  // Integrate the source terms (e.g., acceleration).
  SourceTerms(t, dt, meshsrc);
  UserSourceTerms(t, dt, meshsrc);
}

//--------------------------------------------------------------------------------------
//! \fn Particles::OutputParticle()
//! \brief outputs the particle data in tabulated format.
void Particles::OutputOneParticle(std::ostream &os, int k, bool header) {
  if (header) {
    os << "time,dt";
    for (int ip = 0; ip < nint; ++ip)
      os << "," << intpropname[ip];
    for (int ip = 0; ip < nreal; ++ip)
      os << "," << realpropname[ip];
    for (int ip = 0; ip < naux; ++ip)
      os << "," << auxpropname[ip];
    os << std::endl;
  }

  // Write the time.
  os << std::scientific << std::showpoint << std::setprecision(18);
  os << pmy_mesh_->time << "," << pmy_mesh_->dt;

  // Write the particle data in the meshblock.
  for (int ip = 0; ip < nint; ++ip)
    os << "," << intprop(ip,k);
  for (int ip = 0; ip < nreal; ++ip)
    os << "," << realprop(ip,k);
  for (int ip = 0; ip < naux; ++ip)
    os << "," << auxprop(ip,k);
  os << std::endl;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ComputePMDensityAndCommunicate(Mesh *pm, bool include_momentum)
//! \brief finds particle mesh densities for all particle containers.
//!
//! If include_momentum is true, the momentum density field is also computed.

void Particles::ComputePMDensityAndCommunicate(Mesh *pm, bool include_momentum) {
  // Assign particle properties to mesh and send boundary.
  int nblocks(pm->nblocal);

  for (int b = 0; b < nblocks; ++b) {
    MeshBlock *pmb(pm->my_blocks(b));
    if (pm->shear_periodic) {
      pmb->pbval->ComputeShear(pm->time, pm->time);
    }
    pmb->pbval->StartReceivingSubset(BoundaryCommSubset::pm,
                                     pmb->pbval->bvars_pm);
    for (Particles *ppar : pmb->ppars) {
      ppar->ppm->ComputePMDensity(include_momentum);
      ppar->ppm->pmbvar->SendBoundaryBuffers();
    }
  }

  for (int b = 0; b < nblocks; ++b) {
    MeshBlock *pmb(pm->my_blocks(b));
    for (Particles *ppar : pmb->ppars) {
      ppar->ppm->pmbvar->ReceiveAndSetBoundariesWithWait();
      if (pm->shear_periodic)
        ppar->ppm->pmbvar->SendShearingBoxBoundaryBuffers();
    }
  }

  if (pm->shear_periodic) {
    for (int b = 0; b < nblocks; ++b) {
      MeshBlock *pmb(pm->my_blocks(b));
      for (Particles *ppar : pmb->ppars) {
        ppar->ppm->pmbvar->ReceiveAndSetShearingBoxBoundariesWithWait();
        ppar->ppm->pmbvar->SetShearingBoxBoundaryBuffers();
      }
    }
  }

  for (int b = 0; b < nblocks; ++b) {
    MeshBlock *pmb(pm->my_blocks(b));
    pmb->pbval->ClearBoundarySubset(BoundaryCommSubset::pm,
                                    pmb->pbval->bvars_pm);
    for (Particles *ppar : pmb->ppars) ppar->ppm->updated=false;
  }
}

//--------------------------------------------------------------------------------------
//! \fn Particles::Initialize(Mesh *pm, ParameterInput *pin)
//! \brief initializes the class.

void Particles::Initialize(Mesh *pm, ParameterInput *pin) {
  if (initialized) return;

  InputBlock *pib = pin->pfirst_block;
  // pm->particle_params.reserve(1);
  // loop over input block names.  Find those that start with "particle", read parameters,
  // and construct singly linked list of ParticleTypes.
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 8, "particle") == 0) {
      ParticleParameters pp;  // define temporary ParticleParameters struct

      // extract integer number of particle block.  Save name and number
      std::string parn = pib->block_name.substr(8); // 7 because counting starts at 0!
      pp.block_number = atoi(parn.c_str());
      pp.block_name.assign(pib->block_name);

      // set particle type = [tracer, star, sink, dust, none]
      pp.partype = pin->GetString(pp.block_name, "type");
      if (pp.partype.compare("none") != 0) { // skip input block if the type is none
        if ((pp.partype.compare("dust") == 0) ||
            (pp.partype.compare("tracer") == 0) ||
            (pp.partype.compare("star") == 0) ||
            (pp.partype.compare("sink") == 0)) {
          pp.ipar = num_particles++;
          idmax.push_back(0); // initialize idmax with 0
          pp.table_output = pin->GetOrAddBoolean(pp.block_name,"output",false);
          pp.gravity = pin->GetOrAddBoolean(pp.block_name,"gravity",false);
          if (pp.table_output) num_particles_output++;
          if (pp.gravity) num_particles_grav++;
          // Set the maximum radius of influence
          // This determines the thickness of overlap region.
          if (pp.partype.compare("sink") == 0) {
            pp.max_rinfl = 1;
          }
          pm->particle_params.push_back(pp);
        } else { // unsupported particle type
          std::stringstream msg;
          msg << "### FATAL ERROR in Particle Initializer" << std::endl
              << "Unrecognized particle type = '" << pp.partype
              << "' in particle block '" << pp.block_name << "'" << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
    pib = pib->pnext;  // move to next input block name
  }

  if (num_particles > 0) {
    pm->particle = true;

    if (SELF_GRAVITY_ENABLED && (num_particles_grav > 0))
      pm->particle_gravity = true;

    if (Globals::my_rank == 0) {
      std::cout << "Particles are initalized with "
                << "N containers = " << num_particles << std::endl;
      for (ParticleParameters ppnew : pm->particle_params)
        std::cout << "  ipar = " << ppnew.ipar
                  << "  partype = " << ppnew.partype
                  << "  output = " << ppnew.table_output
                  << "  block_name = " << ppnew.block_name
                  << std::endl;
    }
    // Remember the pointer to input parameters.
    pinput = pin;

#ifdef MPI_PARALLEL
    // Get my MPI communicator.
    MPI_Comm_dup(MPI_COMM_WORLD, &my_comm);
#endif
  }

  initialized = true;
}

//--------------------------------------------------------------------------------------
//! \fn Particles::PostInitialize(Mesh *pm, ParameterInput *pin)
//! \brief preprocesses the class after problem generator and before the main loop.

void Particles::PostInitialize(Mesh *pm, ParameterInput *pin) {
  // Set particle IDs.
  for (int ipar = 0; ipar < Particles::num_particles; ++ipar)
    ProcessNewParticles(pm, ipar);

  // Set position indices.
  for (int b = 0; b < pm->nblocal; ++b)
    for (Particles *ppar : pm->my_blocks(b)->ppars)
      ppar->UpdatePositionIndices();

  // Print individual particle csv output
  for (int b = 0; b < pm->nblocal; ++b)
    for (Particles *ppar : pm->my_blocks(b)->ppars)
      ppar->OutputParticles();
}

//--------------------------------------------------------------------------------------
//! \fn Particles::FormattedTableOutput()
//! \brief outputs the particle data in tabulated format.

void Particles::FormattedTableOutput(Mesh *pm, OutputParameters op) {
  std::stringstream fname, msg;
  std::ofstream os;

  // Loop over Particle containers
  for (int ipar = 0; ipar < num_particles ; ++ipar) {
    if (pm->particle_params[ipar].table_output) {
      // Loop over MeshBlocks
      for (int b = 0; b < pm->nblocal; ++b) {
        const MeshBlock *pmb(pm->my_blocks(b));
        const Particles *ppar(pmb->ppars[ipar]);

        // Create the filename.
        fname << op.file_basename
              << ".block" << pmb->gid << '.' << op.file_id
              << '.' << std::setw(5) << std::right << std::setfill('0') << op.file_number
              << '.' << "par" << ipar << ".tab";

        // Open the file for write.
        os.open(fname.str().data());
        if (!os.is_open()) {
          msg << "### FATAL ERROR in function [Particles::FormattedTableOutput]"
              << std::endl << "Output file '" << fname.str() << "' could not be opened"
              << std::endl;
          ATHENA_ERROR(msg);
        }

        // Write the time.
        os << std::scientific << std::showpoint << std::setprecision(18);
        os << "# Athena++ particle data at time = " << pm->time << std::endl;

        // Write header.
        os << "# ";
        for (int ip = 0; ip < ppar->nint; ++ip)
          os << ppar->intpropname[ip] << "  ";
        for (int ip = 0; ip < ppar->nreal; ++ip)
          os << ppar->realpropname[ip] << "  ";
        os << std::endl;

        // Write the particle data in the meshblock.
        for (int k = 0; k < ppar->npar_; ++k) {
          for (int ip = 0; ip < ppar->nint; ++ip)
            os << ppar->intprop(ip,k) << "  ";
          for (int ip = 0; ip < ppar->nreal; ++ip)
            os << ppar->realprop(ip,k) << "  ";
          os << std::endl;
        }

        // Close the file and get the next meshblock.
        os.close();
        fname.str("");
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::GetHistoryOutputNames(std::string output_names[])
//! \brief gets the names of the history output variables in history_output_names[].

void Particles::GetHistoryOutputNames(std::string output_names[], int ipar) {
  std::string head = "p";
  head.append(std::to_string(ipar)); // TODO(SMOON) how about partype instead of ipar?
  output_names[0] = head + "-n";
  output_names[1] = head + "-v1";
  output_names[2] = head + "-v2";
  output_names[3] = head + "-v3";
  output_names[4] = head + "-v1sq";
  output_names[5] = head + "-v2sq";
  output_names[6] = head + "-v3sq";
  output_names[7] = head + "-m";
}

//--------------------------------------------------------------------------------------
//! \fn int Particles::GetTotalNumber(Mesh *pm)
//! \brief returns total number of particles (from all processes).
//! \todo This should separately count different types of particles
std::int64_t Particles::GetTotalNumber(Mesh *pm) {
  std::int64_t npartot(0);
  for (int b = 0; b < pm->nblocal; ++b)
    for (Particles *ppar : pm->my_blocks(b)->ppars)
      npartot += ppar->npar_;
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &npartot, 1, MPI_LONG, MPI_SUM, my_comm);
#endif
  return npartot;
}

//--------------------------------------------------------------------------------------
//! \fn void Particles::ProcessNewParticles(Mesh *pmesh, int ipar)
//! \brief searches for and books new particles.

void Particles::ProcessNewParticles(Mesh *pmesh, int ipar) {
  // Count new particles.
  const int nbtotal(pmesh->nbtotal), nblocks(pmesh->nblocal);
  std::vector<int> nnewpar(nbtotal, 0);
  for (int b = 0; b < nblocks; ++b) {
    const MeshBlock *pmb(pmesh->my_blocks(b));
    nnewpar[pmb->gid] = pmb->ppars[ipar]->CountNewParticles();
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &nnewpar[0], nbtotal, MPI_INT, MPI_MAX, my_comm);
#endif

  // Make the counts cumulative.
  for (int i = 1; i < nbtotal; ++i)
    nnewpar[i] += nnewpar[i-1];

  // Set particle IDs.
  for (int b = 0; b < nblocks; ++b) {
    const MeshBlock *pmb(pmesh->my_blocks(b));
    int newid_start = idmax[ipar] + (pmb->gid > 0 ? nnewpar[pmb->gid - 1] : 0);
    pmb->ppars[ipar]->SetNewParticleID(newid_start);
  }
  idmax[ipar] += nnewpar[nbtotal - 1];
}
