#ifndef PARTICLES_PARTICLE_MESH_HPP_
#define PARTICLES_PARTICLE_MESH_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file particle_mesh.hpp
//! \brief defines ParticleMesh class used for communication between meshblocks needed
//!        by particle-mesh methods.

// C++ standard library
#include <cmath>

// Athena++ classes headers
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/cc/pm/bvals_pm.hpp"
#include "../mesh/mesh.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Particle-mesh constants.
const Real RINF = 1;  //!> radius of influence

//! Define the size of a particle cloud = 2 * RINF + 1
#define NPC 3

// Forward declaration
class Particles;
class ParameterInput;

//--------------------------------------------------------------------------------------
//! \class ParticleMesh
//! \brief defines the class for particle-mesh methods

class ParticleMesh {
friend class Particles;
friend class ParticleMeshBoundaryVariable;
friend class DustParticles;
friend class TracerParticles;
friend class StarParticles;
friend class ParticleGravity;
friend class OutputType;

 public:
  // Constructor and destructor
  explicit ParticleMesh(Particles *ppar, MeshBlock *pmb);
  ~ParticleMesh();

  // Interface
  void ComputePMDensity(bool include_momentum=false);
  void DepositPMtoMesh(int stage);
  Real FindMaximumDensity() const;
  AthenaArray<Real> GetMassDensity() const { return dens_; }
  AthenaArray<Real> GetMomentumDensityX1() const { return mom1_; }
  AthenaArray<Real> GetMomentumDensityX2() const { return mom2_; }
  AthenaArray<Real> GetMomentumDensityX3() const { return mom3_; }
  AthenaArray<Real> GetVelocityField() const;

  ParticleMeshBoundaryVariable *pmbvar;

  bool updated; //!> flag whether pm is recalculated

 private:
  // Instance methods
  int AddMeshAux();
  void InterpolateMeshToParticles(
           const AthenaArray<Real>& meshsrc, int ms1,
           AthenaArray<Real>& par, int p1, int nprop);
  void DepositParticlesToMeshAux(
           const AthenaArray<Real>& par, int p1, int ma1, int nprop);
  void DepositMeshAux(AthenaArray<Real>& u, int ma1, int mb1, int nprop);

  // Instance Variables
  int imom1, imom2, imom3;   //!> index to momentum vector in meshaux
  int idens;   //!> index to density in meshaux

  int nmeshaux_;  //!> number of auxiliaries to the meshblock
  int is, ie, js, je, ks, ke;  // beginning and ending indices
  bool active1_, active2_, active3_;  // active dimensions
  Real dxi1_, dxi2_, dxi3_;           // range of influence from a particle cloud
  int nx1_, nx2_, nx3_;               // number of cells in meshaux in each dimension
  int ncells_;                        // total number of cells in meshaux
  int npc1_, npc2_, npc3_;            // size of a particle cloud

  Particles *ppar_;            //!> ptr to my Particles instance
  MeshBlock *pmb_;             //!> ptr to my MeshBlock
  Mesh *pmesh_;                //!> ptr to my Mesh

  AthenaArray<Real> meshaux_, coarse_meshaux_;   //!> auxiliaries to the meshblock
  AthenaArray<Real> dens_, mom1_, mom2_, mom3_;    //!> shorthand to density and momentum
};
#endif  // PARTICLES_PARTICLE_MESH_HPP_
