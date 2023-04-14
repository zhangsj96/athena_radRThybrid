#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
//======================================================================================
//! \file particles.hpp
//! \brief defines classes for particle dynamics.
//======================================================================================

// C/C++ Standard Libraries
#include <limits>
#include <string>
#include <vector>

// Athena headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "particle_buffer.hpp"
#include "particle_gravity.hpp"
#include "particle_mesh.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Forward definitions
class ParticleGravity;

//--------------------------------------------------------------------------------------
//! \struct Neighbor
//  \brief defines a structure for links to neighbors

struct Neighbor {
  NeighborBlock *pnb;
  MeshBlock *pmb;
  Neighbor *next, *prev;

  Neighbor() : pnb(NULL), pmb(NULL), next(NULL), prev(NULL) {}
};

//----------------------------------------------------------------------------------------
//! \struct ParticleParameters
//! \brief container for parameters read from `<particle?>` block in the input file

struct ParticleParameters {
  // max_rinfl is the maximum integer radius of influence of a particle.
  // max_rinfl = -1 for a particle that has no influence at all, i.e., it does not modify
  // the fluid variables at grid cells.
  // max_rinfl = 0 means that a particle only modifies the cell in which it is contained.
  int block_number, ipar, max_rinfl;
  bool table_output, gravity;
  std::string block_name;
  std::string partype;
  // TODO(SMOON) Add nhistory variable
  ParticleParameters() : block_number(0), ipar(-1), max_rinfl(-1), table_output(false),
                         gravity(false) {}
};

//--------------------------------------------------------------------------------------
//! \class Particles
//! \brief defines the base class for all implementations of particles.

class Particles {
friend class ParticleGravity;
friend class ParticleMesh;

 public:
  // TODO(SMOON) this must be variable, initialized through ParticleParameters,
  // because different particle species may have different number of history output.
  static const int NHISTORY = 8;  //!> number of variables in history output

  // Constructor
  Particles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  virtual ~Particles();

  // Particle interface
  int AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3,
                     bool allow_ghost=false);
  void RemoveOneParticle(int k);
  virtual void Integrate(int step);
  virtual void CreateInLoop() {}
  Real NewBlockTimeStep();
  std::size_t GetSizeInBytes() const;
  bool IsGravity() const { return isgravity_; }
  int GetNumPar() const { return npar_; }
  void MeshBlockIndex(Real xp, Real yp, Real zp, int &ip, int &jp, int &kp) const;
  // InteractWithMesh must use conservative variables instead of primitive variables,
  // because this will be called before ConservedToPrimitive in the time integrator
  // task list.
  virtual void InteractWithMesh() {}

  // Input/Output interface
  void UnpackParticlesForRestart(char *mbdata, std::size_t &os);
  void PackParticlesForRestart(char *&pdata);
  void AddHistoryOutput(Real data_sum[], int pos);
  void OutputParticles(); // individual particle history;
  void ToggleParHstOutFlag();

  // Boundary communication interface (defined in particles_bvals.cpp)
  void ClearBoundary();
  void ClearNeighbors();
  void LinkNeighbors(MeshBlockTree &tree, int64_t nrbx1, int64_t nrbx2, int64_t nrbx3,
                     int root_level);
  void LoadParticleBuffer(ParticleBuffer *ppb, int k);
#ifdef MPI_PARALLEL
  void SendParticleBuffer(ParticleBuffer& send, int dst);
  void ReceiveParticleBuffer(int nb_rank, ParticleBuffer& recv,
                             enum BoundaryStatus& bstatus);
#endif
  void SendToNeighbors();
  bool ReceiveFromNeighbors(bool ghost=false);
  void SendGhostParticles();
  void StartReceivingParticlesShear();
  void SendParticlesShear();
  int FindTargetGidAlongX2(Real x2);
  void ClearBoundaryShear();
  bool ReceiveFromNeighborsShear();

  // Static functions
  static void AMRCoarseToFine(Particles *pparc, Particles *pparf, MeshBlock* pmbf);
  static void AMRFineToCoarse(Particles *pparc, Particles *pparf);
  static void ComputePMDensityAndCommunicate(Mesh *pm, bool include_momentum);
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void PostInitialize(Mesh *pm, ParameterInput *pin);
  static void FormattedTableOutput(Mesh *pm, OutputParameters op);
  static void GetHistoryOutputNames(std::string output_names[], int ipar);
  static std::int64_t GetTotalNumber(Mesh *pm);
  static void ProcessNewParticles(Mesh *pmesh, int ipar);

  // Data members
  // number of particle containers
  static int num_particles, num_particles_grav, num_particles_output;
  ParticleMesh *ppm;  //!> ptr to particle-mesh
  const int ipar;     // index of this Particle in ppars vector
  std::string input_block_name, partype;

  // Shallow slices of the actual data container (intprop, realprop, auxprop, work)
  AthenaArray<int> pid, sh;                  //!> particle ID
  AthenaArray<Real> mass;                //!> mass
  AthenaArray<Real> xp, yp, zp;        //!> position
  AthenaArray<Real> vpx, vpy, vpz;     //!> velocity
  AthenaArray<Real> xp0, yp0, zp0;     //!> position at the previous timestep
  AthenaArray<Real> vpx0, vpy0, vpz0;  //!> velocity at the previous timestep

 protected:
  // Protected interfaces (to be used by derived classes)
  // SMOON: A possibility of forgetting to call these two functions may indicate
  // an antipattern.
  void AssignShorthands(); //!> Needs to be called in the derived class constructor
  void AllocateMemory();   //!> Needs to be called in the derived class constructor
  int AddIntProperty();
  int AddRealProperty();
  int AddAuxProperty();
  int AddWorkingArray();
  void UpdateCapacity(int new_nparmax);  //!> Change the capacity of particle arrays
  bool CheckInMeshBlock(Real x1, Real x2, Real x3, bool ghost=false);
  void SaveStatus(); // x->x0, v->v0
  void UpdatePositionIndices();
  void UpdatePositionIndices(int k);
  void UpdatePositionIndices(int ks, int npar);


  int npar_;     //!> number of particles
  int npar_gh_;     //!> number of ghost particles
  int nparmax_;  //!> maximum number of particles per meshblock
  const int req_nghost_; //!> required number of ghost cells for hydro/MHD
  const int noverlap_; //!> minimum thickness of the overlap region
  // When a particle enters the "overlap region" where its region of influence overlaps
  // with the ghost cells of neighboring MeshBlock, that particle is sent to the neighbor
  // MeshBlock as a "ghost particle". The variable "noverlap_" defines the thickness of
  // the overlap region, such that the overlap region in the x1 direction is:
  //   [is, is+noverlap_-1] and [ie-noverlap_+1, ie].
  //
  // For example, consider a star particle with the feedback radius = 1. When it explodes
  // at i = ie - 2 in MeshBlock "A" (see the diagram below), it modifies the active cell
  // i = ie - 1 of the MeshBlock A, which corresponds to the ghost cell i = is - 2 of the
  // neighboring MeshBlock "B".
  //
  //               --------- MeshBlock A ---|--- MeshBlock B -----
  //  particle location |<---|--x-|--->|    |    |    |
  //      index in A    |ie-3|ie-2|ie-1| ie |ie+1|ie+2|
  //      index in B    |is-4|is-3|is-2|is-1| is |is+1|
  //
  // Because, e.g., PLM requires 2 ghost cells for hydro/MHD, MeshBlock B needs
  // noverlap_ >= 3 to fully update its 2 ghost cells by receiving the ghost particle
  // from MeshBlock A.
  // In general, if particle modifies the fluid variable, noverlap_ should be at least:
  // (number of ghost cells required for hydro/MHD) + (radius of influence of a particle).
  // It is developer's responsibility to set the maximum radius of influence in
  // [Particles::Initialize].

  Real cfl_par_;  //!> CFL number for particles

  ParticleGravity *ppgrav; //!> ptr to particle-gravity
  MeshBlock* pmy_block;  //!> MeshBlock pointer
  Mesh* pmy_mesh_;        //!> Mesh pointer

  // shearing box parameters
  Real Omega_0_, qshear_, qomL_;
  int ShBoxCoord_;
  bool orbital_advection_defined_;

  // special tags
  enum ParticleIDTag {NEW=-1, DEL=-2};

  // The actual data storage of all particle properties
  // Note to developers:
  // Direct access to these containers is discouraged; use shorthands instead.
  // e.g.) use mass(k) instead of realprop(imass, k)
  // Auxiliary properties (auxprop) is communicated when particles moving to
  // another meshblock. Working arrays (work) is not communicated.
  std::vector<std::string> intpropname, realpropname, auxpropname;
  AthenaArray<int> intprop;
  AthenaArray<Real> realprop, auxprop, work;

 private:
  // Methods (implementation)
  // Need to be implemented in derived classes
  virtual void AssignShorthandsForDerived()=0;
  virtual void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                             AthenaArray<Real>& meshdst) {}
  void ReindexOneParticleAndClear(int src, int dst);
  int CountNewParticles() const;
  void SetNewParticleID(int id);
  // hooks for further timestep constraints for derived particles
  virtual Real NewDtForDerived() { return std::numeric_limits<Real>::max(); }
  void EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc);

  // Input/Output
  void OutputOneParticle(std::ostream &os, int k, bool header);

  // boundary conditions (implemented in particles_bvals.cpp)
  void ApplyBoundaryConditions(int k);
  void ApplyBoundaryConditionsGhost(int k, int ox1, int ox2, int ox3);
  void FlushReceiveBuffer(ParticleBuffer& recv, bool ghost=false);
  struct Neighbor* FindTargetNeighbor(
      int ox1, int ox2, int ox3, int xi1, int xi2, int xi3);
  void ApplyBoundaryConditionsShear(int k, Real &x1, Real &x2, Real &x3);

  // Class variable
  static std::vector<int> idmax; //!> largest particle ID for each Particles
  static bool initialized;  //!> whether or not the class is initialized
  static ParameterInput *pinput;

  // Data members
  int nint;          //!> numbers of integer particle properties
  int nreal;         //!> numbers of real particle properties
  int naux;          //!> number of auxiliary particle properties
  int nwork;         //!> number of working arrays for particles
  int nint_buf, nreal_buf; //!> number of properties for buffer

  // indices for intprop shorthands
  int ipid;                 // index for the particle ID
  int ish;                  // index for shear boundary flag

  // indices for realprop shorthands
  int imass;                // index for the particle mass
  int ixp, iyp, izp;        // indices for the position components
  int ivpx, ivpy, ivpz;     // indices for the velocity components

  // indices for auxprop shorthands
  int ixp0, iyp0, izp0;     // indices for beginning position components
  int ivpx0, ivpy0, ivpz0;  // indices for beginning velocity components

  // indices for work shorthands
  int ixi1, ixi2, ixi3;     // indices for position indices
  int igx, igy, igz; // indices for gravity force

  AthenaArray<Real> xi1_, xi2_, xi3_;     //!> position indices in local meshblock

  bool parhstout_; //!> flag for individual particle history output
  bool isgravity_; //!> flag for gravity
  bool active1_, active2_, active3_;  // active dimensions

  // MeshBlock-to-MeshBlock communication:
  BoundaryValues *pbval_;                            //!> ptr to my BoundaryValues
  Neighbor neighbor_[3][3][3];                       //!> links to neighbors
  ParticleBuffer recv_[56], recv_gh_[56], recv_sh_[8];   //!> particle receive buffers
  enum BoundaryStatus bstatus_[56], bstatus_gh_[56];  //!> boundary status
  enum BoundaryStatus bstatus_recv_sh_[8];            //!> boundary status for shearing BC

#ifdef MPI_PARALLEL
  static MPI_Comm my_comm;   //!> my MPI communicator
  ParticleBuffer send_[56], send_gh_[56], send_sh_[8];  //!> particle send buffers
  enum BoundaryStatus bstatus_send_sh_[8];  //!> comm. flags
#endif
};

//--------------------------------------------------------------------------------------
//! \class DustParticles
//! \brief defines the class for dust particles that interact with the gas via drag
//!        force.

class DustParticles : public Particles {
 public:
  // Constructor
  DustParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~DustParticles();

  // Methods (interface)
  bool GetBackReaction() const { return backreaction; }
  bool GetDragForce() const { return dragforce; }
  bool IsVariableTaus() const { return variable_taus; }
  Real GetStoppingTime() const { return taus0; }

  // Data members
  // shorthand for additional properties
  AthenaArray<Real> taus;              // shorthand for stopping time

 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                     AthenaArray<Real>& meshdst) override;
  void UserStoppingTime(Real t, Real dt, const AthenaArray<Real>& meshsrc);
  Real NewDtForDerived() override;

  // Data members
  bool backreaction;   //!> turn on/off back reaction
  bool dragforce;      //!> turn on/off drag force
  bool variable_taus;  //!> whether or not the stopping time is variable

  // indicies for additional shorthands
  int itaus;                 // index for stopping time
  int iwx, iwy, iwz;         // indices for working arrays

  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays

  Real taus0;  //!> constant/default stopping time (in code units)
};

//--------------------------------------------------------------------------------------
//! \class TracerParticles
//! \brief defines the class for velocity Tracer particles

class TracerParticles : public Particles {
 public:
  // Constructor
  TracerParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~TracerParticles();

  // Methods (interface)

  // Data members
  // shorthand for additional properties


 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;

  // Data members
  // indicies for additional shorthands
  int iwx, iwy, iwz;         // indices for working arrays

  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays
};

//--------------------------------------------------------------------------------------
//! \class StarParticles
//! \brief defines the class for Star particles

class StarParticles : public Particles {
 public:
  // Constructor
  StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~StarParticles();

  // Methods (interface)
  void Integrate(int step) override;

  // Data members
  // shorthand for additional properties
  AthenaArray<Real> metal, age, fgas;

 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;

  void VL2DKD(int step);
  void RK2KDK(int step);
  void RK2(int step);
  void Kick(Real t, Real dt);
  void Drift(Real t, Real dt);
  void BorisKick(Real t, Real dt);
  void Age(Real t, Real dt);

  void ExertTidalForce(Real t, Real dt);
  void PointMass(Real t, Real dt, Real gm);
  void ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3);

  // Data members
  int imetal, iage, ifgas;  // indices for metalicity, age, and gas fraction
  std::string hydro_integrator; // hydro integrator (vl2 or rk2)
};

//--------------------------------------------------------------------------------------
//! \class SinkParticles
//! \brief defines the class for Sink particles

class SinkParticles : public StarParticles {
 public:
  // Constructor
  SinkParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~SinkParticles();

  // Methods (interface)
  void CreateInLoop() override;
  void InteractWithMesh() override;
  void Accrete();
  void SetControlVolume();

  // Data members
  const int rctrl = 1; // Extent of the control volume. The side length of the control
                       // volume is 2*rctrl + 1.
 private:
  // Methods (implementation)
  bool IsControlVolumeOverlap(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2,
                              bool &flag_strong);
  bool TestMerger(int &par1_index, int &par2_index, bool &strong_overlap);
  void MergeTwoParticles(int i, int j, bool strong_overlap);
  void Merge();
  int Accrete(int n, bool old_pos=false);
  void SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp);

  // Data members
};

#endif  // PARTICLES_PARTICLES_HPP_
