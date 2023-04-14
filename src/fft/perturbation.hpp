#ifndef FFT_PERTURBATION_HPP_
#define FFT_PERTURBATION_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file perturbation.hpp
//  \brief defines PERTURBATION class

// C headers

// C++ headers
#include <random>     // mt19937, normal_distribution, uniform_real_distribution

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "athena_fft.hpp"

class Mesh;
class MeshBlock;
class ParameterInput;
class Coordinates;
class FFTBlock;
class FFTDriver;

//! \class PerturbationBlock
//! \brief MeshBlock level container class for communication purpose
class PerturbationBlock {
friend class PerturbationGenerator;
 public:
  explicit PerturbationBlock(MeshBlock *pmb);
  ~PerturbationBlock();

  AthenaArray<Real> vec, scal;
  AthenaArray<Real> empty_flux[3];

  CellCenteredBoundaryVariable ptbvar;
};

//! \class PerturbationGenerator
//  \brief Perturbation Generator

class PerturbationGenerator : public FFTDriver{
 public:
  PerturbationGenerator(Mesh *pm, ParameterInput *pin);
  ~PerturbationGenerator();
  void GenerateVector();
  void GenerateVector(std::complex<Real> **fv);
  void GenerateScalar();
  void GenerateScalar(std::complex<Real> *fv);
  void AssignVector();
  void AssignVector(std::complex<Real> **fv);
  void AssignScalar();
  void AssignScalar(std::complex<Real> *fv);
  void PowerSpectrum(std::complex<Real> *amp);
  void Project(std::complex<Real> **fv, Real f_shear);
  void Project(std::complex<Real> **fv, std::complex<Real> **fv_sh,
               std::complex<Real> **fv_co);
  void SetBoundary();
  AthenaArray<Real> GetVector(int nb);
  AthenaArray<Real> GetScalar(int nb);
  std::int64_t GetKcomp(int idx, int disp, int Nx);
  AthenaArray<PerturbationBlock*> my_ptblocks;

 protected:
  std::int64_t rseed;
  int nlow, nhigh;
  Real f_shear, expo, dvol;
  bool global_ps_ = false;

  std::complex<Real> **fv_;
  std::complex<Real> **fv_sh_, **fv_co_;
  std::mt19937_64 rng_generator;
};

//! \class TurbulenceDriver
//  \brief Turbulence Driver

class TurbulenceDriver : public PerturbationGenerator{
 public:
  TurbulenceDriver(Mesh *pm, ParameterInput *pin);
  ~TurbulenceDriver();
  void Driving();
  void Generate();
  void Perturb(Real dt);

  int turb_flag;
  Real tdrive, dtdrive, tcorr, dedt;

 private:
  std::complex<Real> **fv_new_;
};

#endif // FFT_PERTURBATION_HPP_
