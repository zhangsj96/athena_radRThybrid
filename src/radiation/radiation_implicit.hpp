#ifndef IMRADIATION_HPP_
#define IMRADIATION_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================


// C++ headers
#include <cstdint>     // int64_t
#include <functional>  // reference_wrapper
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "./radiation.hpp"
#include "./integrators/rad_integrators.hpp"

class Mesh;
class ParameterInput;
class Radiation;
class RadIntegrator;


class IMRadiation {
  friend class Radiation;
  friend class RadIntegrator;
public:
  IMRadiation(Mesh *pm, ParameterInput *pin);
//  ~Radiation();

  void JacobiIteration(Mesh *pm, int stage);


  void CheckResidual(MeshBlock *pmb, Radiation *prad, 
        AthenaArray<Real> &ir_old, AthenaArray<Real> &ir_new);


private:

  Real sum_diff_;
  Real sum_full_;
  int nlimit_;       // threadhold for the number of iterations
  Real error_limit_; // 
  

};

#endif // IMRADIATION_HPP_
