//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file rad_source.cpp
//  \brief Add radiation source terms to both radiation and gas
//======================================================================================




#include <algorithm>
#include <string>
#include <vector>

//Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../radiation.hpp"

// class header
#include "rad_integrators.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


void RadIntegrator::CheckResidual(MeshBlock *pmb, AthenaArray<Real> &ir_old,
        AthenaArray<Real> &ir_new)
{
  Radiation *prad=pmb->prad;
  
  int &nang =prad->nang;
  int &nfreq=prad->nfreq;


  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real prad->sum_diff = 0.0;
  Real prad->sum_full = 0.0;
  for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      for(int i=is; i<=ie; ++i){
        for(int ifr=0; ifr<nfreq; ++ifr){
          Real *iro = &(ir_old(k,j,i,ifr*nang));
          Real *irn = &(ir_new(k,j,i,ifr*nang));
          for(int n=0; n<nang; ++n){
            sum_diff += abs(iro[n] - irn[n]);
            sum_full += abs(irn[n]);
          }
        }// end ifr
      }// end i
    }// end j
  }// end k

  // store the residual in meshblock
  // each core can have multiple meshblocks
  
}







