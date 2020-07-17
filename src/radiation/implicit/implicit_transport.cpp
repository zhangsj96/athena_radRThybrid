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
//! \file rad_transport.cpp
//  \brief implementation of radiation integrators
//======================================================================================


// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../radiation.hpp"
#include "../../coordinates/coordinates.hpp" //
#include "../../reconstruct/reconstruction.hpp"


// class header
#include "../integrators/rad_integrators.hpp"



// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// flux at each interface is given as
// x flux: alpha_i-1 I(i-1) + alpha_1 I + alpha_i+1 I(i+1)
// y flux: alpha_j-1 I(j-1) + alpha_2 I + alpha_j+1 I(j+1)
// z flux: alpha_k-1 I(k-1) + alpha_3 I + alpha_k+1 I(k+1)

// the flux should be Area * I^{i+1/2}/Vol

void RadIntegrator::FirstOrderFluxDivergence(const Real wght, 
                                       AthenaArray<Real> &ir)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;

  AthenaArray<Real> &x1flux=prad->flux[X1DIR];
  AthenaArray<Real> &x2flux=prad->flux[X2DIR];
  AthenaArray<Real> &x3flux=prad->flux[X3DIR];

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
                 &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
                 &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

  AthenaArray<Real> &area_zeta = zeta_area_, &area_psi = psi_area_, 
                 &ang_vol = ang_vol_, &dflx_ang = dflx_ang_;
  int &nzeta = prad->nzeta, &npsi = prad->npsi;

  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) {

      // calculate x1-flux divergence 
      pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
      for(int i=is; i<=ie; ++i){
        for(int n=0; n<prad->n_fre_ang; ++n){
          // first left flux
          Real vel_l = velx_(k,j,i,n);
          Real vel_r = velx_(k,j,i+1,n);
          Real lflux = 0.0;
          Real rflux = 0.0;
          const_coef1_(k,j,i,n) = 0.0;
          if(vel_l > 0.0)
            lflux = vel_l * ir(k,j,i-1,n);
          else if(vel_l < 0.0){
            const_coef1_(k,j,i,n) = (-x1area(i) * vel_l); 
          }

          if(vel_r > 0.0){
            const_coef1_(k,j,i,n) += (x1area(i+1) * vel_r);
          }else if(vel_r < 0.0){
            rflux = vel_r * ir(k,j,i+1,n);
          }

          dflx(i,n) = (x1area(i+1) * rflux - x1area(i) * lflux);
        }// end n
      }// End i

     // calculate x2-flux
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
        for(int i=is; i<=ie; ++i){
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vel_l = vely_(k,j,i,n);
            Real vel_r = vely_(k,j+1,i,n);
            Real lflux = 0.0;
            Real rflux = 0.0;
            const_coef2_(k,j,i,n) = 0.0;
            if(vel_l > 0.0)
              lflux = vel_l * ir(k,j-1,i,n);
            else if(vel_l < 0.0)
              const_coef2_(k,j,i,n) = (-x2area(i) * vel_l);
            if(vel_r > 0.0)
              const_coef2_(k,j,i,n) += (x2area_p1(i) * vel_r);
            else if(vel_r < 0.0)
              rflux = vel_r * ir(k,j+1,i,n); 

            dflx(i,n) += (x2area_p1(i)*rflux - x2area(i)*lflux);
          }// end n
        }// end i
      }// end nx2

      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for(int i=is; i<=ie; ++i){
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vel_l = velz_(k,j,i,n);
            Real vel_r = velz_(k+1,j,i,n);
            Real lflux = 0.0;
            Real rflux = 0.0;
            const_coef3_(k,j,i,n) = 0.0;
            if(vel_l > 0.0)
              lflux = vel_l * ir(k-1,j,i,n);
            else if(vel_l < 0.0)
              const_coef3_(k,j,i,n) = (-x3area(i) * vel_l);
            if(vel_r > 0.0)
              const_coef3_(k,j,i,n) += (x3area_p1(i) * vel_r);
            else if(vel_r < 0.0)
              rflux = vel_r * ir(k+1,j,i,n); 

            dflx(i,n) += (x3area_p1(i)*rflux - x3area(i)*lflux);
          }
        }
      }// end nx3
      // update variable with flux divergence
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for(int i=is; i<=ie; ++i){
        Real *divn = &(divflx_(k,j,i,0));
        Real *flxn = &(dflx(i,0));
        Real dtvol = wght/vol(i);
#pragma omp simd aligned(divn,flxn:ALI_LEN)
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] = -dtvol*flxn[n];
        }
        // multiple the matrix coefficient with dt/vol
        Real *coef1n = &(const_coef1_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          coef1n[n] *= dtvol;
        }
        if(pmb->block_size.nx2 > 1){
          Real *coef2n = &(const_coef2_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef2n[n] *= dtvol;
          }          
        }// end nx2 > 1
        if(pmb->block_size.nx3 > 1){
          Real *coef3n = &(const_coef3_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef3n[n] *= dtvol;
          }          
        }// end nx3 > 1
      }// end i
    }// end j
  }// End k
  
}
