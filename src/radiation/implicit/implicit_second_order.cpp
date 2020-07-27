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

void RadIntegrator::SecondOrderFluxDivergence(const Real wght, 
                                       AthenaArray<Real> &ir)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco = pmb->pcoord;
  Reconstruction *prec = pmb->precon;

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

      // First, calculate the limiter based on variable at last step
      for(int i=is-1; i<=ie+1; ++i){
        Real *qn = &(ir(k,j,i,0));
        Real *ql = &(ir(k,j,i-1,0));
        Real *qr = &(ir(k,j,i+1,0));
        Real *dql = &(dql_(0));
        Real *dqr = &(dqr_(0));
        Real *limn= &(limiter_(i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          dql[n] = qn[n] - ql[n];
          dqr[n] = qr[n] - qn[n];
        }
        if (prec->uniform[X1DIR] && !prec->curvilinear[X1DIR]) {
          for (int n=0; n<prad->n_fre_ang; ++n) {
            Real dq2 = dql[n]*dqr[n];
            limn[n] = 2.0*dq2/(dql[n] + dqr[n]);
            if (dq2 <= 0.0) limn[n] = 0.0;
          }
        } else {
          Real cf = pco->dx1v(i  )/(pco->x1f(i+1) - pco->x1v(i)); // (Mignone eq 33)
          Real cb = pco->dx1v(i-1)/(pco->x1v(i  ) - pco->x1f(i));  
          Real dxF = pco->dx1f(i)/pco->dx1v(i);
          Real dxB = pco->dx1f(i)/pco->dx1v(i-1);  
          for (int n=0; n<prad->n_fre_ang; ++n) {
            Real dqF =  dqr[n]*dxF;
            Real dqB =  dql[n]*dxB;
            Real dq2 = dqF*dqB;
        // (modified) VL limiter (Mignone eq 37)
        // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
            limn[n] = (dq2*(cf*dqB + cb*dqF)/
                    (SQR(dqB) + SQR(dqF) + dq2*(cf + cb - 2.0)));
            if (dq2 <= 0.0) limn[n] = 0.0; // ---> no concern for divide-by-0 in above line

          }
        }// end uniform or not
      }// end i

      // calculate x1-flux divergence 
      pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
      for(int i=is; i<=ie; ++i){
        Real ratio_l1 = (pco->x1f(i) - pco->x1v(i-1))/pco->dx1f(i-1);
        Real ratio_l=(pco->x1f(i+1) - pco->x1v(i))/pco->dx1f(i);
        Real ratio_r=(pco->x1v(i  ) - pco->x1f(i))/pco->dx1f(i);
        Real ratio_r1 = (pco->x1v(i+1) - pco->x1f(i+1))/pco->dx1f(i+1);
        Real *vel_ln = &(velx_(k,j,i,0));
        Real *vel_rn = &(velx_(k,j,i+1,0));
        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *irln = &(ir(k,j,i-1,0));
        Real *irrn = &(ir(k,j,i+1,0));
        Real *limln = &(limiter_(i-1,0));
        Real *limn = &(limiter_(i,0));
        Real *limrn = &(limiter_(i+1,0));
        Real areal = x1area(i);
        Real arear = x1area(i+1);
        Real *flxn = &(dflx(i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          // first left flux
          Real lflux = 0.0;
          Real rflux = 0.0;
          coef1n[n] = 0.0;
          if(vel_ln[n] > 0.0){
            Real il = irln[n] + ratio_l1 * limln[n];
            lflux = vel_ln[n] * il;
          }
          else if(vel_ln[n] < 0.0){
            lflux = vel_ln[n] * (-ratio_r * limn[n]); 
            coef1n[n] = (-areal * vel_ln[n]); 
          }

          if(vel_rn[n] > 0.0){
            coef1n[n] += (arear * vel_rn[n]);
            rflux = vel_rn[n] * (ratio_l * limn[n]);
          }else if(vel_rn[n] < 0.0){
            Real ir = irrn[n] - ratio_r1 * limrn[n];
            rflux = vel_rn[n] * ir;
          }

          flxn[n] = (arear * rflux - areal * lflux);
        }// end n
      }// End i

      for(int i=is; i<=ie; ++i){
        Real *divn = &(divflx_(k,j,i,0));
        Real *flxn = &(dflx(i,0));
#pragma omp simd aligned(divn,flxn:ALI_LEN)
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] = -flxn[n];
        }
      }// end i
    }// end j
  }// end k

  if (pmb->block_size.nx2 > 1) {
    for(int k=ks; k<=ke; ++k){
    // first, calculate limiter
      for(int j=js-1; j<=je+1; ++j){
        Real cf = 1.0;
        Real cb = 1.0;
        Real dxF = 1.0;
        Real dxB = 1.0;
        if (!prec->uniform[X2DIR] || prec->curvilinear[X2DIR]){
          cf = pco->dx2v(j  )/(pco->x2f(j+1) - pco->x2v(j)); // (Mignone eq 33)
          cb = pco->dx2v(j-1)/(pco->x2v(j  ) - pco->x2f(j));  
          dxF = pco->dx2f(j)/pco->dx2v(j);
          dxB = pco->dx2f(j)/pco->dx2v(j-1);  
        }

        for(int i=is; i<=ie; ++i){
          Real *qn = &(ir(k,j,i,0));
          Real *ql = &(ir(k,j-1,i,0));
          Real *qr = &(ir(k,j+1,i,0));
          Real *dql = &(dql_(0));
          Real *dqr = &(dqr_(0));
          Real *limn= &(limiterj_(j,i,0));
#pragma omp simd aligned(dql,dqr,qn,qr,ql:ALI_LEN)
          for(int n=0; n<prad->n_fre_ang; ++n){
            dql[n] = qn[n] - ql[n];
            dqr[n] = qr[n] - qn[n];
          }
          if (prec->uniform[X2DIR] && !prec->curvilinear[X2DIR]) {
            for (int n=0; n<prad->n_fre_ang; ++n) {
              Real dq2 = dql[n]*dqr[n];
              limn[n] = 2.0*dq2/(dql[n] + dqr[n]);
              if (dq2 <= 0.0) limn[n] = 0.0;
            }
          } else {
            for (int n=0; n<prad->n_fre_ang; ++n) {
              Real dqF =  dqr[n]*dxF;
              Real dqB =  dql[n]*dxB;
              Real dq2 = dqF*dqB;
          // (modified) VL limiter (Mignone eq 37)
          // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
              limn[n] = (dq2*(cf*dqB + cb*dqF)/
                      (SQR(dqB) + SQR(dqF) + dq2*(cf + cb - 2.0)));
              if (dq2 <= 0.0) limn[n] = 0.0; // ---> no concern for divide-by-0 in above line

            }
          }// end uniform or not
        }// end i
      }// end j

      // now construct the flux
      for(int j=j; j<=je; ++j){
        Real dxp1 = (pco->x2f(j) - pco->x2v(j-1))/pco->dx2f(j-1);
        Real dxp = (pco->x2f(j+1) - pco->x2v(j))/pco->dx2f(j);
        Real dxm = (pco->x2v(j  ) - pco->x2f(j))/pco->dx2f(j);
        Real dxm1 = (pco->x2v(j+1) - pco->x2f(j+1))/pco->dx2f(j+1);
        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);

        for(int i=is; i<=ie; ++i){
          Real *divn = &(divflx_(k,j,i,0));
          Real *flxn = &(dflx(i,0));
          Real arear = x2area_p1(i);
          Real areal = x2area(i);
          Real *irln = &(ir(k,j-1,i,0));
          Real *irrn = &(ir(k,j+1,i,0));
          Real *vel_ln = &(vely_(k,j,i,0));
          Real *vel_rn = &(vely_(k,j+1,i,0));
          Real *limn = &(limiterj_(j,i,0));
          Real *limln = &(limiterj_(j-1,i,0));
          Real *limrn = &(limiterj_(j+1,i,0));
          Real *coef2n = &(const_coef2_(k,j,i,0));

          for(int n=0; n<prad->n_fre_ang; ++n){
         // first left flux
            Real lflux = 0.0;
            Real rflux = 0.0;
            coef2n[n] = 0.0;
            if(vel_ln[n] > 0.0){
              Real il = irln[n] + dxp1 * limln[n];
              lflux = vel_ln[n] * il;
            }
            else if(vel_ln[n] < 0.0){
              lflux = vel_ln[n] * (-dxm * limn[n]); 
              coef2n[n] = (-areal * vel_ln[n]); 
            }

            if(vel_rn[n] > 0.0){
              coef2n[n] += (arear * vel_rn[n]);
              rflux = vel_rn[n] * (dxp * limn[n]);
            }else if(vel_rn[n] < 0.0){
              Real ir = irrn[n] - dxm1 * limrn[n];
              rflux = vel_rn[n] * ir;
            }

            flxn[n] = (arear * rflux - areal * lflux);
          }// end n 
#pragma omp simd aligned(divn,flxn:ALI_LEN)
          for(int n=0; n<prad->n_fre_ang; ++n){
            divn[n] -= flxn[n];
          }
        }
      }// end j  

    }// end k
  }// end 2D


  if (pmb->block_size.nx3 > 1) {
    for(int k=ks-1; k<=ke+1; ++k){
    // first, calculate limiter
      Real dxF = 1.0;
      Real dxB = 1.0;
      if (!prec->uniform[X3DIR]){
        dxF = pco->dx3f(k)/pco->dx3v(k);
        dxB = pco->dx3f(k)/pco->dx3v(k-1);  
      }
      for(int j=js; j<=je; ++j){
        for(int i=is; i<=ie; ++i){
          Real *qn = &(ir(k,j,i,0));
          Real *ql = &(ir(k-1,j,i,0));
          Real *qr = &(ir(k+1,j,i,0));
          Real *dql = &(dql_(0));
          Real *dqr = &(dqr_(0));
          Real *limn= &(limiterk_(k,j,i,0));
#pragma omp simd aligned(dql,dqr,qn,qr,ql:ALI_LEN)
          for(int n=0; n<prad->n_fre_ang; ++n){
            dql[n] = qn[n] - ql[n];
            dqr[n] = qr[n] - qn[n];
          }
          if (prec->uniform[X3DIR]) {
            for (int n=0; n<prad->n_fre_ang; ++n) {
              Real dq2 = dql[n]*dqr[n];
              limn[n] = 2.0*dq2/(dql[n] + dqr[n]);
              if (dq2 <= 0.0) limn[n] = 0.0;
            }
          } else {
            for (int n=0; n<prad->n_fre_ang; ++n) {
              Real dqF =  dqr[n]*dxF;
              Real dqB =  dql[n]*dxB;
              Real dq2 = dqF*dqB;
          // (modified) VL limiter (Mignone eq 37)
          // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
              limn[n] = 2.0*dq2/(dqF + dqB);

              if (dq2 <= 0.0) limn[n] = 0.0; // ---> no concern for divide-by-0 in above line

            }
          }// end uniform or not
        }// end i
      }// end j
    }// end k

    // now construct the flux
    for(int k=ks; k<=ke; ++k){
      Real dxp1 = (pco->x3f(k) - pco->x3v(k-1))/pco->dx3f(k-1);
      Real dxp = (pco->x3f(k+1) - pco->x3v(k))/pco->dx3f(k);
      Real dxm = (pco->x3v(k  ) - pco->x3f(k))/pco->dx3f(k);
      Real dxm1 = (pco->x3v(k+1) - pco->x3f(k+1))/pco->dx3f(k+1);

      for(int j=j; j<=je; ++j){
        pmb->pcoord->Face3Area(k,j  ,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);

        for(int i=is; i<=ie; ++i){
          Real *divn = &(divflx_(k,j,i,0));
          Real *flxn = &(dflx(i,0));
          Real arear = x3area_p1(i);
          Real areal = x3area(i);
          Real *irln = &(ir(k-1,j,i,0));
          Real *irrn = &(ir(k+1,j,i,0));
          Real *vel_ln = &(velz_(k,j,i,0));
          Real *vel_rn = &(velz_(k+1,j,i,0));
          Real *limn = &(limiterk_(k,j,i,0));
          Real *limln = &(limiterk_(k-1,j,i,0));
          Real *limrn = &(limiterk_(k+1,j,i,0));
          Real *coef3n = &(const_coef3_(k,j,i,0));

          for(int n=0; n<prad->n_fre_ang; ++n){
         // first left flux
            Real lflux = 0.0;
            Real rflux = 0.0;
            coef3n[n] = 0.0;
            if(vel_ln[n] > 0.0){
              Real il = irln[n] + dxp1 * limln[n];
              lflux = vel_ln[n] * il;
            }
            else if(vel_ln[n] < 0.0){
              lflux = vel_ln[n] * (-dxm * limn[n]); 
              coef3n[n] = (-areal * vel_ln[n]); 
            }

            if(vel_rn[n] > 0.0){
              coef3n[n] += (arear * vel_rn[n]);
              rflux = vel_rn[n] * (dxp * limn[n]);
            }else if(vel_rn[n] < 0.0){
              Real ir = irrn[n] - dxm1 * limrn[n];
              rflux = vel_rn[n] * ir;
            }

            flxn[n] = (arear * rflux - areal * lflux);
          }// end n 
#pragma omp simd aligned(divn,flxn:ALI_LEN)
          for(int n=0; n<prad->n_fre_ang; ++n){
            divn[n] -= flxn[n];
          }
        }
      }// end j  

    }// end k
  }// end 3D



  for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      pmb->pcoord->CellVolume(k,j,is,ie,vol);

      for(int i=is; i<=ie; ++i){
        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *divn = &(divflx_(k,j,i,0));
        Real dtvol = wght/vol(i);
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] *= dtvol;
          coef1n[n] *= dtvol;
        }
        if(pmb->block_size.nx2 > 1){
          Real *coef2n = &(const_coef2_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef2n[n] *= dtvol;
          }         
        }
        if(pmb->block_size.nx3 > 1){
          Real *coef3n = &(const_coef3_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef3n[n] *= dtvol;
          }         
        }// end nx3
      }// end i
    }// end j
  }// end i
  
}// end func
