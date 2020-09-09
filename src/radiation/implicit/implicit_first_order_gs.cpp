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


// the flux is 
// Smax F(U_L)/(Smax-Smin) - Smin F(U_R) /(Smax-Smin)
// + Smax Smin * (U_R - U_L)/(Smax - Smin)
// Gauss-Seidel iteration, use upper triangle from last iteration
// lower triangle is from next iteration

void RadIntegrator::FirstOrderGSFluxDivergence(const Real wght, 
                                       AthenaArray<Real> &ir)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco= pmb->pcoord;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nang = prad->nang; 
  int nfreq = prad->nfreq;
  
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
                 &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
                 &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

  AthenaArray<Real> &x1flux=prad->flux[X1DIR];
  AthenaArray<Real> &x2flux=prad->flux[X2DIR];
  AthenaArray<Real> &x3flux=prad->flux[X3DIR];


  AthenaArray<Real> &area_zeta = zeta_area_, &area_psi = psi_area_, 
                 &ang_vol = ang_vol_, &dflx_ang = dflx_ang_;
  int &nzeta = prad->nzeta, &npsi = prad->npsi;

  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) {
      pco->CenterWidth1(k,j,is-1,ie+1,dxw1_);
      for(int i=is; i<=ie+1; ++i){
        Real tau = 0.0;
        for(int ifr=0; ifr<nfreq; ++ifr){
          Real sigmal = prad->sigma_a(k,j,i-1,ifr) + prad->sigma_s(k,j,i-1,ifr);
          Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
          tau += prad->wfreq(ifr)*(dxw1_(i-1) * sigmal + dxw1_(i) * sigmar);
        }// end ifr
        Real factor1 = 1.0;
        Real factor2 = 1.0;
        GetTaufactor(tau,factor1,factor2);

        Real *s1n = &(sfac1_x_(i,0));
        Real *s2n = &(sfac2_x_(i,0));
        Real *velxn = &(velx_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          if(velxn[n] > 0.0){
            s1n[n] = factor1 * velxn[n];
            s2n[n] = -factor2 * velxn[n];
          }else{
            s1n[n] = -factor2 * velxn[n];
            s2n[n] = factor1 * velxn[n];
          }

        }
      }// end i
      // calculate x1-flux divergence 
      pco->Face1Area(k,j,is,ie+1,x1area);
      for(int i=is; i<=ie; ++i){
        Real areal = x1area(i);
        Real arear = x1area(i+1);
        Real *s1_ln = &(sfac1_x_(i,0));
        Real *s1_rn = &(sfac1_x_(i+1,0)); // the minimum signal speed
        Real *s2_ln = &(sfac2_x_(i,0));  
        Real *s2_rn = &(sfac2_x_(i+1,0)); // the maximum signal speed
        Real *vel_ln = &(velx_(k,j,i,0));
        Real *vel_rn = &(velx_(k,j,i+1,0));        

        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        Real *irln = &(ir(k,j,i-1,0));
        Real *irrn = &(ir(k,j,i+1,0));
        Real *divn = &(divflx_(k,j,i,0));
        Real advl = 0.0;
        Real advr = 0.0;
        if(adv_flag_ > 0){
          advl = adv_vel(0,k,j,i);
          advr = adv_vel(0,k,j,i+1);
        }
        // first, consider left hand side
        if(i==is){
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_ln[n];
            Real smin = s2_ln[n];
            Real vl = vel_ln[n] - advl;
            Real lflux = 0.5 * vl * irln[n] + 0.5 * smax * irln[n];
            lcoefn[n] = 0.0;
            coef1n[n] = -areal * 0.5 * (vl - smax); 
            divn[n] = areal * lflux;          
          }// end n
        }else{
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_ln[n];
            Real smin = s2_ln[n];
            Real vl = vel_ln[n] - advl;

            lcoefn[n] = -areal * 0.5 * (vl + smax);
            coef1n[n] = -areal * 0.5 * (vl - smax);  
            divn[n] = 0.0;          
          }// end n          
        }

        // now consider the right hand side, which is always from last iteration
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real smax = s1_rn[n];
          Real smin = s2_rn[n];
          Real vr = vel_rn[n] - advr;
          Real rflux = 0.5 * vr * irrn[n] - 0.5 * smax * irrn[n];
          coef1n[n] += arear * 0.5 * (vr + smax);
          divn[n] += -(arear * rflux);
        }// end n
        if(adv_flag_ > 0){
          Real *flxr = &(x1flux(k,j,i+1,0));
          Real *flxl = &(x1flux(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            divn[n] += -(arear * flxr[n] - areal * flxl[n]);
          }        
        }// end adv_flag

      }// End i

    }// end j
  }// end k

     // calculate x2-flux
  if (pmb->block_size.nx2 > 1) {
    for(int k=ks; k<=ke; ++k){
      // first, calculate speed
      for(int j=js; j<=je+1; ++j){
        pco->CenterWidth2(k,j-1,is,ie,dxw1_);
        pco->CenterWidth2(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i){
          Real tau = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k,j-1,i,ifr) + prad->sigma_s(k,j-1,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            tau += prad->wfreq(ifr) * (dxw1_(i) * sigmal + dxw2_(i) * sigmar);
          }
          Real factor1 = 1.0;
          Real factor2 = 1.0;
          GetTaufactor(tau,factor1,factor2);

          Real *s1n = &(sfac1_y_(j,i,0));
          Real *s2n = &(sfac2_y_(j,i,0));
          Real *velyn = &(vely_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            if(velyn[n] > 0.0){
              s1n[n] = factor1 * velyn[n];
              s2n[n] = -factor2 * velyn[n];
            }else{
              s1n[n] = -factor2 * velyn[n];
              s2n[n] = factor1 * velyn[n];
            }

          }// end n

        }// end i
      }// end j

      for(int j=js; j<=je; ++j){

        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);

        for(int i=is; i<=ie; ++i){
          Real *vel_ln = &(vely_(k,j,i,0));
          Real *vel_rn = &(vely_(k,j+1,i,0));
          Real *coef2n = &(const_coef2_(k,j,i,0));
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          Real *s1_ln = &(sfac1_y_(j,i,0));
          Real *s1_rn = &(sfac1_y_(j+1,i,0));
          Real *s2_ln = &(sfac2_y_(j,i,0));
          Real *s2_rn = &(sfac2_y_(j+1,i,0));

          Real *irln = &(ir(k,j-1,i,0));
          Real *irrn = &(ir(k,j+1,i,0));
          Real areal = x2area(i);
          Real arear = x2area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));
          Real advl = 0.0;
          Real advr = 0.0;
          if(adv_flag_ > 0){
            advl = adv_vel(1,k,j,i);
            advr = adv_vel(1,k,j+1,i);
          }
          // the left hand side
          if(j==js){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = vel_ln[n] - advl;
              Real lflux = 0.5 * vl * irln[n] + 0.5 * smax * irln[n];
              lcoefn[n] = 0.0;
              coef2n[n] = -areal * 0.5 * (vl - smax); 
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = vel_ln[n] - advl;

              lcoefn[n] = -areal * 0.5 * (vl + smax);
              coef2n[n] = -areal * 0.5 * (vl - smax);  
//              divn[n] += 0.0;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_rn[n];
            Real smin = s2_rn[n];
            Real vr = vel_rn[n] - advr;
            Real rflux = 0.5 * vr * irrn[n] - 0.5 * smax * irrn[n];
            coef2n[n] += arear * 0.5 * (vr + smax);
            divn[n] += -(arear * rflux);  

          }// end n
          if(adv_flag_ > 0){
            Real *flxr = &(x2flux(k,j+1,i,0));
            Real *flxl = &(x2flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx2


  // calculate x3-flux divergence
  if (pmb->block_size.nx3 > 1) {
    for(int k=ks; k<=ke+1; ++k){
      for(int j=js; j<=je; ++j){
        pco->CenterWidth3(k-1,j,is,ie,dxw1_);
        pco->CenterWidth3(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i){
          Real tau = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k-1,j,i,ifr) + prad->sigma_s(k-1,j,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            tau += prad->wfreq(ifr) * (dxw1_(i) * sigmal + dxw2_(i) * sigmar);
            tau *= taufact_;
          }
          Real factor1 = 1.0;
          Real factor2 = 1.0;
          GetTaufactor(tau,factor1,factor2);

          Real *s1n = &(sfac1_z_(k,j,i,0));
          Real *s2n = &(sfac2_z_(k,j,i,0));
          Real *velzn = &(velz_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            if(velzn[n] > 0.0){
              s1n[n] =  factor1 * velzn[n];
              s2n[n] =  -factor2 * velzn[n];
            }else{
              s1n[n] =  -factor2 * velzn[n];
              s2n[n] =  factor1 * velzn[n];
            }

          }// end n
        }// end i
      }// end j
    }// end k
    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for(int i=is; i<=ie; ++i){
          Real *s1_ln = &(sfac1_z_(k,j,i,0));
          Real *s1_rn = &(sfac1_z_(k+1,j,i,0));
          Real *s2_ln = &(sfac2_z_(k,j,i,0));
          Real *s2_rn = &(sfac2_z_(k+1,j,i,0));
          Real *vel_ln = &(velz_(k,j,i,0));
          Real *vel_rn = &(velz_(k+1,j,i,0));
          Real *coef3n = &(const_coef3_(k,j,i,0));
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          Real *irln = &(ir(k-1,j,i,0));
          Real *irrn = &(ir(k+1,j,i,0));
          Real areal = x3area(i);
          Real arear = x3area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));
          Real advl = 0.0;
          Real advr = 0.0;
          if(adv_flag_ > 0){
            advl = adv_vel(2,k,j,i);
            advr = adv_vel(2,k+1,j,i);
          }
          // the left hand side
          if(k==ks){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = vel_ln[n] - advl;
              Real lflux = 0.5 * vl * irln[n] + 0.5 * smax * irln[n];
              lcoefn[n] = 0.0;
              coef3n[n] = -areal * 0.5 * (vl - smax);  
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = vel_ln[n] - advl;
               
              lcoefn[n] = -areal * 0.5 * (vl + smax);
              coef3n[n] = -areal * 0.5 * (vl - smax);  
//              divn[n] += 0.0;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_rn[n];
            Real smin = s2_rn[n];
            Real vr = vel_rn[n] - advr;
            Real rflux = 0.5 * vr * irrn[n] - 0.5 * smax * irrn[n];
            coef3n[n] += arear * 0.5 * (vr + smax);
            divn[n] += -(arear * rflux);  

          }// end n  
          if(adv_flag_ > 0){
            Real *flxr = &(x3flux(k+1,j,i,0));
            Real *flxl = &(x3flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx3

  for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for(int i=is; i<=ie; ++i){
        Real *divn = &(divflx_(k,j,i,0));
        Real dtvol = wght/vol(i);
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] *= dtvol;
        }
        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          coef1n[n] *= dtvol;
          lcoefn[n] *= dtvol;
        }
        if(pmb->block_size.nx2 > 1){
          Real *coef2n = &(const_coef2_(k,j,i,0));
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef2n[n] *= dtvol;
            lcoefn[n] *= dtvol;
          }          
        }// end nx2 > 1
        if(pmb->block_size.nx3 > 1){
          Real *coef3n = &(const_coef3_(k,j,i,0));
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef3n[n] *= dtvol;
            lcoefn[n] *= dtvol;
          }          
        }// end nx3 > 1

      }// end i
    }// end j
  }// end k


}// end function

/*
void RadIntegrator::FirstOrderGSFluxDivergenceSafe(const Real wght, 
                                       AthenaArray<Real> &ir)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco= pmb->pcoord;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nang = prad->nang; 
  int nfreq = prad->nfreq;
  
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
                 &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
                 &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

  AthenaArray<Real> &x1flux=prad->flux[X1DIR];
  AthenaArray<Real> &x2flux=prad->flux[X2DIR];
  AthenaArray<Real> &x3flux=prad->flux[X3DIR];


  AthenaArray<Real> &area_zeta = zeta_area_, &area_psi = psi_area_, 
                 &ang_vol = ang_vol_, &dflx_ang = dflx_ang_;
  int &nzeta = prad->nzeta, &npsi = prad->npsi;

  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) {
      // first, calculate maximum and minimum speeds
      for(int i=is; i<=ie+1; ++i){
        Real tau = 0.0;
        for(int ifr=0; ifr<nfreq; ++ifr){
          Real sigmal = prad->sigma_a(k,j,i-1,ifr) + prad->sigma_s(k,j,i-1,ifr);
          Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
          tau += prad->wfreq(ifr)*((pco->x1f(i) - pco->x1v(i-1)) * sigmal 
                    + (pco->x1v(i) - pco->x1f(i)) * sigmar);
        }// end ifr
        Real factor1 = 1.0;
        Real factor2 = 1.0;
        GetTaufactor(tau,factor1,factor2);

        Real *s1n = &(sfac1_x_(i,0));
        Real *s2n = &(sfac2_x_(i,0));
        Real *velxn = &(velx_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          if(velxn[n] > 0.0){
            s1n[n] = factor1 * velxn[n];
            s2n[n] = -factor2 * velxn[n];
          }else{
            s1n[n] = -factor2 * velxn[n];
            s2n[n] = factor1 * velxn[n];
          }
        }// end n
        Real adv = 0.0;
        if(adv_flag_ > 0)
          adv = adv_vel(0,k,j,i);
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real v = velxn[n] - adv;
          if(v > 0){
            p_velx_(k,j,i,n) = v;
            n_velx_(k,j,i,n) = 0.0;
          }else{
            p_velx_(k,j,i,n) = 0.0;
            n_velx_(k,j,i,n) = v;
          }
        }// end n
      }// end i
      // calculate x1-flux divergence 
      pco->Face1Area(k,j,is,ie+1,x1area);
      for(int i=is; i<=ie; ++i){
        Real areal = x1area(i);
        Real arear = x1area(i+1);
        Real *s1_ln = &(sfac1_x_(i,0));
        Real *s1_rn = &(sfac1_x_(i+1,0)); // the minimum signal speed
        Real *s2_ln = &(sfac2_x_(i,0));  
        Real *s2_rn = &(sfac2_x_(i+1,0)); // the maximum signal speed

        Real *pvel_ln = &(p_velx_(k,j,i,0));
        Real *nvel_ln = &(n_velx_(k,j,i,0));
        Real *pvel_rn = &(p_velx_(k,j,i+1,0));
        Real *nvel_rn = &(n_velx_(k,j,i+1,0));       
      

        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        Real *irln = &(ir(k,j,i-1,0));
        Real *irrn = &(ir(k,j,i+1,0));
        Real *irn = &(ir(k,j,i,0));
        Real *divn = &(divflx_(k,j,i,0));

        // first, consider left hand side
        if(i==is){
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_ln[n];
            Real smin = s2_ln[n];
            Real vl = pvel_ln[n] + nvel_ln[n];
            Real lflux = (smax * vl * irln[n] - smin * pvel_ln[n] * irn[n] 
                      - smax * smin * irln[n])/(smax - smin);
            lcoefn[n] = 0.0;
            coef1n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
            divn[n] = areal * lflux;          
          }// end n
        }else{
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_ln[n];
            Real smin = s2_ln[n];
            Real vl = pvel_ln[n] + nvel_ln[n];
            Real lflux = (-smin * pvel_ln[n] * irn[n])/(smax - smin);
            lcoefn[n] = -areal * (smax * vl - smax * smin)/(smax - smin);
            coef1n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
            divn[n] = areal * lflux;          
          }// end n          
        }

        // now consider the right hand side, which is always from last iteration
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real smax = s1_rn[n];
          Real smin = s2_rn[n];
          Real vr = pvel_rn[n] + nvel_rn[n];
          Real rflux = (smax * nvel_rn[n] * irn[n] -smin * vr * irrn[n] 
                     + smax * smin * irrn[n])/(smax - smin);
          coef1n[n] += arear * (smax * pvel_rn[n] - smax * smin)/(smax - smin);
          divn[n] += -(arear * rflux);
        }// end n
        if(adv_flag_ > 0){
          Real *flxr = &(x1flux(k,j,i+1,0));
          Real *flxl = &(x1flux(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            divn[n] += -(arear * flxr[n] - areal * flxl[n]);
          }        
        }// end adv_flag

      }// End i

    }// end j
  }// end k

     // calculate x2-flux
  if (pmb->block_size.nx2 > 1) {
    for(int k=ks; k<=ke; ++k){
      // first, calculate speed
      for(int j=js; j<=je+1; ++j){
        for(int i=is; i<=ie; ++i){
          Real tau = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k,j-1,i,ifr) + prad->sigma_s(k,j-1,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            tau += prad->wfreq(ifr) * ((pco->x2f(j) - pco->x2v(j-1)) * sigmal 
                    + (pco->x2v(j) - pco->x2f(j)) * sigmar);
          }
          Real factor1 = 1.0;
          Real factor2 = 1.0;
          GetTaufactor(tau,factor1,factor2);

          Real *s1n = &(sfac1_y_(j,i,0));
          Real *s2n = &(sfac2_y_(j,i,0));
          Real *velyn = &(vely_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            if(velyn[n] > 0.0){
              s1n[n] = factor1 * velyn[n];
              s2n[n] = -factor2 * velyn[n];
            }else{
              s1n[n] = -factor2 * velyn[n];
              s2n[n] = factor1 * velyn[n];
            }
          }// end n
          Real adv = 0.0;
          if(adv_flag_ > 0)
            adv = adv_vel(1,k,j,i);
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real v = velyn[n] - adv;
            if(v > 0){
              p_vely_(k,j,i,n) = v;
              n_vely_(k,j,i,n) = 0.0;
            }else{
              p_vely_(k,j,i,n) = 0.0;
              n_vely_(k,j,i,n) = v;
            } 
          }// end n          

        }// end i
      }// end j

      for(int j=js; j<=je; ++j){

        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);

        for(int i=is; i<=ie; ++i){

          Real *coef2n = &(const_coef2_(k,j,i,0));
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          Real *s1_ln = &(sfac1_y_(j,i,0));
          Real *s1_rn = &(sfac1_y_(j+1,i,0));
          Real *s2_ln = &(sfac2_y_(j,i,0));
          Real *s2_rn = &(sfac2_y_(j+1,i,0));

          Real *pvel_ln = &(p_vely_(k,j,i,0));
          Real *nvel_ln = &(n_vely_(k,j,i,0));
          Real *pvel_rn = &(p_vely_(k,j+1,i,0));
          Real *nvel_rn = &(n_vely_(k,j+1,i,0));    

          Real *irln = &(ir(k,j-1,i,0));
          Real *irrn = &(ir(k,j+1,i,0));
          Real *irn = &(ir(k,j,i,0));
          Real areal = x2area(i);
          Real arear = x2area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));

          // the left hand side
          if(j==js){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = pvel_ln[n] + nvel_ln[n];
              Real lflux = (smax * vl * irln[n] - smin * pvel_ln[n] * irn[n]
                      - smax * smin * irln[n])/(smax - smin);
              lcoefn[n] = 0.0;
              coef2n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = pvel_ln[n] + nvel_ln[n];
              Real lflux = (-smin * pvel_ln[n] * irn[n])/(smax - smin);
              lcoefn[n] = -areal * (smax * vl - smax * smin)/(smax - smin);
              coef2n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
              divn[n] += areal * lflux;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_rn[n];
            Real smin = s2_rn[n];
            Real vr = pvel_rn[n] + nvel_rn[n];
            Real rflux = (smax * nvel_ln[n] * irn[n] - smin * vr * irrn[n]
                         + smax * smin * irrn[n])/(smax - smin);
            coef2n[n] += arear * (smax * pvel_ln[n] - smax * smin)/(smax - smin);
            divn[n] += -(arear * rflux);  

          }// end n
          if(adv_flag_ > 0){
            Real *flxr = &(x2flux(k,j+1,i,0));
            Real *flxl = &(x2flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx2


  // calculate x3-flux divergence
  if (pmb->block_size.nx3 > 1) {
    for(int k=ks; k<=ke+1; ++k){
      for(int j=js; j<=je; ++j){
        for(int i=is; i<=ie; ++i){
          Real tau = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k-1,j,i,ifr) + prad->sigma_s(k-1,j,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            tau += prad->wfreq(ifr) * ((pco->x3f(k) - pco->x3v(k-1)) * sigmal 
                    + (pco->x3v(k) - pco->x3f(k)) * sigmar);
            tau *= taufact_;
          }
          Real factor1 = 1.0;
          Real factor2 = 1.0;
          GetTaufactor(tau,factor1,factor2);

          Real *s1n = &(sfac1_z_(k,j,i,0));
          Real *s2n = &(sfac2_z_(k,j,i,0));
          Real *velzn = &(velz_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            if(velzn[n] > 0.0){
              s1n[n] =  factor1 * velzn[n];
              s2n[n] =  -factor2 * velzn[n];
            }else{
              s1n[n] =  -factor2 * velzn[n];
              s2n[n] =  factor1 * velzn[n];
            }

          }// end n
          Real adv = 0.0;
          if(adv_flag_ > 0)
            adv = adv_vel(2,k,j,i);
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real v = velzn[n] - adv;
            if(v > 0){
              p_velz_(k,j,i,n) = v;
              n_velz_(k,j,i,n) = 0.0;
            }else{
              p_velz_(k,j,i,n) = 0.0;
              n_velz_(k,j,i,n) = v;
            } 
          }// end n
        }// end i
      }// end j
    }// end k
    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for(int i=is; i<=ie; ++i){
          Real *s1_ln = &(sfac1_z_(k,j,i,0));
          Real *s1_rn = &(sfac1_z_(k+1,j,i,0));
          Real *s2_ln = &(sfac2_z_(k,j,i,0));
          Real *s2_rn = &(sfac2_z_(k+1,j,i,0));

          Real *pvel_ln = &(p_velz_(k,j,i,0));
          Real *nvel_ln = &(n_velz_(k,j,i,0));
          Real *pvel_rn = &(p_velz_(k+1,j,i,0));
          Real *nvel_rn = &(n_velz_(k+1,j,i,0));   

          Real *coef3n = &(const_coef3_(k,j,i,0));
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          Real *irln = &(ir(k-1,j,i,0));
          Real *irrn = &(ir(k+1,j,i,0));
          Real *irn = &(ir(k,j,i,0));
          Real areal = x3area(i);
          Real arear = x3area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));

          // the left hand side
          if(k==ks){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = pvel_ln[n] + nvel_ln[n];
              Real lflux = (smax * vl * irln[n] - smin * pvel_ln[n] * irn[n]
                      - smax * smin * irln[n])/(smax - smin);
              lcoefn[n] = 0.0;
              coef3n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real smax = s1_ln[n];
              Real smin = s2_ln[n];
              Real vl = pvel_ln[n] + nvel_ln[n];
              Real lflux = (-smin * pvel_ln[n] * irn[n])/(smax - smin);
              lcoefn[n] = -areal * (smax * vl - smax * smin)/(smax - smin);
              coef3n[n] = -areal * (-smin * nvel_ln[n] + smax * smin)/(smax - smin);  
              divn[n] += areal * lflux;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real smax = s1_rn[n];
            Real smin = s2_rn[n];
            Real vr = pvel_ln[n] + nvel_ln[n];
            Real rflux = (smax * nvel_ln[n] * irn[n] -smin * vr * irrn[n] 
                        + smax * smin * irrn[n])/(smax - smin);
            coef3n[n] += arear * (smax * pvel_ln[n] - smax * smin)/(smax - smin);
            divn[n] += -(arear * rflux);  

          }// end n  
          if(adv_flag_ > 0){
            Real *flxr = &(x3flux(k+1,j,i,0));
            Real *flxl = &(x3flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx3

  for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for(int i=is; i<=ie; ++i){
        Real *divn = &(divflx_(k,j,i,0));
        Real dtvol = wght/vol(i);
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] *= dtvol;
        }
        Real *coef1n = &(const_coef1_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          coef1n[n] *= dtvol;
          lcoefn[n] *= dtvol;
        }
        if(pmb->block_size.nx2 > 1){
          Real *coef2n = &(const_coef2_(k,j,i,0));
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef2n[n] *= dtvol;
            lcoefn[n] *= dtvol;
          }          
        }// end nx2 > 1
        if(pmb->block_size.nx3 > 1){
          Real *coef3n = &(const_coef3_(k,j,i,0));
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            coef3n[n] *= dtvol;
            lcoefn[n] *= dtvol;
          }          
        }// end nx3 > 1

      }// end i
    }// end j
  }// end k


}// end function
*/