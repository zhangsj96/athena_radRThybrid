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

#include <sstream>    // stringstream
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



// calculate advective flux for the implicit scheme
// only work for 1D spherical polar 
// The equation to solve is
// \partial /\partial t - c/r \partial(\sin^2 zeta I)/\sin \zeta\partial \zeta
// the output is stored in angflx



void RadIntegrator::ImplicitAngularFluxes(const Real wght, 
                                     AthenaArray<Real> &ir_ini)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco=pmb->pcoord;
  std::stringstream msg;

  int &nzeta = prad->nzeta;
  int &npsi = prad->npsi;

  AthenaArray<Real> &area_zeta = zeta_area_, &ang_vol = ang_vol_;

  
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for(int k=ks; k<=ke; ++k)
    for(int j=js; j<=je; ++j)
      for(int i=is; i<=ie; ++i){
        if(prad->npsi == 0){
        // first, related all angle to 2*nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
        // The relation is I_m = coef *I_m+1 + const

        // now, all the other angles
        //  Ir_new - ir_old + coef0 * Ir_l + coef1 ir_ new = 0.0;
          for(int n=0; n<2*nzeta-1; ++n){
            Real g_coef_l = g_zeta_(n);
            Real g_coef_r = g_zeta_(n+1);
            Real coef0 = wght * g_coef_l * prad->reduced_c * 
                       area_zeta(n)/ang_vol(n);
            Real coef1 = -wght * g_coef_r * prad->reduced_c * 
                       area_zeta(n+1)/ang_vol(n);
          // the equation is
          // ir_new - ir_old + coef0 * ir_new + coef1 * ir_new1 = 0
            ang_flx_(k,j,i,n) = -coef1 * ir_ini(k,j,i,n+1);
            imp_ang_coef_(k,j,i,n) = coef0;
          }// end nzeta
          // the last one
          Real g_coef_l = g_zeta_(2*nzeta-1);
          Real g_coef_r = g_zeta_(2*nzeta);
          Real coef0 = wght * g_coef_l * prad->reduced_c * 
                    area_zeta(2*nzeta-1)/ang_vol(2*nzeta-1);
          Real coef1 = -wght * g_coef_r * prad->reduced_c * 
                    area_zeta(2*nzeta)/ang_vol(2*nzeta-1);
        // ir_new - ir_old + (coef0 + coef1)i_new=0
          ang_flx_(k,j,i,2*nzeta-1) = 0.0;
          imp_ang_coef_(k,j,i,2*nzeta-1) = coef0 + coef1;

        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0
          // first, starting from the zeta angle 2*nzeta-1
          // zeta area is only 0 at nzeta=0, not zero at nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);

          // now go from 2*nzeta-2 to 0
          for(int n=0; n<2*nzeta; ++n){
            Real g_zeta_coef_l = g_zeta_(n);
            Real g_zeta_coef_r = g_zeta_(n+1);
            Real zeta_coef0 =  wght * g_zeta_coef_l * prad->reduced_c;
            Real zeta_coef1 = -wght * g_zeta_coef_r * prad->reduced_c;
  
            ImplicitPsiFlux(k,j,i, n, wght, zeta_coef1, 
                              zeta_coef0, ir_ini);

          }// end n
        }// end npsi > 0
      }// end k,j,i

}// end calculate_flux




void RadIntegrator::ImplicitPsiFlux(int k, int j, int i, int n_zeta, Real wght, Real zeta_coef1, 
            Real zeta_coef, AthenaArray<Real> &ir_ini)
{

  Radiation *prad=pmy_rad;
  Coordinates *pco=prad->pmy_block->pcoord;
  AthenaArray<Real> &area_psi = psi_area_, &ang_vol = ang_vol_, &zeta_area = zeta_area_;
  int &npsi = prad->npsi;

  int nzeta_r = n_zeta + 1;
  if(n_zeta == 2*prad->nzeta - 1)
    nzeta_r = n_zeta;

  // the equation to solve
  //(1+zeta_coef0+zeta_coef1) I + Div F_psi = 0
  pco->GetGeometryPsi(prad,k,j,i,n_zeta,g_psi_);
  // g_psi_ =sin zeta * cot \theta sin\psi/r
  // g_psi_(0) is always 0
  if(g_psi_(1) < 0){
  // starting from m=0, 
    int ang_num = n_zeta*2*npsi;
    int ang_psi_r = n_zeta*2+npsi+1;
    int ang_zeta_r = nzeta_r*2*npsi;
    Real g_psi_coef_l = 0.0;
    Real g_psi_coef_r = g_psi_(1);
    Real coef0 = 0.0;
    Real coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
                       area_psi(n_zeta,1)/ang_vol(ang_num);
    Real z_coef1 = zeta_coef1 * zeta_area(0,n_zeta+1)/ang_vol(ang_num);
    Real z_coef = zeta_coef * zeta_area(0,n_zeta)/ang_vol(ang_num);
    if(n_zeta == 2*prad->nzeta-1){
      ang_flx_(k,j,i,ang_num) = 0.0;
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef + z_coef1;
    }else{
      ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;      
    }

    for(int m=1; m<npsi; ++m){ // all take the left state
      ang_num = n_zeta*2*npsi+m;
      int ang_psi_l = n_zeta*2*npsi+m-1;
      ang_zeta_r = nzeta_r*2*npsi+m;

      g_psi_coef_l = g_psi_(m);
      g_psi_coef_r = g_psi_(m+1);
      coef0 = wght * g_psi_coef_l * prad->reduced_c * 
              area_psi(n_zeta,m)/ang_vol(ang_num);
      coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
              area_psi(n_zeta,m+1)/ang_vol(ang_num);
      z_coef1 = zeta_coef1 * zeta_area(m,n_zeta+1)/ang_vol(ang_num);
      z_coef = zeta_coef * zeta_area(m,n_zeta)/ang_vol(ang_num);
      if(n_zeta == 2*prad->nzeta-1){
        ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l);
        imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef + z_coef1;
      }else{
        ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l)
                                -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
        imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;
      }

    }
            // from npsi to 2*npsi, we take the right state`.
            // starting from 2*npsi
    ang_num = n_zeta*2*npsi + 2*npsi-1;
    ang_zeta_r = nzeta_r*2*npsi+2*npsi-1;

    g_psi_coef_l = g_psi_(2*npsi-1);
    coef0 = wght * g_psi_coef_l * prad->reduced_c * 
            area_psi(n_zeta,2*npsi-1)/ang_vol(ang_num);
    z_coef1 = zeta_coef1 * zeta_area(2*npsi-1,n_zeta+1)/ang_vol(ang_num);
    z_coef = zeta_coef * zeta_area(2*npsi-1,n_zeta)/ang_vol(ang_num);
    if(n_zeta < 2*prad->nzeta-1){
      ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;
    }else{
      ang_flx_(k,j,i,ang_num) = 0.0;
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef + z_coef1;      
    }

    for(int m=2*npsi-2; m>=npsi; --m){
      ang_num = n_zeta*2*npsi+m;
      int ang_psi_r = n_zeta*2*npsi+m+1;
      ang_zeta_r = nzeta_r*2*npsi+m;
      g_psi_coef_l = g_psi_(m);
      g_psi_coef_r = g_psi_(m+1);
      coef0 = wght * g_psi_coef_l * prad->reduced_c * 
              area_psi(n_zeta,m)/ang_vol(ang_num);
      coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
              area_psi(n_zeta,m+1)/ang_vol(ang_num);
      z_coef1 = zeta_coef1 * zeta_area(m,n_zeta+1)/ang_vol(ang_num);
      z_coef = zeta_coef*zeta_area(m,n_zeta)/ang_vol(ang_num);
      if(n_zeta < 2*prad->nzeta - 1){
        ang_flx_(k,j,i,ang_num) = -coef1 * ir_ini(k,j,i,ang_psi_r) 
                            - z_coef1 * ir_ini(k,j,i,ang_zeta_r);
        imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef; 
      }else{
        ang_flx_(k,j,i,ang_num) = -coef1 * ir_ini(k,j,i,ang_psi_r);
        imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef + z_coef1; 
      }           
    }
  }else{// end g_psi(1) < 0
           // go from npsi-1 to 0s
    int ang_num = n_zeta*2*npsi+npsi-1;
    int ang_zeta_r = nzeta_r*2*npsi+npsi-1;
    Real g_psi_coef_l = g_psi_(npsi-1);
    Real g_psi_coef_r = 0.0;
    Real coef0 = wght * g_psi_coef_l * prad->reduced_c * 
                 area_psi(n_zeta,npsi-1)/ang_vol(ang_num);
    Real coef1 = 0.0;
    Real z_coef1 = zeta_coef1*zeta_area(npsi-1,n_zeta+1)/ang_vol(ang_num);
    Real z_coef = zeta_coef*zeta_area(npsi-1,n_zeta)/ang_vol(ang_num);
    if(n_zeta < 2*prad->nzeta -1){
      ang_flx_(k,j,i,ang_num) = -z_coef1*ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;
    }else{
      ang_flx_(k,j,i,ang_num) = 0.0;
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef + z_coef1;      
    }

    for(int m=npsi-2; m>=0; --m){ // all take the right state
      ang_num = n_zeta*2*npsi+m;
      ang_zeta_r = nzeta_r*2*npsi+m;
      int ang_psi_r = n_zeta*2*npsi+m+1;
      g_psi_coef_l = g_psi_(m);
      g_psi_coef_r = g_psi_(m+1);
      coef0 = wght * g_psi_coef_l * prad->reduced_c * 
              area_psi(n_zeta,m)/ang_vol(ang_num);
      coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
              area_psi(n_zeta,m+1)/ang_vol(ang_num);
      z_coef1 = zeta_coef1*zeta_area(m,n_zeta+1)/ang_vol(ang_num);
      z_coef = zeta_coef*zeta_area(m,n_zeta)/ang_vol(ang_num);
      if(n_zeta < 2*prad->nzeta-1){
        ang_flx_(k,j,i,ang_num) = -coef1*ir_ini(k,j,i,ang_psi_r)
                                -z_coef1*ir_ini(k,j,i,ang_zeta_r);
        imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;
      }else{
        ang_flx_(k,j,i,ang_num) = -coef1*ir_ini(k,j,i,ang_psi_r);
        imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef + z_coef1;
      }

    }
            // from npsi to 2*npsi, we take the left state`.
            // starting from npsi
    ang_num = n_zeta*2*npsi + npsi;
    ang_zeta_r=nzeta_r*2*npsi+npsi;
    g_psi_coef_r = g_psi_(npsi+1);

    coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
            area_psi(n_zeta,npsi+1)/ang_vol(ang_num);
    z_coef1 = zeta_coef1*zeta_area(npsi,n_zeta+1)/ang_vol(ang_num);
    z_coef = zeta_coef*zeta_area(npsi,n_zeta)/ang_vol(ang_num);
    if(n_zeta < 2*prad->nzeta-1){
      ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;
    }else{
      ang_flx_(k,j,i,ang_num) = 0.0;
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef + z_coef1;      
    }

    for(int m=npsi+1; m<2*npsi; ++m){
      ang_num = n_zeta*2*npsi+m;
      ang_zeta_r=nzeta_r*2*npsi+m;
      int ang_psi_l=n_zeta*2*npsi+m-1;
      g_psi_coef_l = g_psi_(m);
      g_psi_coef_r = g_psi_(m+1);
      coef0 = wght * g_psi_coef_l * prad->reduced_c * 
              area_psi(n_zeta,m)/ang_vol(ang_num);
      coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
              area_psi(n_zeta,m+1)/ang_vol(ang_num);
      z_coef1 = zeta_coef1*zeta_area(m,n_zeta+1)/ang_vol(ang_num);
      z_coef = zeta_coef*zeta_area(m,n_zeta)/ang_vol(ang_num);
      if(n_zeta < 2*prad->nzeta-1){
        ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l)
                                -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
        imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;     
      }else{
        ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l);
        imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef + z_coef1;         
      } 
    }// end m
  }// end g_psi(1) > 0

}

//switch to centered difference in optically thick regime
// the flux is 
// (Smax v_f I_l-Smin v_f I_r) + Smin*Smax(Ir - Il)/(Smax-Smin)

void RadIntegrator::ImplicitAngularFluxesCenter(const Real wght, 
                                     AthenaArray<Real> &ir_ini)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco=pmb->pcoord;
  std::stringstream msg;

  int &nzeta = prad->nzeta;
  int &npsi = prad->npsi;

  AthenaArray<Real> &area_zeta = zeta_area_, &ang_vol = ang_vol_;

  
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for(int k=ks; k<=ke; ++k)
    for(int j=js; j<=je; ++j){
       // get the optical depth per cell
      pco->CenterWidth1(k,j,is,ie,dxw1_);
      if(je > js){
        pco->CenterWidth2(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i)
          dxw1_(i) += dxw2_(i);
      }
      if(ke > ks){
        pco->CenterWidth3(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i)
          dxw1_(i) += dxw2_(i);
      }
      for(int i=is; i<=ie; ++i){
        // get the optical depth per cell
        Real tau_c= 0.0;
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          Real sigma = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
          tau_c += prad->wfreq(ifr) * dxw1_(i) * sigma;
        }
        tau_c *= taufact(k,j,i);
        Real f_l = 1.0;
        Real f_r = 1.0;
        GetTaufactor(tau_c,f_r,1);
        GetTaufactor(tau_c,f_l,-1);   



        if(prad->npsi == 0){
        // first, related all angle to 2*nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
        // The relation is I_m = coef *I_m+1 + const
          // the interface velocity is always negative
          for(int n=0; n<2*nzeta; ++n){
            int nl = n-1;
            int nr = n+1;
            if(n==0)
              nl = 0;
            
            if(nr == 2*nzeta-1)
              nr = n;

            // the left hand side
            Real vl = -g_zeta_(n) * prad->reduced_c * area_zeta(n);
            Real smax = -vl * f_l;
            Real smin = vl * f_r;

            Real ratio = smax - smin;
            if(ratio > TINY_NUMBER)
              ratio = 1.0/ratio;
            Real coef_l = -wght * (vl - smin) * smax * ratio/ang_vol(n);
            Real coef_c = -wght * (smax - vl) * smin * ratio/ang_vol(n);  
            if(n == 0){
              coef_l = 0.0;
              coef_c = -wght * vl/ang_vol(0);
            }         

            // the right hand side
            Real vr = -g_zeta_(n+1) * prad->reduced_c * area_zeta(n+1);
            smax = -vr * f_l;
            smin = vr * f_r;
            ratio = smax - smin;
            if(ratio > TINY_NUMBER)
              ratio = 1.0/ratio;
            Real coef_r = wght * (smax - vr) * smin * ratio/ang_vol(n);
            if(n == 2*nzeta-1){
              coef_r = 0.0;
              coef_c +=  wght * vr/ang_vol(n);
            }else
             coef_c += wght * (vr - smin) * smax * ratio/ang_vol(n);    

          // the equation is
          // ir_new - ir_old + coef0 * ir_new + coef1 * ir_new1 = 0
            ang_flx_(k,j,i,n) = -coef_r * ir_ini(k,j,i,nr)
                                -coef_l * ir_ini(k,j,i,nl);
            imp_ang_coef_(k,j,i,n) = coef_c;
          }// end nzeta

        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0

          for(int n=0; n<2*nzeta; ++n){
            Real g_zeta_coef_l = g_zeta_(n);
            Real g_zeta_coef_r = g_zeta_(n+1);
            Real zeta_coef0 =  g_zeta_coef_l * prad->reduced_c;
            Real zeta_coef1 =  g_zeta_coef_r * prad->reduced_c;
  
            ImplicitPsiFluxCenter(k,j,i, n, wght, zeta_coef1, 
                              zeta_coef0, f_l, f_r, ir_ini);
          }// end n

        }// end npsi > 0
      }
    }// end k,j,i

}// end calculate_flux




void RadIntegrator::ImplicitPsiFluxCenter(int k, int j, int i, int n_zeta, Real wght, Real zeta_coef1, 
            Real zeta_coef, Real f_l, Real f_r, AthenaArray<Real> &ir_ini)
{

  Radiation *prad=pmy_rad;
  Coordinates *pco=prad->pmy_block->pcoord;
  AthenaArray<Real> &area_psi = psi_area_, &ang_vol = ang_vol_, &area_zeta = zeta_area_;
  int &npsi = prad->npsi;
  int nzeta_l = n_zeta-1;
  if(n_zeta == 0)
    nzeta_l = 0;
  int nzeta_r = n_zeta+1;
  if(n_zeta == 2*prad->nzeta-1)
    nzeta_r = n_zeta;

  // the equation to solve
  //(1+zeta_coef0+zeta_coef1) I + Div F_psi = 0
  pco->GetGeometryPsi(prad,k,j,i,n_zeta,g_psi_);


    // in this case, psi velocity positive for m=0 to npsi
    // psi velocity negative for m=npsi to 2*npsi
  for(int m=0; m<2*npsi; ++m){
    int ang_num = n_zeta*2*npsi+m;
    int ang_psi_r = n_zeta*2*npsi+m+1;
    int ang_psi_l = n_zeta*2*npsi+m-1;
    int ang_zeta_l = nzeta_l*2*npsi+m;
    int ang_zeta_r = nzeta_r*2*npsi+m;
    if(m==0)
      ang_psi_l = n_zeta*2*npsi+m;
    if(m==2*npsi-1)
      ang_psi_r = n_zeta*2*npsi+m;

      // the left hand side of zeta
    Real v_zeta_l =  -zeta_coef * area_zeta(m,n_zeta);
    Real smax = -v_zeta_l * f_l;
    Real smin = v_zeta_l * f_r;
    Real ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;
    Real coef_zeta_l = -wght * (v_zeta_l - smin) * smax * ratio/ang_vol(ang_num);
    Real coef_zeta_c = -wght * (smax - v_zeta_l) * smin * ratio/ang_vol(ang_num);  

    if(n_zeta == 0){
      coef_zeta_l = 0.0;
      coef_zeta_c = -wght * v_zeta_l/ang_vol(ang_num);
    }

      // the right hand side of zeta

    Real v_zeta_r = -zeta_coef1 * area_zeta(m,n_zeta+1);
    smax = -v_zeta_r * f_l;
    smin = v_zeta_r * f_r;

    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;
    Real coef_zeta_r = wght * (smax - v_zeta_r) * smin * ratio/ang_vol(ang_num);
    if(n_zeta == 2*prad->nzeta-1){
      coef_zeta_c += wght * v_zeta_r/ang_vol(ang_num);
      coef_zeta_r = 0.0;
    }else{
      coef_zeta_c += wght * (v_zeta_r - smin) * smax * ratio/ang_vol(ang_num);   
    }

      // the left hand side of Psi
    Real v_psi_l = -g_psi_(m) * prad->reduced_c * area_psi(n_zeta,m);
    smax = v_psi_l * f_r;
    smin = -v_psi_l * f_l;
    if(v_psi_l < 0){
      smax = -v_psi_l * f_l;
      smin = v_psi_l * f_r;
    }
    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;    
    Real coef_psi_l = -wght * (v_psi_l - smin) * smax * ratio/ang_vol(ang_num);
    Real coef_psi_c = -wght * (smax - v_psi_l) * smin * ratio/ang_vol(ang_num);   
    if(m == 0){
      coef_psi_l = 0.0;
    }

    //the right hand side of psi
    Real v_psi_r = -g_psi_(m+1) * prad->reduced_c * area_psi(n_zeta,m+1);
    smax = v_psi_r * f_r;
    smin = -v_psi_r * f_l;    
    if(v_psi_r < 0){
      smax = -v_psi_r * f_l;
      smin = v_psi_r * f_r;
    }
    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;    
    Real coef_psi_r = wght * (smax - v_psi_r) * smin * ratio/ang_vol(ang_num);
    if(m == 2*npsi-1){
      coef_psi_r = 0.0;
    }else{
      coef_psi_c += wght * (v_psi_r - smin) * smax * ratio/ang_vol(ang_num);   
    }
    
        

    ang_flx_(k,j,i,ang_num) = -coef_zeta_l * ir_ini(k,j,i,ang_zeta_l) 
                              -coef_zeta_r * ir_ini(k,j,i,ang_zeta_r)
                              -coef_psi_l * ir_ini(k,j,i,ang_psi_l) 
                              -coef_psi_r * ir_ini(k,j,i,ang_psi_r);
    imp_ang_coef_(k,j,i,ang_num) = coef_zeta_c + coef_psi_c;   

  }// end m=0 to 2*npsi

}
