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
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
          Real g_zeta_coef_l = g_zeta_(2*nzeta-1);
          Real g_zeta_coef_r = g_zeta_(2*nzeta);
          Real zeta_coef0 =  wght * g_zeta_coef_l * prad->reduced_c;
          Real zeta_coef1 = -wght * g_zeta_coef_r * prad->reduced_c;
          // now for each psi, the equation becomes
          // (1+zeta_coef0+zeta_coef1) I + Div F_psi = 0
          // n is for zeta
          ImplicitPsiFlux(k,j,i, 2*nzeta-1, wght, 0.0, 
                           zeta_coef0, ir_ini);
          // now go from 2*nzeta-2 to 0
          for(int n=2*nzeta-2; n>=0; --n){
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
    ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
    imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;

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
      ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l)
                                -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;

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
    ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
    imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;

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
      ang_flx_(k,j,i,ang_num) = -coef1 * ir_ini(k,j,i,ang_psi_r) 
                            - z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;            
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
    ang_flx_(k,j,i,ang_num) = -z_coef1*ir_ini(k,j,i,ang_zeta_r);
    imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;

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
      ang_flx_(k,j,i,ang_num) = -coef1*ir_ini(k,j,i,ang_psi_r)
                                -z_coef1*ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef0 + z_coef;

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

    ang_flx_(k,j,i,ang_num) = -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
    imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;

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

      ang_flx_(k,j,i,ang_num) = -coef0 * ir_ini(k,j,i,ang_psi_l)
                                -z_coef1 * ir_ini(k,j,i,ang_zeta_r);
      imp_ang_coef_(k,j,i,ang_num) = coef1 + z_coef;       
    }// end m
  }// end g_psi(1) > 0

}



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
    for(int j=js; j<=je; ++j)
      for(int i=is; i<=ie; ++i){
        if(prad->npsi == 0){
        // first, related all angle to 2*nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
        // The relation is I_m = coef *I_m+1 + const

        // now, all the other angles
        //  Ir_new - ir_old + coef0 * Ir_l + coef1 ir_ new = 0.0;
        // the first one:
          Real g_coef_l = g_zeta_(0);
          Real g_coef_r = g_zeta_(1);
          Real coef0 = wght * g_coef_l * prad->reduced_c * 
                  area_zeta(0)/ang_vol(0);
          Real coef1 = -wght * g_coef_r * prad->reduced_c * 
                    area_zeta(1)/ang_vol(0);
        // ir_new - ir_old + (coef0 + coef1)i_new=0
          ang_flx_(k,j,i,2*nzeta-1) = -0.5 * coef1 * ir_ini(k,j,i,1);
          imp_ang_coef_(k,j,i,2*nzeta-1) = coef0 + 0.5*coef1;

          for(int n=1; n<2*nzeta-1; ++n){
            g_coef_l = g_zeta_(n);
            g_coef_r = g_zeta_(n+1);
            coef0 = wght * g_coef_l * prad->reduced_c * 
                   area_zeta(n)/ang_vol(n);
            coef1 = -wght * g_coef_r * prad->reduced_c * 
                   area_zeta(n+1)/ang_vol(n);
          // the equation is
          // ir_new - ir_old + coef0 * ir_new + coef1 * ir_new1 = 0
            ang_flx_(k,j,i,n) = -0.5 * coef1 * ir_ini(k,j,i,n+1)
                                -0.5 * coef0 * ir_ini(k,j,i,n-1);
            imp_ang_coef_(k,j,i,n) = 0.5 * coef0 + 0.5 * coef1;
          }// end nzeta
          // the last one
          g_coef_l = g_zeta_(2*nzeta-1);
          g_coef_r = g_zeta_(2*nzeta);
          coef0 = wght * g_coef_l * prad->reduced_c * 
                 area_zeta(2*nzeta-1)/ang_vol(2*nzeta-1);
          coef1 = -wght * g_coef_r * prad->reduced_c * 
                 area_zeta(2*nzeta)/ang_vol(2*nzeta-1);
        // ir_new - ir_old + (coef0 + coef1)i_new=0
          ang_flx_(k,j,i,2*nzeta-1) = -0.5 * coef0 * ir_ini(k,j,i,2*nzeta-2);
          imp_ang_coef_(k,j,i,2*nzeta-1) = 0.5 * coef0 + 0.5 * coef1;

        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0

          for(int n=0; n<2*nzeta; ++n){
            Real g_zeta_coef_l = g_zeta_(n);
            Real g_zeta_coef_r = g_zeta_(n+1);
            Real zeta_coef0 =  wght * g_zeta_coef_l * prad->reduced_c;
            Real zeta_coef1 = -wght * g_zeta_coef_r * prad->reduced_c;
  
            ImplicitPsiFluxCenter(k,j,i, n, wght, zeta_coef1, 
                              zeta_coef0, ir_ini);
          }// end n

        }// end npsi > 0
      }// end k,j,i

}// end calculate_flux




void RadIntegrator::ImplicitPsiFluxCenter(int k, int j, int i, int n_zeta, Real wght, Real zeta_coef1, 
            Real zeta_coef, AthenaArray<Real> &ir_ini)
{

  Radiation *prad=pmy_rad;
  Coordinates *pco=prad->pmy_block->pcoord;
  AthenaArray<Real> &area_psi = psi_area_, &ang_vol = ang_vol_, &zeta_area = zeta_area_;
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
  // m=0
  int ang_num = n_zeta*2*npsi;
  int ang_psi_r = n_zeta*2*npsi+1;
  int ang_psi_l = n_zeta*2*npsi;
  int ang_zeta_l = nzeta_l*2*npsi;
  int ang_zeta_r = nzeta_r*2*npsi;

  Real g_psi_coef_l = 0.0;
  Real g_psi_coef_r = g_psi_(1);
  Real coef0 = 0.0;
  Real coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
               area_psi(n_zeta,1)/ang_vol(ang_num);
  Real z_coef1 = zeta_coef1 * zeta_area(0,n_zeta+1)/ang_vol(ang_num);
  Real z_coef = zeta_coef * zeta_area(0,n_zeta)/ang_vol(ang_num);

  ang_flx_(k,j,i,ang_num) = -0.5 * z_coef1 * ir_ini(k,j,i,ang_zeta_r)
                            -0.5 * z_coef * ir_ini(k,j,i,ang_zeta_l)
                            -0.5 * coef1 * ir_ini(k,j,i,ang_psi_r);
  imp_ang_coef_(k,j,i,ang_num) = 0.5 * coef1 + 0.5 * z_coef + 0.5*z_coef1;  

  for(int m=1; m<2*npsi-1; ++m){
    ang_num = n_zeta*2*npsi+m;
    ang_psi_r = n_zeta*2*npsi+m+1;
    ang_psi_l = n_zeta*2*npsi+m-1;
    ang_zeta_r = nzeta_r*2*npsi+m;
    ang_zeta_l = nzeta_l*2*npsi+m;
    g_psi_coef_l = g_psi_(m);
    g_psi_coef_r = g_psi_(m+1);
    coef0 = wght * g_psi_coef_l * prad->reduced_c * 
            area_psi(n_zeta,m)/ang_vol(ang_num);
    coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
            area_psi(n_zeta,m+1)/ang_vol(ang_num);
    z_coef1 = zeta_coef1 * zeta_area(m,n_zeta+1)/ang_vol(ang_num);
    z_coef = zeta_coef * zeta_area(m,n_zeta)/ang_vol(ang_num);
    ang_flx_(k,j,i,ang_num) = -0.5 * coef0 * ir_ini(k,j,i,ang_psi_l)
                              -0.5 * coef1 * ir_ini(k,j,i,ang_psi_r)
                              -0.5 * z_coef1 * ir_ini(k,j,i,ang_zeta_r)
                              -0.5 * z_coef * ir_ini(k,j,i,ang_zeta_l);
    imp_ang_coef_(k,j,i,ang_num) = 0.5 * (coef0 + coef1 + z_coef + z_coef1);

  }

  // the last one

  ang_num = n_zeta*2*npsi+2*npsi-1;
  ang_psi_r = n_zeta*2*npsi+2*npsi-1;
  ang_psi_l = n_zeta*2*npsi+2*npsi-2;
  ang_zeta_r = nzeta_r*2*npsi+2*npsi-1;
  ang_zeta_l = nzeta_l*2*npsi+2*npsi-1;
  g_psi_coef_l = g_psi_(2*npsi-1);
  g_psi_coef_r = g_psi_(2*npsi);
  coef0 = wght * g_psi_coef_l * prad->reduced_c * 
          area_psi(n_zeta,2*npsi-1)/ang_vol(ang_num);
  coef1 = -wght * g_psi_coef_r * prad->reduced_c * 
          area_psi(n_zeta,2*npsi)/ang_vol(ang_num);
  z_coef1 = zeta_coef1 * zeta_area(2*npsi-1,n_zeta+1)/ang_vol(ang_num);
  z_coef = zeta_coef * zeta_area(2*npsi-1,n_zeta)/ang_vol(ang_num);
  ang_flx_(k,j,i,ang_num) = -0.5 * coef0 * ir_ini(k,j,i,ang_psi_l)
                            -0.5 * z_coef1 * ir_ini(k,j,i,ang_zeta_r)
                            -0.5 * z_coef * ir_ini(k,j,i,ang_zeta_l);
  imp_ang_coef_(k,j,i,ang_num) = 0.5 * (coef0  + z_coef + z_coef1);

}
