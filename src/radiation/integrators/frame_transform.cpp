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
//! \file absorption.cpp
//  \brief  Add absorption source terms
//======================================================================================

#include <stdexcept>  // runtime_error
#include <sstream>  // msg

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../radiation.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../utils/utils.hpp"

// this class header
#include "./rad_integrators.hpp"

//--------------------------------------------------------------------------------------
//! \fn RadIntegrator::LabToCom(const Real vx, const Real vy, const Real vz,
//                          AthenaArray<Real> &ir, AthenaArray<Real> &ir_cm)
//  \brief Transform specific intensity from lab frame to co-moving frame
// with flow velocity vx, vy, vz



void RadIntegrator::LabToCom(const Real vx, const Real vy, const Real vz,
                          Real *mux, Real *muy, Real *muz,
                          Real *ir_lab, AthenaArray<Real> &ir_cm)
{

  Real& prat = pmy_rad->prat;
  Real invcrat = 1.0/pmy_rad->crat;
  int& nang=pmy_rad->nang;
  int& nfreq=pmy_rad->nfreq;
  
  
  
  // square of Lorentz factor
  Real lorzsq = 1.0/(1.0 - (vx * vx + vy * vy + vz * vz) * invcrat * invcrat);
  


  for(int ifr=0; ifr<nfreq; ++ifr){
#pragma omp simd
    for(int n=0; n<nang; n++){
       Real vnc = vx * mux[n] + vy * muy[n] + vz * muz[n];
      
       vnc = 1.0 - vnc * invcrat;
       Real coef = vnc * vnc * vnc * vnc * lorzsq * lorzsq;
      
       ir_cm(n+ifr*nang) = ir_lab[n+ifr*nang] * coef;
    }
  }

  return;
}


//--------------------------------------------------------------------------------------
//! \fn RadIntegrator::ComToLab(const Real vx, const Real vy, const Real vz,
//                          AthenaArray<Real> &ir, AthenaArray<Real> &ir_cm)
//  \brief Transform specific intensity from co-moving frame to lab frame
// with flow velocity vx, vy, vz



void RadIntegrator::ComToLab(const Real vx, const Real vy, const Real vz,
                          Real *mux, Real *muy, Real *muz,
                          AthenaArray<Real> &ir_cm, Real *ir_lab)
{

  Real& prat = pmy_rad->prat;
  Real invcrat = 1.0/pmy_rad->crat;
  int& nang=pmy_rad->nang;
  int& nfreq=pmy_rad->nfreq;
  
  
  
  // square of Lorentz factor
  Real lorzsq = 1.0/(1.0 - (vx * vx + vy * vy + vz * vz) * invcrat * invcrat);
  


  for(int ifr=0; ifr<nfreq; ++ifr){
#pragma omp simd
    for(int n=0; n<nang; n++){
       Real vnc = vx * mux[n] + vy * muy[n] + vz * muz[n];
      
       vnc = 1.0 - vnc * invcrat;
       Real coef = vnc * vnc * vnc * vnc * lorzsq * lorzsq;
      
       ir_lab[n+ifr*nang] = ir_cm(n+ifr*nang) / coef;
    }
  }

  return;
}

// transform co-moving frame intensity to the lab frame for a given spectrum in 
// co-moving frame
void RadIntegrator::ComToLabMultiGroup(const Real vx, const Real vy, const Real vz,
                          Real *mux, Real *muy, Real *muz,
                          AthenaArray<Real> &ir_cm, Real *ir_lab)
{

  Real& prat = pmy_rad->prat;
  Real invcrat = 1.0/pmy_rad->crat;
  int& nang=pmy_rad->nang;
  int& nfreq=pmy_rad->nfreq;
  
  
  
  // square of Lorentz factor
  Real lorz = sqrt(1.0/(1.0 - (vx * vx + vy * vy + vz * vz) * invcrat * invcrat));
  // first, get the lorentz transformation factor
  Real *cm_nu = &(tran_coef_(0));
  // first, set cm_nu=1
  // as we assume the frequency to be in the co-moving frame to begin with
  for(int n=0; n<nang; ++n)
    cm_nu[n] = 1.0;
  // now call the function to get value at frequency center, face and slope
  GetCmMCIntensity(ir_cm, tran_coef_, ir_cen_, ir_slope_);

  // now calculate the actual transformation factor
  for(int n=0; n<nang; ++n){
     Real vnc = vx * mux[n] + vy * muy[n] + vz * muz[n];
      vnc = 1.0 - vnc * invcrat;
      cm_nu[n] = vnc * lorz;
  }

  // now calculate the shifted frequency


  for(int n=0; n<nang; ++n){
    Real *nu_shift = &(nu_shift_(n,0));
    for(int ifr=0; ifr<nfreq; ++ifr){
      nu_shift[ifr] = cm_nu[n] * pmy_rad->nu_grid(ifr);
    }
  }


  // now map ir_cm to the shifted frequency grid for each angle
    // initialize ir_shift to be 0
  ir_shift_.ZeroClear();

  //--------------------------------------------------------------------

  // map intensity to the desired bin
  for(int ifr=0; ifr<nfreq-1; ++ifr){
   // map shifted intensity to the nu_grid
  // inside each bin, the profile is 
  // slope (nu-nu_cen[ifr]) + ir_cen[ifr] 

    Real &nu_cen_lab = pmy_rad->nu_cen(ifr);

    Real &nu_l = pmy_rad->nu_grid(ifr);
    Real &nu_r = pmy_rad->nu_grid(ifr+1);

    int *bin_start = &(map_bin_start_(ifr,0));
    int *bin_end = &(map_bin_end_(ifr,0));



    for(int n=0; n<nang; ++n){
      Real *nu_shift = &(nu_shift_(n,0));
      int l_bd = ifr;
      int r_bd = ifr;
      if(cm_nu[n] < 1.0){
        while((nu_l > nu_shift[l_bd+1]) && (l_bd < nfreq-1))   l_bd++;
        r_bd = l_bd; // r_bd always > l_bd
        while((nu_r > nu_shift[r_bd+1]) && (r_bd < nfreq-1))   r_bd++;   
      }else if(cm_nu[n] > 1.0){
        while((nu_r < nu_shift[r_bd]) && (r_bd > 0))   r_bd--;
        l_bd = r_bd; // r_bd always > l_bd
        while((nu_l < nu_shift[l_bd]) && (l_bd > 0))   l_bd--; 
      }

      if(r_bd-l_bd+1 > nmax_map_){
        std::stringstream msg;
        msg << "### FATAL ERROR in function [MapIrcmFrequency]"
            << std::endl << "Frequency shift '" << r_bd-l_bd+1 << 
              "' larger than maximum allowed " << nmax_map_;
            ATHENA_ERROR(msg);

      }

      bin_start[n] = l_bd;
      bin_end[n] = r_bd;      

      if(rad_fre_order == 1){
        SplitFrequencyBinConstant(l_bd, r_bd, nu_shift, nu_l, nu_r, 
                                             &(split_ratio_(ifr,n)));
      }else if(rad_fre_order == 2){
        Real dim_slope = ir_slope_(ifr,n);
        if(fabs(ir_cm(ifr*nang+n)) > TINY_NUMBER)
          dim_slope /= ir_cm(ifr*nang+n);
        else
          dim_slope = 0.0;
        SplitFrequencyBinLinear(l_bd, r_bd, nu_shift, nu_l, nu_r, 
                                    dim_slope, &(split_ratio_(ifr,n)));          

      }// end rad_fre_order=2
    }// end n  
  }// end ifr=nfreq-2
  //-----------------------------------------------------
  // now the last frequency bin


  for(int n=0; n<nang; ++n){
    Real *nu_shift = &(nu_shift_(n,0));
    Real &nu_l = pmy_rad->nu_grid(nfreq-1);
    if(cm_nu[n] <= 1.0){
      split_ratio_(nfreq-1,n,0) = 1.0;
      map_bin_start_(nfreq-1,n) = nfreq-1;
      map_bin_end_(nfreq-1,n) = nfreq-1;
    }else if(cm_nu[n] > 1.0){
      int r_bd = nfreq-1;
      int l_bd = nfreq-2;// it will always be <= current bin
      while((nu_l < nu_shift[l_bd]) && (l_bd > 0))   l_bd--; 
      map_bin_start_(nfreq-1,n) = l_bd;
      map_bin_end_(nfreq-1,n) = r_bd;  
      // nu_l/kt
      Real nu_tr = pmy_rad->EffectiveBlackBody(ir_cm((nfreq-1)*nang+n), nu_l);
      // FitBlackBody is integral _0 to nu_tr
      // the integral we need is 1 - ori_norm
      Real ori_norm = pmy_rad->FitBlackBody(nu_tr);
      Real div_ori = 0.0;
      if(1.0 - ori_norm > TINY_NUMBER)
        div_ori = 1.0/(1.0 - ori_norm);

      // the first bin
      // the effective temperature 1/T = nu_tr/nu_l
      Real ratio = pmy_rad->FitBlackBody(nu_tr*nu_shift[l_bd+1]/nu_l);
      
      // the difference is (1 - ori_norm) - (1 - ratio)
      split_ratio_(nfreq-1,n,0) = (ratio - ori_norm) * div_ori;
      Real sum = split_ratio_(nfreq-1,n,0);

      for(int m=l_bd+1; m<r_bd; ++m){
        Real ratio_r = pmy_rad->FitBlackBody(nu_tr*nu_shift[m+1]/nu_l);
        Real ratio_l = pmy_rad->FitBlackBody(nu_tr*nu_shift[m]/nu_l);
        split_ratio_(nfreq-1,n,m-l_bd) = (ratio_r - ratio_l) * div_ori;
        sum += split_ratio_(nfreq-1,n,m-l_bd);
      }

      split_ratio_(nfreq-1,n,r_bd-l_bd) = 1.0 - sum;

    }

  }// end loop for n


  // now map the array based on the ratio
  // we need map_start and map_end in this function
  MapIrcmFrequency(ir_cm, ir_shift_);

  //transform from ir_shift to ir_lab
  for(int ifr=0; ifr<nfreq; ++ifr){
#pragma omp simd
    for(int n=0; n<nang; n++){
       ir_lab[n+ifr*nang] = ir_shift_(ifr,n)/(cm_nu[n]*cm_nu[n]*cm_nu[n]*cm_nu[n]);
    }
  }

  return;
}




//--------------------------------------------------------------------------------------
//! \fn RadIntegrator::ComAngle(const Real vx, const Real vy, const Real vz,
//                  Real mux, Real muy, Real muz, Real mux0, Real muy0, Real muz0)
//  \brief Transform angles from lab frame to co-moving frame


void RadIntegrator::ComAngle(const Real vx, const Real vy, const Real vz,
                  Real mux, Real muy, Real muz, Real *mux0, Real *muy0, Real *muz0)
{

  Real invcrat = 1.0/pmy_rad->crat;
  
  
  // square of Lorentz factor
  Real lorz = 1.0/(1.0 - (vx * vx + vy * vy + vz * vz) * invcrat * invcrat);
  lorz = sqrt(lorz);
  Real vdotn = vx * mux + vy * muy + vz * muz;
  
  Real angcoef = lorz * invcrat * (1.0 - lorz * vdotn * invcrat/(1.0 + lorz));
  Real incoef = 1.0/(lorz*(1.0-vdotn * invcrat));
  
  (*mux0) = (mux - vx * angcoef) * incoef;
  (*muy0) = (muy - vy * angcoef) * incoef;
  (*muz0) = (muz - vz * angcoef) * incoef;
  

  return;
}

