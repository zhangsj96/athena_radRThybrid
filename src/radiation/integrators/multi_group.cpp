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

//The function calculate the amount of shift for each I we need
// The default frequency grid is nu_grid with nfreq frequency bins
// This covers from 0 to infty
// the original frequency grid is nu_grid(0,...,nfreq-2)
// the shifted frequency grid is cm_nu(0,...,nfreq-2)
// for each angle, the shift is the same \gamma (1-nv/c)
// The procedure is: Interpolate the intensity to the frequency grid nu_grid from cm_nu
// solve the source terms and update co-moving frame intensity
// interpolate the intensity back to the frequency grid cm_nu from nu_grid for each angle
// Then transform the intensity back to the lab-frame

void RadIntegrator::FrequencyShiftCoef(AthenaArray<Real> &tran_coef, 
         AthenaArray<Real> &nu_flx_l, AthenaArray<Real> &nu_flx_r)
{

  // the current co-moving frequency is nu_grid * prad->mu_cm_factor
  // now we need to map to the original frequency grid nu_grid
  // no need to do this if nfreq == 1
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  // nfreq == 2 case needs to be done specially
  if(nfreq > 2){
    // this is loop over frequency 0 to nfre-2
    // the last frequency bin is special
    for(int n=0; n<nang; ++n){
      nu_flx_l(0,n) = 0.0;
      nu_flx_r(0,n) = 0.0;
    }
    for(int ifr=1; ifr < nfreq-1; ++ifr){
      Real *nuflxl = &(nu_flx_l(ifr,0));
      Real *nuflxr = &(nu_flx_r(ifr,0));
      Real *cm_nu = &(tran_coef(0));
      Real delta_nu = pmy_rad->nu_grid(ifr)-pmy_rad->nu_grid(ifr-1);
      Real delta_nu2= pmy_rad->nu_grid(ifr+1)-pmy_rad->nu_grid(ifr);
      for(int n=0; n<nang; ++n){
        if(cm_nu[n] > 1.0){
          nuflxl[n]= (cm_nu[n] - 1.0) * pmy_rad->nu_grid(ifr)/(cm_nu[n] * delta_nu);
          nuflxr[n] = 0.0;
        }else if(cm_nu[n] < 1.0){
          nuflxl[n]= 0.0;
          nuflxr[n]= (cm_nu[n] - 1.0) * pmy_rad->nu_grid(ifr)/(cm_nu[n] * delta_nu2);
        }else{
          nuflxl[n] = 0.0;
          nuflxr[n] = 0.0;
        }
      }// end loop over all angles
    }// end loop over frequency

  }// end if nfreq > 2

  return;
}

void RadIntegrator::FrequencyInvShiftCoef(AthenaArray<Real> &tran_coef, 
         AthenaArray<Real> &nu_flx_l, AthenaArray<Real> &nu_flx_r)
{

  // the current co-moving frequency is nu_grid * prad->mu_cm_factor
  // now we need to map to the original frequency grid nu_grid
  // no need to do this if nfreq == 1
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  // nfreq == 2 case needs to be done specially
  if(nfreq > 2){
    // swap nu_flx_l and nu_flx_r
    // then add a minus sign
    nu_flx_l.SwapAthenaArray(nu_flx_r);
    for(int ifr=1; ifr < nfreq-1; ++ifr){
      Real *nuflxl = &(nu_flx_l(ifr,0));
      Real *nuflxr = &(nu_flx_r(ifr,0));
      for(int n=0; n<nang; ++n){
        nuflxl[n] *= -1.0;
        nuflxr[n] *= -1.0;
      }
    }
  }// end nfreq > 2

  return;
}


// interpolate co-moving frame specific intensity over frequency grid
void RadIntegrator::MapIrcmFrequency(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_cm, 
                                     AthenaArray<Real> &ir_shift)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));


  if(nfreq == 2){
    // the frequency grid is 0, nu, infty in this case

    Real *ir_l = &(ir_cm(0));
    Real *ir_r = &(ir_cm(nang));
    Real *ir_shift_l=&(ir_shift(0));
    Real *ir_shift_r=&(ir_shift(nang));
    Real delta_i = 0.0;
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        delta_i = ir_l[n] * (cm_nu[n] - 1.0) /cm_nu[n];
        ir_shift_l[n] = ir_l[n] - delta_i;
        ir_shift_r[n] = ir_r[n] + delta_i;
      }else{
        // ir_r is already shifted into the co-moving frame
        // nu_tr = cm_nu nu/ T_r
        Real nu_tr = pmy_rad->EffectiveBlackBody(ir_r[n], cm_nu[n] *pmy_rad->nu_grid(1));
        Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr/cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
        delta_i = ir_r[n] * (1.0 - ratio);
        ir_shift_l[n] = ir_l[n] + delta_i;
        ir_shift_r[n] = ir_r[n] - delta_i;
      }


    }// end all angles n
  }// end nfreq == 2
  else if(nfreq > 2){

    Real *fre_flx_l = &(fre_flx_l_(0));
    Real *fre_flx_r = &(fre_flx_r_(0));   

    // first, initialize fre_flx_l = 0
    for(int n=0; n<nang; ++n)
      fre_flx_l[n] = 0.0; 

    for(int ifr=0; ifr<nfreq-2; ++ifr){
      Real *nuflxl = &(nu_flx_l_(ifr+1,0));
      Real *nuflxr = &(nu_flx_r_(ifr+1,0));
      Real *ir_l = &(ir_cm(ifr*nang));
      Real *ir_r = &(ir_cm((ifr+1)*nang));
      Real *ir_s=&(ir_shift(ifr*nang));
      // calculate flux at right hand side
      for(int n=0; n<nang; ++n)
        fre_flx_r[n] = nuflxl[n] * ir_l[n] + nuflxr[n] * ir_r[n];
      // now apply flux divergence
      for(int n=0; n<nang; ++n)
        ir_s[n] = ir_l[n] - (fre_flx_r[n] - fre_flx_l[n]);

      //save nu_flx_r to nu_flx_l
      fre_flx_l_.SwapAthenaArray(fre_flx_r_);

    }// end ifr from 0 to nfreq-2
    // now check the interface between last two frequency bins
    Real *ir_l = &(ir_cm((nfreq-2)*nang));
    Real *ir_r = &(ir_cm((nfreq-1)*nang));

    for(int n=0; n<nang; ++n){
      if(cm_nu[n] < 1.0){
        Real nu_tr = pmy_rad->EffectiveBlackBody(ir_r[n], pmy_rad->nu_grid(nfreq-1)*cm_nu[n]);
        Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr/cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
        fre_flx_r[n] = ir_r[n] * (ratio - 1.0);
      }else{
        Real delta_nu = pmy_rad->nu_grid(nfreq-1) - pmy_rad->nu_grid(nfreq-2);
        Real ratio = (cm_nu[n] - 1.0) * pmy_rad->nu_grid(nfreq-1)/(cm_nu[n] * delta_nu);
        fre_flx_r[n] = ir_l[n] * (1.0 - ratio);
      }
    }
    for(int n=0; n<nang; ++n){
      ir_shift((nfreq-2)*nang+n) = ir_l[n] - (fre_flx_r[n] - fre_flx_l[n]);
      ir_shift((nfreq-1)*nang+n) = ir_r[n] + fre_flx_r[n];
    }

  }// end nfreq > 2

  return;

}// end map function



// interpolate co-moving frame specific intensity over frequency grid
// from the default frequency grid back to the shifted frequency grid
void RadIntegrator::InverseMapFrequency(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_shift, 
                                     AthenaArray<Real> &ir_cm)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));


  if(nfreq == 2){
    // the frequency grid is 0, nu, infty in this case

    Real *ir_l = &(ir_cm(0));
    Real *ir_r = &(ir_cm(nang));
    Real *ir_shift_l=&(ir_shift(0));
    Real *ir_shift_r=&(ir_shift(nang));
    Real delta_i = 0.0;
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] < 1.0){
        delta_i = ir_shift_l[n] * (1.0 - cm_nu[n]);

        ir_l[n] = ir_shift_l[n] - delta_i;
        ir_r[n] = ir_shift_r[n] + delta_i;

      }else{
        // nu_tr = cm_nu nu/ T_r
        Real nu_tr = pmy_rad->EffectiveBlackBody(ir_shift_r[n], pmy_rad->nu_grid(1));
        Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr*cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
        delta_i = ir_shift_r[n] * (1.0 - ratio);

        ir_l[n] = ir_shift_l[n] + delta_i;
        ir_r[n] = ir_shift_r[n] - delta_i;
      }


    }// end all angles n
  }// end nfreq == 2
  else if(nfreq > 2){

    Real *fre_flx_l = &(fre_flx_l_(0));
    Real *fre_flx_r = &(fre_flx_r_(0));   

    // first, initialize fre_flx_l = 0
    for(int n=0; n<nang; ++n)
      fre_flx_l[n] = 0.0; 

    for(int ifr=0; ifr<nfreq-2; ++ifr){
      Real *nuflxl = &(nu_flx_l_(ifr+1,0));
      Real *nuflxr = &(nu_flx_r_(ifr+1,0));
      Real *ir_shift_l = &(ir_shift(ifr*nang));
      Real *ir_shift_r = &(ir_shift((ifr+1)*nang));
      Real *ir_l = &(ir_cm(ifr*nang));
      // calculate flux at right hand side
      for(int n=0; n<nang; ++n)
        fre_flx_r[n] = nuflxl[n] * ir_shift_l[n] + nuflxr[n] * ir_shift_r[n];
      // now apply flux divergence
      for(int n=0; n<nang; ++n)
        ir_l[n] = ir_shift_l[n] - (fre_flx_r[n] - fre_flx_l[n]);

      //save nu_flx_r to nu_flx_l
      fre_flx_l_.SwapAthenaArray(fre_flx_r_);

    }// end ifr from 0 to nfreq-2
    // now check the interface between last two frequency bins
    Real *ir_shift_l = &(ir_shift((nfreq-2)*nang));
    Real *ir_shift_r = &(ir_shift((nfreq-1)*nang));

    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        Real nu_tr = pmy_rad->EffectiveBlackBody(ir_shift_r[n], pmy_rad->nu_grid(nfreq-1));
        Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr*cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
        fre_flx_r[n] = ir_shift_r[n] * (ratio - 1.0);
      }else{
        Real delta_nu = pmy_rad->nu_grid(nfreq-1) - pmy_rad->nu_grid(nfreq-2);
        Real ratio = (1.0 - cm_nu[n]) * pmy_rad->nu_grid(nfreq-1)/(delta_nu);
        fre_flx_r[n] = ir_shift_l[n] * ratio;
      }
    }
    for(int n=0; n<nang; ++n){
      ir_cm((nfreq-2)*nang+n) = ir_shift_l[n] - (fre_flx_r[n] - fre_flx_l[n]);
      ir_cm((nfreq-1)*nang+n) = ir_shift_r[n] + fre_flx_r[n];
    }

  }// end nfreq > 2

  return;

}// end map function

