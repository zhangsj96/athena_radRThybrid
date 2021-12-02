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


// interpolate co-moving frame specific intensity over frequency grid
void RadIntegrator::MapIrcmFrequency(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_cm, 
                                     AthenaArray<Real> &ir_shift)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));

  // first copy the data
  for(int n=0; n<pmy_rad->n_fre_ang; ++n)
    ir_shift(n) = ir_cm(n);


  if(nfreq == 2){
    // the frequency grid is 0, nu, infty in this case

    Real *ir_l = &(ir_cm(0));
    Real *ir_r = &(ir_cm(nang));
    Real *ir_shift_l=&(ir_shift(0));
    Real *ir_shift_r=&(ir_shift(nang));
    Real *delta_i = &(delta_i_(1,0));
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        Real di = ir_l[n] * (cm_nu[n] - 1.0) /cm_nu[n];
        ir_shift_l[n] -= di;
        ir_shift_r[n] += di;
        // di_r[n] stores the ratio that gets shifted
        delta_i[n] = di/ir_shift_r[n];
      }else if(cm_nu[n] < 1.0){
        // ir_r is already shifted into the co-moving frame
        // nu_tr = cm_nu nu/ T_r
        Real nu_tr = pmy_rad->EffectiveBlackBody(ir_r[n], cm_nu[n] *pmy_rad->nu_grid(1));
        Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr/cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
        Real di = ir_r[n] * (1.0 - ratio);
        ir_shift_l[n] += di;
        ir_shift_r[n] -= di;
        // di_l[n] now stores the ratio to recover the shift
        delta_i[n] = di/ir_shift_l[n];
      }// end cm_nu

    }// end all angles n
  }// end nfreq == 2
  else if(nfreq > 2){

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

    // first copy the data
  for(int n=0; n<pmy_rad->n_fre_ang; ++n)
    ir_cm(n) = ir_shift(n);


  if(nfreq == 2){
    // the frequency grid is 0, nu, infty in this case

    Real *ir_l = &(ir_cm(0));
    Real *ir_r = &(ir_cm(nang));
    Real *ir_shift_l=&(ir_shift(0));
    Real *ir_shift_r=&(ir_shift(nang));
    Real *delta_i = &(delta_i_(1,0));
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        Real di = delta_i[n] * ir_shift_r[n];
        ir_l[n] += di;
        ir_r[n] -= di;
      }else if(cm_nu[n] < 1.0){
        Real di = delta_i[n] * ir_shift_l[n];
        ir_l[n] -= di;
        ir_r[n] += di;
      }

    }// end all angles n
  }// end nfreq == 2
  else if(nfreq > 2){


  }// end nfreq > 2

  return;

}// end map function

