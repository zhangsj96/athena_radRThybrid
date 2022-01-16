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

//Giving the frequency integrated intensity for each group in the co-moving frame,
// get the monochromatic intensity at the face and center of the frequency grid
void RadIntegrator::GetCmMCIntensity(AthenaArray<Real> &ir_cm, AthenaArray<Real> &tran_coef, 
                                     AthenaArray<Real> &ir_cen, AthenaArray<Real> &ir_slope)
{
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));

  if(nfreq > 1){
    // no frequency center for the last bin 
    for(int ifr=0; ifr<nfreq-1; ++ifr){
      Real &delta_nu = pmy_rad->delta_nu(ifr);
      Real *ir_int = &(ir_cm(ifr*nang));
      Real *ir_n_cen = &(ir_cen(ifr,0));
      for(int n=0; n<nang; ++n){
        ir_n_cen[n] = ir_int[n]/(cm_nu[n] * delta_nu);
      }
    }// end ir_n-cen
    // get value at frequency grid face
    for(int n=0; n<nang; ++n)
      ir_face_(0,n) = 0.0;
    for(int ifr=1; ifr<nfreq-1; ++ifr){
      Real d_nu = pmy_rad->nu_cen(ifr) - pmy_rad->nu_cen(ifr-1);
      Real d_nu_face = pmy_rad->nu_grid(ifr) - pmy_rad->nu_cen(ifr-1);
      Real *ir_n_cen = &(ir_cen(ifr,0));
      Real *ir_l_cen = &(ir_cen(ifr-1,0));
      Real *ir_face = &(ir_face_(ifr,0));
      // the cm_nu[n] factor is cancelled 
      for(int n=0; n<nang; ++n){
        ir_face[n] = ir_l_cen[n] + (ir_n_cen[n] - ir_l_cen[n]) * d_nu_face/d_nu;
      }
    }

  //-------------------------------------------------------------
    // now get the slope with a limiter
    if((rad_fre_order == 1) || (nfreq == 2)){
      for(int ifr=0; ifr<nfreq; ++ifr)
        for(int n=0; n<nang; ++n){
          ir_slope(ifr,n) = 0.0;
        }
    }else{
      // the first bin
      // nfreq > 2
      Real delta_nu = pmy_rad->nu_grid(1) - pmy_rad->nu_cen(0);
      for(int n=0; n<nang; ++n){
        ir_slope(0,n) = (ir_face_(1,n) - ir_cen(0,n))/(cm_nu[n]*delta_nu);
      }


      // now frequency bin from 1 to nfreq-2
      for(int ifr=1; ifr<nfreq-2; ++ifr){
        Real &delta_nu = pmy_rad->delta_nu(ifr);
        Real *ir_face = &(ir_face_(ifr,0));
        Real *ir_face_r = &(ir_face_(ifr+1,0));
        Real *slope = &(ir_slope(ifr,0));
        Real *ir_n = &(ir_cen(ifr,0));
        for(int n=0; n<nang; ++n){
          if((ir_face_r[n]-ir_n[n]) * (ir_n[n]-ir_face[n]) > 0.0)
            slope[n] = (ir_face_r[n] - ir_face[n])/(cm_nu[n]*delta_nu);
          else
            slope[n] = 0.0;
        }// end n

      }// end ifr nfreq-3
      // the last bin

      delta_nu = pmy_rad->nu_cen(nfreq-2) - pmy_rad->nu_grid(nfreq-2);
      for(int n=0; n<nang; ++n){
        ir_slope(nfreq-2,n) = (ir_cen(nfreq-2,n) - ir_face_(nfreq-2,n))/(cm_nu[n]*delta_nu);
      }

      // check to make sure it does not cause negative intensity
      for(int ifr=0; ifr<nfreq-1; ++ifr){
        Real *slope = &(ir_slope(ifr,0));
        Real *ir_n = &(ir_cen(ifr,0));
        Real &delta_nu = pmy_rad->delta_nu(ifr);
        for(int n=0; n<nang; ++n){
          if(slope[n] < 0.0){
            Real ir_rbd = ir_n[n] + slope[n] * 0.5 * cm_nu[n] * delta_nu;
            if(ir_rbd < 0.0)
              slope[n] = 0.0;
          }else if(slope[n] > 0){
            Real ir_lbd = ir_n[n] - slope[n] * 0.5 * cm_nu[n] * delta_nu;
            if(ir_lbd < 0.0)
              slope[n] = 0.0;            
          }
        }
      }// end ifr 
 
    }// end nfreq > 2

  }// end nfreq > 1

}

// interpolate co-moving frame specific intensity over frequency grid
// every bin is 
void RadIntegrator::MapIrcmFrequency(AthenaArray<Real> &tran_coef, 
            AthenaArray<Real> &ir_cm, AthenaArray<Real> &ir_shift)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));
  Real *nu_lab = &(pmy_rad->nu_grid(0));

  // initialize ir_shift to be 0
  ir_shift.ZeroClear();

  // check to make sure nfreq > 2
  if(nfreq < 2){

    std::stringstream msg;
    msg << "### FATAL ERROR in function [MapIrcmFrequency]"
        << std::endl << "nfreq '" << nfreq << 
          "' is smaller than 2! ";
    ATHENA_ERROR(msg);
  }

  // map intensity to the desired bin
  for(int ifr=0; ifr<nfreq-1; ++ifr){
   // map shifted intensity to the nu_grid
  // inside each bin, the profile is 
  // slope (nu-nu_cen[ifr]) + ir_cen[ifr] 
    Real &nu_r_lab = pmy_rad->nu_grid(ifr+1);
    Real &nu_l_lab = pmy_rad->nu_grid(ifr);
    Real &nu_cen_lab = pmy_rad->nu_cen(ifr);

    int *map_start = &(map_bin_start_(ifr,0));
    int *map_end = &(map_bin_end_(ifr,0));
    for(int n=0; n<nang; ++n){
      Real nu_l = nu_l_lab * cm_nu[n];
      Real nu_r = nu_r_lab * cm_nu[n];

      if(cm_nu[n] > 1.0){

        // find the overlap bin in nu_grid
        int l_bd = ifr;// it will always be >= current bin
        while((nu_l > nu_lab[l_bd+1]) && (l_bd < nfreq-1))   l_bd++;
        int r_bd = l_bd; // r_bd always > l_bd
        while((nu_r > nu_lab[r_bd+1]) && (r_bd < nfreq-1))   r_bd++;   
        // This frequency bin now maps to l_bd to r_bd 
        map_start[n] = l_bd;
        map_end[n] = r_bd;

        if(r_bd-l_bd+1 > nmax_map_){
          std::stringstream msg;
          msg << "### FATAL ERROR in function [MapIrcmFrequency]"
              << std::endl << "Frequency shift '" << r_bd-l_bd+1 << 
              "' larger than maximum allowed " << nmax_map_;
          ATHENA_ERROR(msg);

        }

        SplitFrequencyBin(n, l_bd, r_bd, nu_lab, nu_l, nu_r, &(delta_i_(ifr,n,0)), 
                          ir_cm(ifr*nang+n), ir_cen_(ifr,n), nu_cen_lab, cm_nu[n], 
                          ir_slope_(ifr,n), ir_shift);




      //-----------------------------------------------------
      }else if(cm_nu[n] < 1.0){
        // find the overlap bin in nu_grid
        int r_bd = ifr;// it will always be <= current bin
        while((nu_r < nu_lab[r_bd]) && (r_bd > 0))   r_bd--;
        int l_bd = r_bd; // r_bd always > l_bd
        while((nu_l < nu_lab[l_bd]) && (l_bd > 0))   l_bd--;   
        // This frequency bin now maps to l_bd to r_bd 
        map_start[n] = l_bd;
        map_end[n] = r_bd;

        if(r_bd-l_bd+1 > nmax_map_){
          std::stringstream msg;
          msg << "### FATAL ERROR in function [MapIrcmFrequency]"
              << std::endl << "Frequency shift '" << r_bd-l_bd+1 << 
              "' larger than maximum allowed " << nmax_map_;
          ATHENA_ERROR(msg);

        }

        SplitFrequencyBin(n, l_bd, r_bd, nu_lab, nu_l, nu_r, &(delta_i_(ifr,n,0)), 
                          ir_cm(ifr*nang+n), ir_cen_(ifr,n), nu_cen_lab, cm_nu[n], 
                          ir_slope_(ifr,n), ir_shift);



      }else{
        map_start[n] = ifr;
        map_end[n] = ifr;
        delta_i_(ifr,n,0) = ir_cm(ifr*nang+n);
        ir_shift(ifr*nang+n) = ir_cm(ifr*nang+n);
      }
          
    }// end nang

  }// end ifr=nfreq-2

  // now the last frequency bin

  for(int n=0; n<nang; ++n){
    if(cm_nu[n] >= 1.0){
      map_bin_start_(nfreq-1,n) = nfreq-1;
      map_bin_end_(nfreq-1,n) = nfreq-1;
      delta_i_(nfreq-1,n,0) = ir_cm((nfreq-1)*nang+n);
      ir_shift_((nfreq-1)*nang+n) = ir_cm((nfreq-1)*nang+n);
    }else if(cm_nu[n] < 1.0){
      Real nu_l = pmy_rad->nu_grid(nfreq-1) * cm_nu[n];
      int r_bd = nfreq-1;
      int l_bd = nfreq-2;// it will always be <= current bin
      while((nu_l < nu_lab[l_bd]) && (l_bd > 0))   l_bd--;   
        // This frequency bin now maps to l_bd to r_bd 
      map_bin_start_(nfreq-1,n) = l_bd;
      map_bin_end_(nfreq-1,n) = r_bd;
      // nu_l/kt
      Real nu_tr = pmy_rad->EffectiveBlackBody(ir_cm((nfreq-1)*nang+n), nu_l);
      Real ori_norm = pmy_rad->FitBlackBody(nu_tr);

      // the first bin
      Real ratio = pmy_rad->FitBlackBody(nu_tr*nu_lab[l_bd+1]/nu_l);
      delta_i_(nfreq-1,n,0) = ir_cm((nfreq-1)*nang+n) * (ratio - ori_norm)/(1.0 - ori_norm);
      ir_shift(l_bd*nang+n) += delta_i_(nfreq-1,n,0);

      for(int m=l_bd+1; m<r_bd; ++m){
        Real ratio_r = pmy_rad->FitBlackBody(nu_tr*nu_lab[m+1]/nu_l);
        Real ratio_l = pmy_rad->FitBlackBody(nu_tr*nu_lab[m]/nu_l);
        delta_i_(nfreq-1,n,m-l_bd) = ir_cm((nfreq-1)*nang+n) * (ratio_r - ratio_l)/(1.0 - ori_norm);
        ir_shift(m*nang+n) += delta_i_(nfreq-1,n,m-l_bd);
      }
      // the last r_bd
      ratio = pmy_rad->FitBlackBody(nu_tr*nu_lab[r_bd]/nu_l);
      delta_i_(nfreq-1,n,r_bd-l_bd) = ir_cm((nfreq-1)*nang+n) * (1.0 - ratio)/(1.0 - ori_norm);
      ir_shift(r_bd*nang+n) += delta_i_(nfreq-1,n,r_bd-l_bd);


    }

  }
   //-----------------------------------------------------
  // Now determine the ratio
  for(int ifr=0; ifr<nfreq; ++ifr){
    int *map_start = &(map_bin_start_(ifr,0));
    int *map_end = &(map_bin_end_(ifr,0));
    for(int n=0; n<nang; ++n){
      int start_fre = map_bin_start_(ifr,n);
      int end_fre=map_bin_end_(ifr,n);
      for(int m=start_fre; m<=end_fre; ++m){
        delta_ratio_(ifr,n,m-start_fre) = delta_i_(ifr,n,m-start_fre)
                                          /ir_shift(m*nang+n);
      }

    }// end n
  }// end ifr


  return;

}// end map function

void RadIntegrator::SplitFrequencyBin(int n, int &l_bd, int &r_bd, Real *nu_lab, Real &nu_l, 
                  Real &nu_r, Real *delta_i, Real &ir_cm, Real &ir_cen, Real &nu_cen_lab, 
                  Real &cm_nu, Real &ir_slope, AthenaArray<Real> &ir_shift)
{
  int &nang = pmy_rad->nang;

  if(r_bd == l_bd){
//    delta_i_(ifr,n,0) = ir_n[n];
    delta_i[0] = ir_cm;
    ir_shift(l_bd*nang+n) += ir_cm;
  }else{
  // between nu_l and nu_lab[l_bd+1]
    Real d_nu = nu_lab[l_bd+1] - nu_l;
    Real mean_nu = 0.5 * (nu_lab[l_bd+1] + nu_l);
    delta_i[0] = d_nu * ir_cen + ir_slope * 
              d_nu * (mean_nu - nu_cen_lab * cm_nu);
    ir_shift(l_bd*nang+n) += delta_i[0];

          
    for(int m=l_bd+1; m<r_bd; ++m){
      d_nu = nu_lab[m+1] - nu_lab[m];
      mean_nu = 0.5 * (nu_lab[m+1] + nu_lab[m]);
      delta_i[m-l_bd] = d_nu * ir_cen + ir_slope * 
                       d_nu * (mean_nu - nu_cen_lab * cm_nu);
      ir_shift(m*nang+n) += delta_i[m-l_bd];     


    }// end m=r_bd-1
    // the last bin
//    d_nu = nu_r - nu_lab[r_bd];
//    mean_nu = 0.5 * (nu_lab[r_bd] + nu_r);
//    delta_i[r_bd-l_bd] = d_nu * ir_cen + ir_slope * 
//                      d_nu * (mean_nu - nu_cen_lab * cm_nu);

    // make sure sum is always to be the original
    Real sum = 0;
    for(int m=l_bd; m<r_bd; ++m)
      sum += delta_i[m-l_bd];
    delta_i[r_bd-l_bd] = ir_cm - sum;

    ir_shift(r_bd*nang+n) += delta_i[r_bd-l_bd];          
  }

  return;

}


// interpolate co-moving frame specific intensity over frequency grid
// from the default frequency grid back to the shifted frequency grid
void RadIntegrator::InverseMapFrequency(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_shift, 
                                     AthenaArray<Real> &ir_cm)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 

  // clear zero
  ir_cm.ZeroClear();


  for(int ifr=0; ifr<nfreq; ++ifr){
    int *map_start = &(map_bin_start_(ifr,0));
    int *map_end = &(map_bin_end_(ifr,0));
    Real *ir_n = &(ir_cm(ifr*nang));
    for(int n=0; n<nang; ++n){
      int start_fre = map_bin_start_(ifr,n);
      int end_fre=map_bin_end_(ifr,n);
      for(int m=start_fre; m<=end_fre; ++m){
        ir_n[n] += delta_ratio_(ifr,n,m-start_fre) * ir_shift(m*nang+n);
      }

    }// end n
  }// end ifr


  return;

}// end map function




