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
void RadIntegrator::GetCmMCIntensity(AthenaArray<Real> &ir_cm, AthenaArray<Real> &delta_nu_n,
                                     AthenaArray<Real> &ir_cen, AthenaArray<Real> &ir_slope)
{
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 


  if(nfreq > 1){
    // no frequency center for the last bin 
    for(int ifr=0; ifr<nfreq-1; ++ifr){
      Real *delta_nu = &(delta_nu_n(ifr,0));
      Real *ir_int = &(ir_cm(ifr*nang));
      Real *ir_n_cen = &(ir_cen(ifr,0));
      for(int n=0; n<nang; ++n){
        ir_n_cen[n] = ir_int[n]/delta_nu[n];
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
      for(int n=0; n<nang; ++n){
        ir_slope(0,n) = (ir_face_(1,n) - ir_cen(0,n))/(0.5*delta_nu_n(0,n));
      }


      // now frequency bin from 1 to nfreq-2
      for(int ifr=1; ifr<nfreq-2; ++ifr){
        Real *delta_nu = &(delta_nu_n(ifr,0));
        Real *ir_face = &(ir_face_(ifr,0));
        Real *ir_face_r = &(ir_face_(ifr+1,0));
        Real *slope = &(ir_slope(ifr,0));
        Real *ir_n = &(ir_cen(ifr,0));
        for(int n=0; n<nang; ++n){
          if((ir_face_r[n]-ir_n[n]) * (ir_n[n]-ir_face[n]) > 0.0)
            slope[n] = (ir_face_r[n] - ir_face[n])/delta_nu[n];
          else
            slope[n] = 0.0;
        }// end n

      }// end ifr nfreq-3
      // the last bin

      for(int n=0; n<nang; ++n){
        ir_slope(nfreq-2,n) = (ir_cen(nfreq-2,n) - ir_face_(nfreq-2,n))
                                            /(0.5*delta_nu_n(nfreq-2,n));
      }

      // check to make sure it does not cause negative intensity
      for(int ifr=0; ifr<nfreq-1; ++ifr){
        Real *slope = &(ir_slope(ifr,0));
        Real *ir_n = &(ir_cen(ifr,0));
        Real *delta_nu = &(delta_nu_n(ifr,0));
        for(int n=0; n<nang; ++n){
          if(slope[n] < 0.0){
            Real ir_rbd = ir_n[n] + slope[n] * 0.5 * delta_nu[n];
            if(ir_rbd < 0.0)
              slope[n] = 0.0;
          }else if(slope[n] > 0){
            Real ir_lbd = ir_n[n] - slope[n] * 0.5 * delta_nu[n];
            if(ir_lbd < 0.0)
              slope[n] = 0.0;            
          }
        }
      }// end ifr 
 
    }// end nfreq > 2

  }// end nfreq > 1

}

// general function to split any array in the frequency bin [\Gamma nu_f]
// to the frequency bin [nu_f]
// ir_last_bin is used to determine the shift in the last frequency bin
// assuming BlackBody spectrum in the last frequency bin
// In other bins, we assume piecewise constant, split each bin according 
// to frequency overlap
void RadIntegrator::ForwardSplitting(AthenaArray<Real> &tran_coef, 
                      AthenaArray<Real> &ir_cm, AthenaArray<Real> &slope,
                      AthenaArray<Real> &split_ratio,
                      AthenaArray<int> &map_start,AthenaArray<int> &map_end)
{
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));
  Real *nu_lab = &(pmy_rad->nu_grid(0));

  // check to make sure nfreq > 2
  if(nfreq < 2){

    std::stringstream msg;
    msg << "### FATAL ERROR in function [ForwardSplitting]"
        << std::endl << "nfreq '" << nfreq << 
          "' is smaller than 2! ";
    ATHENA_ERROR(msg);
  }


  // map intensity to the desired bin
  // This is a generic function to shift any array
  for(int ifr=0; ifr<nfreq-1; ++ifr){
   // map shifted intensity to the nu_grid
  // inside each bin, the profile is 
  // slope (nu-nu_cen[ifr]) + ir_cen[ifr] 
    Real &nu_r_lab = pmy_rad->nu_grid(ifr+1);
    Real &nu_l_lab = pmy_rad->nu_grid(ifr);
    Real &nu_cen_lab = pmy_rad->nu_cen(ifr);

    int *bin_start = &(map_start(ifr,0));
    int *bin_end = &(map_end(ifr,0));
    for(int n=0; n<nang; ++n){

      Real nu_l = nu_l_lab * cm_nu[n];
      Real nu_r = nu_r_lab * cm_nu[n];

      int l_bd = ifr;
      int r_bd = ifr;

      if(cm_nu[n] > 1.0){
        // find the overlap bin in nu_grid
        while((nu_l > nu_lab[l_bd+1]) && (l_bd < nfreq-1))   l_bd++;
        r_bd = l_bd; // r_bd always > l_bd
        while((nu_r > nu_lab[r_bd+1]) && (r_bd < nfreq-1))   r_bd++;   
      }else if(cm_nu[n] < 1.0){
        // find the overlap bin in nu_grid
        while((nu_r < nu_lab[r_bd]) && (r_bd > 0))   r_bd--;
        l_bd = r_bd; // r_bd always > l_bd
        while((nu_l < nu_lab[l_bd]) && (l_bd > 0))   l_bd--;   
      }

      if(r_bd-l_bd+1 > nmax_map_){
        std::stringstream msg;
        msg << "### FATAL ERROR in function [ForwardSplitting]"
            << std::endl << "Frequency shift '" << r_bd-l_bd+1 << 
            "' larger than maximum allowed " << nmax_map_;
        ATHENA_ERROR(msg);

      }

      bin_start[n] = l_bd;
      bin_end[n] = r_bd;

      if(rad_fre_order == 1){
        SplitFrequencyBinConstant(l_bd, r_bd, nu_lab, nu_l, nu_r, 
                                             &(split_ratio(ifr,n,0)));
      }else if(rad_fre_order == 2){
        Real dim_slope = slope(ifr,n);
        if(fabs(ir_cm(ifr*nang+n)) > TINY_NUMBER)
          dim_slope /= ir_cm(ifr*nang+n);
        else
          dim_slope = 0.0;
        SplitFrequencyBinLinear(l_bd, r_bd, nu_lab, nu_l, nu_r, 
                                 dim_slope, &(split_ratio(ifr,n,0)));          

      }// end rad_fre_order=2

    }// end nang

  }// end ifr=nfreq-2

  // now split the last frequency bin, assuming blackbody spectrum
  // ifr=nfreq-1
  // now the last frequency bin

  for(int n=0; n<nang; ++n){
    if(cm_nu[n] >= 1.0){
      map_start(nfreq-1,n) = nfreq-1;
      map_end(nfreq-1,n) = nfreq-1;
      split_ratio(nfreq-1,n,0) = 1.0;
    }else if(cm_nu[n] < 1.0){
      Real nu_l = pmy_rad->nu_grid(nfreq-1) * cm_nu[n];
      int r_bd = nfreq-1;
      int l_bd = nfreq-2;// it will always be <= current bin
      while((nu_l < nu_lab[l_bd]) && (l_bd > 0))   l_bd--;   
        // This frequency bin now maps to l_bd to r_bd 
      map_start(nfreq-1,n) = l_bd;
      map_end(nfreq-1,n) = r_bd;
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
      Real ratio = pmy_rad->FitBlackBody(nu_tr*nu_lab[l_bd+1]/nu_l);
      
      // the difference is (1 - ori_norm) - (1 - ratio)
      split_ratio(nfreq-1,n,0) = (ratio - ori_norm) * div_ori;
      Real sum = split_ratio(nfreq-1,n,0);

      for(int m=l_bd+1; m<r_bd; ++m){
        Real ratio_r = pmy_rad->FitBlackBody(nu_tr*nu_lab[m+1]/nu_l);
        Real ratio_l = pmy_rad->FitBlackBody(nu_tr*nu_lab[m]/nu_l);
        split_ratio(nfreq-1,n,m-l_bd) = (ratio_r - ratio_l) * div_ori;
        sum += split_ratio(nfreq-1,n,m-l_bd);
      }

      split_ratio(nfreq-1,n,r_bd-l_bd) = 1.0 - sum;

    }

  }// end loop for n


  return;  


}// end function forward splitting




// interpolate co-moving frame specific intensity over frequency grid
// every bin is 
void RadIntegrator::MapIrcmFrequency( AthenaArray<Real> &input_array, 
                                      AthenaArray<Real> &shift_array)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 

  // initialize ir_shift to be 0
  shift_array.ZeroClear();

  // check to make sure nfreq > 2
  if(nfreq < 2){

    std::stringstream msg;
    msg << "### FATAL ERROR in function [MapIrcmFrequency]"
        << std::endl << "nfreq '" << nfreq << 
          "' is smaller than 2! ";
    ATHENA_ERROR(msg);
  }

  // map intensity to the desired bin
  for(int ifr=0; ifr<nfreq; ++ifr){
   // map shifted intensity to the nu_grid
    Real *ir_input = &(input_array(ifr*nang));

    for(int n=0; n<nang; ++n){
      int fre_start=map_bin_start_(ifr,n);
      int fre_end = map_bin_end_(ifr,n);
      for(int m=fre_start; m<=fre_end; ++m){
        shift_array(m*nang+n) += ir_input[n] * 
                               split_ratio_(ifr,n,m-fre_start);
      }
          
    }// end nang

  }// end ifr=nfreq-1


  return;

}// end map function


void RadIntegrator::DetermineShiftRatio( AthenaArray<Real> &input_array, 
     AthenaArray<Real> &shift_array, AthenaArray<Real> &delta_ratio)
{
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 

  for(int ifr=0; ifr<nfreq; ++ifr){
    Real *ir_input = &(input_array(ifr*nang));
    for(int n=0; n<nang; ++n){
      int start_fre = map_bin_start_(ifr,n);
      int end_fre=map_bin_end_(ifr,n);
      for(int m=start_fre; m<=end_fre; ++m){
        if(shift_array(m*nang+n) < TINY_NUMBER)
          delta_ratio(ifr,n,m-start_fre) = 0.0;
        else
          delta_ratio(ifr,n,m-start_fre) = ir_input[n] * 
                               split_ratio_(ifr,n,m-start_fre)
                                          /shift_array(m*nang+n);
      }//end m
    }// end n
  }// end ifr 

  return;

}


// interpolate co-moving frame specific intensity over frequency grid
// from the default frequency grid back to the shifted frequency grid
void RadIntegrator::InverseMapFrequency(AthenaArray<Real> &input_array, 
                                     AthenaArray<Real> &shift_array)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 

  // clear zero
  shift_array.ZeroClear();


  for(int ifr=0; ifr<nfreq; ++ifr){
    Real *ir_output = &(shift_array(ifr*nang));
    for(int n=0; n<nang; ++n){
      int start_fre = map_bin_start_(ifr,n);
      int end_fre=map_bin_end_(ifr,n);
      for(int m=start_fre; m<=end_fre; ++m){
        ir_output[n] += delta_ratio_(ifr,n,m-start_fre) 
                      * input_array(m*nang+n);
      }

    }// end n
  }// end ifr


  return;

}// end map function




// The spectrum is assumed to be constant 
void RadIntegrator::SplitFrequencyBinConstant(int &l_bd, int &r_bd, 
                Real *nu_lab, Real &nu_l, Real &nu_r, Real *split_ratio)
{

  if(r_bd == l_bd){
    split_ratio[0] = 1.0;
  }else{
    Real sum = 0.0;
    Real delta_nu = 1.0/(nu_r - nu_l);
    // between nu_l and nu_lab[l_bd+1]
    split_ratio[0] = delta_nu * (nu_lab[l_bd+1] - nu_l);
    sum = split_ratio[0];
    for(int m=l_bd+1; m< r_bd; ++m){
      split_ratio[m-l_bd] = delta_nu * (nu_lab[m+1] - nu_lab[m]);
      sum += split_ratio[m-l_bd];
    }
    // make sure the sum is always 1
    split_ratio[r_bd-l_bd] = 1.0 - sum;

  }

  return;

}
// the spectrum is assumed to be Ir/(nu_r-nu_l) + (dI_nu/dnu) * (nu - nu_c)
// I_nu is the chromotic intensity while Ir is the integrated intensity
// The input slope should be (dI_nu/dnu)/Ir
void RadIntegrator::SplitFrequencyBinLinear(int &l_bd, int &r_bd, 
                  Real *nu_lab, Real &nu_l, Real &nu_r, Real &slope, 
                                                 Real *split_ratio)
{

  if(r_bd == l_bd){
    split_ratio[0] = 1.0;
  }else{
    Real sum = 0.0;
    Real delta_nu = 1.0/(nu_r - nu_l);
    Real nu_cen = 0.5*(nu_l + nu_r);
    Real nu_width = nu_lab[l_bd+1] - nu_l;
    // between nu_l and nu_lab[l_bd+1]
    split_ratio[0] = delta_nu * nu_width
                 + slope * (0.5*(nu_lab[l_bd+1] + nu_l) - nu_cen) * nu_width;
    sum = split_ratio[0];
    for(int m=l_bd+1; m< r_bd; ++m){
      nu_width = nu_lab[m+1] - nu_lab[m];      
      split_ratio[m-l_bd] = delta_nu * nu_width
                 + slope * (0.5*(nu_lab[m+1] + nu_lab[m]) - nu_cen) * nu_width;
      sum += split_ratio[m-l_bd];
    }
    // make sure the sum is always 1
    split_ratio[r_bd-l_bd] = 1.0 - sum;

  }


}


void RadIntegrator::MapLabToCmFrequency(AthenaArray<Real> &tran_coef, 
                   AthenaArray<Real> &ir_cm, AthenaArray<Real> &ir_shift)
{
  int &nang =pmy_rad->nang;
  int &nfreq=pmy_rad->nfreq;

  // prepare the frequency bin width
  for(int ifr=0; ifr<nfreq-1; ++ifr){
    for(int n=0; n<nang; ++n){
      delta_nu_n_(ifr,n) = pmy_rad->delta_nu(ifr) * tran_coef(n);
    }
  }


  GetCmMCIntensity(ir_cm, delta_nu_n_, ir_cen_, ir_slope_);
  // calculate the shift ratio
  ForwardSplitting(tran_coef, ir_cm, ir_slope_, split_ratio_,
                                     map_bin_start_,map_bin_end_);
  MapIrcmFrequency(ir_cm,ir_shift);
      
  DetermineShiftRatio(ir_cm,ir_shift,delta_ratio_);


  return;
}



// Ir_cm and ir_shift are both co-moving frame intensities
// Ir_shift is defined in the same frequency grid for all angles
// Ir_cm is defined in the corresponding frequency grid as transoformed 
// from lab frame 
void RadIntegrator::MapCmToLabFrequency(AthenaArray<Real> &tran_coef,
                      AthenaArray<Real> &ir_shift, AthenaArray<Real> &ir_cm)
{

  Real& prat = pmy_rad->prat;
  Real invcrat = 1.0/pmy_rad->crat;
  int& nang=pmy_rad->nang;
  int& nfreq=pmy_rad->nfreq;
  
  Real *nu_fixed = &(pmy_rad->nu_grid(0));

  // prepare width of each frequency bin
  for(int ifr=0; ifr<nfreq-1; ++ifr){
    for(int n=0; n<nang; ++n){
      delta_nu_n_(ifr,n) = pmy_rad->delta_nu(ifr);
    }
  }


  // now call the function to get value at frequency center, face and slope
  GetCmMCIntensity(ir_shift, delta_nu_n_, ir_cen_, ir_slope_);


  // first, get the lorentz transformation factor
  Real *cm_nu = &(tran_coef_(0));

  // now calculate the corresponding frequency in the lab frame

  for(int n=0; n<nang; ++n){
    Real *nu_shift = &(nu_shift_(n,0));
    for(int ifr=0; ifr<nfreq; ++ifr){
      nu_shift[ifr] = pmy_rad->nu_grid(ifr)/cm_nu[n];
    }
  }

  // now map ir_cm to the shifted frequency grid for each angle
    // initialize ir_shift to be 0
  ir_cm.ZeroClear();

  //--------------------------------------------------------------------

  // map intensity to the desired bin
  for(int ifr=0; ifr<nfreq-1; ++ifr){
   // map intensity to the shifted array nu_shift
  // inside each bin, the profile is 
  // slope (nu-nu_cen[ifr]) + ir_cen[ifr] 
  // go through frequency bin in the unshifted array


    int *bin_start = &(map_bin_start_(ifr,0));
    int *bin_end = &(map_bin_end_(ifr,0));

    Real &nu_l = nu_fixed[ifr];
    Real &nu_r = nu_fixed[ifr+1];

    for(int n=0; n<nang; ++n){
      Real *nu_shift = &(nu_shift_(n,0));

      int l_bd = ifr;
      int r_bd = ifr;
      if(cm_nu[n] > 1.0){
        while((nu_l > nu_shift[l_bd+1]) && (l_bd < nfreq-1))   l_bd++;
        r_bd = l_bd; // r_bd always > l_bd
        while((nu_r > nu_shift[r_bd+1]) && (r_bd < nfreq-1))   r_bd++;   
      }else if(cm_nu[n] < 1.0){
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
                                             &(split_ratio_(ifr,n,0)));
      }else if(rad_fre_order == 2){
        Real dim_slope = ir_slope_(ifr,n);
        if(fabs(ir_shift(ifr*nang+n)) > TINY_NUMBER)
          dim_slope /= ir_shift(ifr*nang+n);
        else
          dim_slope = 0.0;
        SplitFrequencyBinLinear(l_bd, r_bd, nu_shift, nu_l, nu_r, 
                                    dim_slope, &(split_ratio_(ifr,n,0)));          

      }// end rad_fre_order=2
    }// end n  
  }// end ifr=nfreq-2
  //-----------------------------------------------------
  // now the last frequency bin


  for(int n=0; n<nang; ++n){
    Real *nu_shift = &(nu_shift_(n,0));
    Real &nu_l = nu_fixed[nfreq-1];
    if(cm_nu[n] >= 1.0){
      split_ratio_(nfreq-1,n,0) = 1.0;
      map_bin_start_(nfreq-1,n) = nfreq-1;
      map_bin_end_(nfreq-1,n) = nfreq-1;
    }else if(cm_nu[n] < 1.0){
      int r_bd = nfreq-1;
      int l_bd = nfreq-2;// it will always be <= current bin
      while((nu_l < nu_shift[l_bd]) && (l_bd > 0))   l_bd--; 
      map_bin_start_(nfreq-1,n) = l_bd;
      map_bin_end_(nfreq-1,n) = r_bd;  
      // nu_l/kt
      Real nu_tr = pmy_rad->EffectiveBlackBody(ir_shift((nfreq-1)*nang+n), nu_l);
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
  MapIrcmFrequency(ir_shift, ir_cm);


  return;
}






