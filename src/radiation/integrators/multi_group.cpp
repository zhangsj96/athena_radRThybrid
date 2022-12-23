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
//monochromatic intensity is always 0 at nu=0, the values at frequency grid faces should 
// satisfy 0.5*(ir_face(f)+ir_face(f+1))*delta_nu = ir_f
// as we assume piecewise linear spectrum shape, and continuous intensity across frequency


void RadIntegrator::GetCmMCIntensity(AthenaArray<Real> &ir_cm, AthenaArray<Real> &delta_nu_n,
                                                                  AthenaArray<Real> &ir_face)
{
  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 


  if(nfreq > 1){

    for(int n=0; n<nang; ++n){
      ir_face(0,n) = 0.0;
    }

    // get values at frequency faces
    // get slope for piecewise lienar spectrum
    for(int ifr=1; ifr<nfreq; ++ifr){
      Real *delta_nu = &(delta_nu_n(ifr-1,0));
      Real *ir_int = &(ir_cm((ifr-1)*nang));
      Real *ir_n_face = &(ir_face(ifr,0));
      Real *ir_l_face = &(ir_face(ifr-1,0));
      for(int n=0; n<nang; ++n){
        ir_n_face[n] = std::max(2*ir_int[n]/delta_nu[n] - ir_l_face[n],0.0);
      }
    }// end ir_n_face

  }// end nfreq > 1

}


// general function to split any array in the frequency bin [\Gamma nu_f]
// to the frequency bin [nu_f]
// ir_last_bin is used to determine the shift in the last frequency bin
// assuming BlackBody spectrum in the last frequency bin
// In other bins, we assume piecewise constant, split each bin according 
// to frequency overlap
void RadIntegrator::ForwardSplitting(AthenaArray<Real> &tran_coef, 
                      AthenaArray<Real> &ir_cm, AthenaArray<Real> &ir_face,
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


      SplitFrequencyBinLinear(l_bd, r_bd, nu_lab, nu_l, nu_r, ir_face(ifr,n), 
                                    ir_face(ifr+1,n), &(split_ratio(ifr,n,0)));          

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


// constrcutre the matrix when map between different frequency grids
// Matrix x intensity in shifted grid = intensity in default grid
// get the matrix inversion if possible
bool RadIntegrator::FreMapMatrix(AthenaArray<Real> &split_ratio, 
          AthenaArray<Real> &tran_coef, AthenaArray<int> &map_bin_start,
          AthenaArray<int> &map_bin_end, AthenaArray<Real> &map_matrix)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 

  map_matrix.ZeroClear();


  bool invertible = true;
  // first, check the diagonal elements are non-zero
  for(int ifr=0; ifr<nfreq; ++ifr)
    for(int n=0; n<nang; ++n){
      if((ifr < map_bin_start(ifr,n)) || (ifr > map_bin_end(ifr,n))){
        invertible = false;
        return invertible;
      }
    }


  // now construct the frequency map matrix for each angle
  for(int n=0; n<nang; ++n){
  
    if(tran_coef(n) >= 1){
      // fre_map_matrix is lower triangle
      // make sure fre_map_matrix(ifr,0) is always the diagonal
      for(int ifr=0; ifr<nfreq; ++ifr){
        int &fre_start=map_bin_start(ifr,n);
        int &fre_end = map_bin_end(ifr,n);
        // m is always larger than ifr
        // lower triangle, when store the matrix, we always start from diagonal
        for(int m=fre_start; m<=fre_end; ++m){
          map_matrix(n,m,m-ifr) = split_ratio(ifr,n,m-fre_start);
        }
      }
    }else{
      // fre_map_matrix is upper triangle
      // make sure fre_map_matrix(ifr,0) is always the diagonal
      for(int ifr=0; ifr<nfreq; ++ifr){
        int &fre_start=map_bin_start(ifr,n);
        int &fre_end = map_bin_end(ifr,n);
        // m is always smaller than ifr
        // upper triangle, when store the matrix, we always start from diagonal
        for(int m=fre_start; m<=fre_end; ++m){
          map_matrix(n,m,ifr-m) = split_ratio(ifr,n,m-fre_start);
        }
      }

    }

  }// do this for each angle 



  return invertible;
}



// We have the equation map_matrix * shift_array = input_array
// need to calculate inverse_map_matrix * input_array 

void RadIntegrator::InverseMapFrequency(
         AthenaArray<Real> &tran_coef, AthenaArray<int> &map_bin_start, 
         AthenaArray<int> &map_bin_end, AthenaArray<Real> &map_matrix,
         AthenaArray<Real> &input_array, AthenaArray<Real> &shift_array)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 


    // clear zero
  shift_array.ZeroClear();


  //Now invert the matrix split_ratio
  // we need to do this for each angle, all frequency bins

  for(int n=0; n<nang; ++n){
    if(tran_coef(n) >= 1){
      // map_matrix is a lower triangle matrix
      // first, ifr=0
      shift_array(n) = input_array(n)/map_matrix(n,0,0);
      for(int ifr=1; ifr<nfreq; ++ifr){
        shift_array(ifr*nang+n) = input_array(ifr*nang+n);
        int &fre_start=map_bin_start(ifr,n);
        int &fre_end = map_bin_end(ifr,n);
        for(int m=1; m<=fre_end-fre_start; ++m){
          shift_array(ifr*nang+n) -= map_matrix(n,ifr,m) * shift_array((ifr-m)*nang+n);
        }
        shift_array(ifr*nang+n) /= map_matrix(n,ifr,0);
        shift_array(ifr*nang+n) = std::max(shift_array(ifr*nang+n),TINY_NUMBER);
      }
    }else{
      // map_matrix is a upper triangle, 
      // we need to start from ifr=nfreq-1
      shift_array((nfreq-1)*nang+n) = input_array((nfreq-1)*nang+n)/map_matrix(n,nfreq-1,0);
      for(int ifr=nfreq-2; ifr>=0; --ifr){
        shift_array(ifr*nang+n) = input_array(ifr*nang+n);
        int &fre_start=map_bin_start(ifr,n);
        int &fre_end = map_bin_end(ifr,n);
        for(int m=1; m<=fre_end-fre_start; ++m){
          shift_array(ifr*nang+n) -= map_matrix(n,ifr,m) * shift_array((ifr+m)*nang+n);
        }
        shift_array(ifr*nang+n) /= map_matrix(n,ifr,0);
        shift_array(ifr*nang+n) = std::max(shift_array(ifr*nang+n),TINY_NUMBER);
      }

    }
  }

  return;

}// end map function


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






// fit a linear line between nu_l and nu_r for ir_l and ir_r
void RadIntegrator::SplitFrequencyBinLinear(int &l_bd, int &r_bd, 
                         Real *nu_lab, Real &nu_l, Real &nu_r, 
                   Real &ir_l, Real &ir_r, Real *split_ratio)
{
  Real ir_sum = ir_l + ir_r;
  if((r_bd == l_bd) || (ir_sum < TINY_NUMBER)){
    
    if(r_bd == l_bd){
      split_ratio[0] = 1.0;
    }else{
    // determine the ratio based on the frequency grid only
      Real delta_nu = 1.0/(nu_r - nu_l);
      Real sum = 0.0;
      split_ratio[0] = (nu_lab[l_bd+1]-nu_l)*delta_nu;
      sum=split_ratio[0];
      for(int m=l_bd+1; m<r_bd; ++m){
        split_ratio[m-l_bd] = (nu_lab[m+1]-nu_lab[m])*delta_nu;
        sum += split_ratio[m-l_bd];
      }

      split_ratio[r_bd-l_bd] = 1.0 - sum;
    }

  }else{
    Real sum = 0.0;
    Real delta_nu = 1.0/(nu_r - nu_l);
    Real nu_ratio = (nu_lab[l_bd+1] - nu_l)*delta_nu;
    Real ir_l_bd1 = ir_l + (ir_r-ir_l)*nu_ratio;
    // between nu_l and nu_lab[l_bd+1]
    split_ratio[0] = ((ir_l_bd1+ir_l)/ir_sum)*nu_ratio;
    sum = split_ratio[0];
    for(int m=l_bd+1; m< r_bd; ++m){
      nu_ratio = (nu_lab[m+1] - nu_lab[m])*delta_nu;    
      Real nu_ratio2 = (nu_lab[m+1] - nu_l) * delta_nu;
      Real nu_ratio1 = (nu_lab[m] - nu_l) * delta_nu;
      Real ir_l_m =  ir_l + (ir_r-ir_l)*nu_ratio1;
      Real ir_l_m1 =  ir_l + (ir_r-ir_l)*nu_ratio2;
      split_ratio[m-l_bd] = ((ir_l_m+ir_l_m1)/ir_sum)*nu_ratio;
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


  GetCmMCIntensity(ir_cm, delta_nu_n_, ir_face_);
  // calculate the shift ratio
  ForwardSplitting(tran_coef, ir_cm, ir_face_, split_ratio_,
                                     map_bin_start_,map_bin_end_);
  MapIrcmFrequency(ir_cm,ir_shift);


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
  GetCmMCIntensity(ir_shift, delta_nu_n_, ir_face_);


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

      SplitFrequencyBinLinear(l_bd, r_bd, nu_shift, nu_l, nu_r, ir_face_(ifr,n), 
                                    ir_face_(ifr+1,n), &(split_ratio_(ifr,n,0)));          

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






