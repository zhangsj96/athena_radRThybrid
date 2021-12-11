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
                                     AthenaArray<Real> &ir_cen, AthenaArray<Real> &ir_face,
                                     AthenaArray<Real> &ir_face_lab)
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

    // now get value at frequency face
    // the face value at nu=0 is always 0
    for(int n=0; n<nang; ++n){
      ir_face(0,n) = 0.0;
      ir_face_lab(0,n) = 0.0;
    }

    for(int ifr=1; ifr<nfreq; ++ifr){
      Real *ir_cen_l = &(ir_cen(ifr-1,0));
      Real *ir_cen_r = &(ir_cen(ifr,0));
      Real &nu_r = pmy_rad->nu_grid(ifr);
      Real &nu_cen_l = pmy_rad->nu_cen(ifr-1);
      Real &nu_cen_r = pmy_rad->nu_cen(ifr);
      Real *ir_n_face = &(ir_face(ifr,0));

      if(ifr<nfreq-1){
        for(int n=0; n<nang; ++n)
          ir_n_face[n] = ir_cen_l[n] + 
                       (ir_cen_r[n] - ir_cen_l[n])*(nu_r - nu_cen_l)/(nu_cen_r-nu_cen_l);
      }else if(ifr == nfreq-1){
        // this is set ir_face(nfreq-1) = ir_cen(nfreq-2)
        for(int n=0; n<nang; ++n)
          ir_n_face[n] = ir_cen_l[n];
      }
    }

    // now get monochromatic frequency at lab frame frequency
    for(int ifr=1; ifr<nfreq-1; ++ifr){
      Real *ir_face_l = &(ir_face(ifr-1,0));
      Real *ir_face_n = &(ir_face(ifr,0));
      Real *ir_face_r = &(ir_face(ifr+1,0));
      Real &nu_l = pmy_rad->nu_grid(ifr-1);
      Real &nu_n = pmy_rad->nu_grid(ifr);
      Real &nu_r = pmy_rad->nu_grid(ifr+1);
      Real *ir_n_face_lab=&(ir_face_lab(ifr,0));
      for(int n=0; n<nang; ++n){
        if(cm_nu[n] > 1.0){
          Real slope = (ir_face_n[n] - ir_face_l[n])/(cm_nu[n] * (nu_n - nu_l));
          ir_n_face_lab[n] = ir_face_l[n] + (nu_n - cm_nu[n] * nu_l) * slope;
        }else{
          Real slope = (ir_face_r[n] - ir_face_n[n])/(cm_nu[n] * (nu_r - nu_n));
          ir_n_face_lab[n] = ir_face_n[n] + (nu_n - cm_nu[n] * nu_n) * slope;          
        }
      }

    }// end ifr=1 to nfreq-2

    // the last frequency 
    Real *ir_n_face_lab=&(ir_face_lab(nfreq-1,0));
    Real *ir_face_n = &(ir_face(nfreq-1,0));
    Real *ir_face_l = &(ir_face(nfreq-2,0));

    Real &nu_l = pmy_rad->nu_grid(nfreq-2);
    Real &nu_n = pmy_rad->nu_grid(nfreq-1);

    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        Real slope = (ir_face_n[n] - ir_face_l[n])/(cm_nu[n] * (nu_n - nu_l));
        ir_n_face_lab[n] = ir_face_l[n] + (nu_n - cm_nu[n] * nu_l) * slope;
      }else{
        ir_n_face_lab[n] = ir_face_n[n];
      }
    }


  }// end nfreq > 1

}

// interpolate co-moving frame specific intensity over frequency grid
void RadIntegrator::MapIrcmFrequency(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_cm, 
                                     AthenaArray<Real> &ir_shift)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));

  // check to make sure nfreq > 2
  if(nfreq < 2){

    std::stringstream msg;
    msg << "### FATAL ERROR in function [MapIrcmFrequency]"
        << std::endl << "nfreq '" << nfreq << 
          "' is smaller than 2! ";
    ATHENA_ERROR(msg);
  }

  // first copy the data
  for(int n=0; n<pmy_rad->n_fre_ang; ++n)
    ir_shift(n) = ir_cm(n);

  for(int ifr=0; ifr<nfreq-1; ++ifr){
    Real *delta_i = &(delta_i_(ifr,0));
    Real *ir_r = &(ir_cm((ifr+1)*nang));
    Real *ir_face_l = &(ir_face_(ifr,0));
    Real *ir_face_n = &(ir_face_(ifr+1,0));
    Real *ir_face_lab_n = &(ir_face_lab_(ifr+1,0));
    Real &nu_n_lab = pmy_rad->nu_grid(ifr+1);
    Real &nu_l_lab = pmy_rad->nu_grid(ifr);
    Real *ir_shift_l=&(ir_shift(ifr*nang));
    Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
    if(ifr == nfreq-1){
      // the last interface
      for(int n=0; n<nang; ++n){
        if(cm_nu[n] > 1.0){
          delta_i[n] = 0.5 * (ir_face_lab_n[n] + ir_face_n[n]) * 
                                (cm_nu[n] * nu_n_lab - nu_n_lab);   

          ir_shift_l[n] -= delta_i[n];
          ir_shift_r[n] += delta_i[n];

        }else if(cm_nu[n] < 1.0){
          Real nu_tr = pmy_rad->EffectiveBlackBody(ir_r[n], cm_nu[n] *nu_n_lab);
          Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr/cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
          delta_i[n] = ir_r[n] * (1.0 - ratio);
          ir_shift_l[n] += delta_i[n];
          ir_shift_r[n] -= delta_i[n];

        }
      }
    }else{
      Real *ir_face_r = &(ir_face_(ifr+2,0));
      Real &nu_r_lab = pmy_rad->nu_grid(ifr+2);
      for(int n=0; n<nang; ++n){
        if(cm_nu[n] > 1.0){
          delta_i[n] = 0.5 * (ir_face_lab_n[n] + ir_face_n[n]) * 
                                (cm_nu[n] * nu_n_lab - nu_n_lab);   

          ir_shift_l[n] -= delta_i[n];
          ir_shift_r[n] += delta_i[n];

        }else if(cm_nu[n] < 1.0){
          delta_i[n] = 0.5 * (ir_face_lab_n[n] + ir_face_n[n]) * 
                                (nu_n_lab - cm_nu[n] * nu_n_lab);   

          ir_shift_l[n] += delta_i[n];
          ir_shift_r[n] -= delta_i[n];

        }
      }

    }// end frequency == nfreq-1
  }

  // Now determine the ratio
  for(int ifr=0; ifr<nfreq-1; ++ifr){
    Real *delta_i = &(delta_i_(ifr,0));
    Real *delta_ratio = &(delta_ratio_(ifr,0));
    Real *ir_shift_l=&(ir_shift(ifr*nang));
    Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        delta_ratio[n] = delta_i[n]/ir_shift_r[n];        
      }else if(cm_nu[n] < 1.0){
        delta_ratio[n] = delta_i[n]/ir_shift_l[n];           
      }
    }// end n
  }// end ifr


  return;

}// end map function



// interpolate co-moving frame specific intensity over frequency grid
void RadIntegrator::MapIrcmFrequencyFirstOrder(AthenaArray<Real> &tran_coef, AthenaArray<Real> &ir_cm, 
                                     AthenaArray<Real> &ir_shift)
{

  int &nfreq = pmy_rad->nfreq;
  int &nang = pmy_rad->nang; 
  Real *cm_nu = &(tran_coef(0));

  // check to make sure nfreq > 2
  if(nfreq < 2){

    std::stringstream msg;
    msg << "### FATAL ERROR in function [MapIrcmFrequency]"
        << std::endl << "nfreq '" << nfreq << 
          "' is smaller than 2! ";
    ATHENA_ERROR(msg);
  }

  // first copy the data
  for(int n=0; n<pmy_rad->n_fre_ang; ++n)
    ir_shift(n) = ir_cm(n);


  if(nfreq > 2){
    // first, work on frequency groups 0 to nfreq-3
    for(int ifr=0; ifr<nfreq-2; ++ifr){
      Real *ir_l = &(ir_cm(ifr*nang));
      Real *ir_r = &(ir_cm((ifr+1)*nang));
      Real *ir_shift_l=&(ir_shift(ifr*nang));
      Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
      Real *delta_i = &(delta_i_(ifr,0));
      Real delta_nu_l = pmy_rad->nu_grid(ifr+1)-pmy_rad->nu_grid(ifr);
      Real delta_nu_r = pmy_rad->nu_grid(ifr+2)-pmy_rad->nu_grid(ifr+1);
      Real &nu_r = pmy_rad->nu_grid(ifr+1);
      // This is working on the right hand side of frequency bin ifr
      for(int n=0; n<nang; ++n){
        if(cm_nu[n] > 1.0){
          delta_i[n] = ir_l[n] * (cm_nu[n] - 1.0) * nu_r/(cm_nu[n] * delta_nu_l);
          ir_shift_l[n] -= delta_i[n];
          ir_shift_r[n] += delta_i[n];
        }else if(cm_nu[n] < 1.0){
          delta_i[n] = ir_r[n] * (1.0 - cm_nu[n]) * nu_r/(cm_nu[n] * delta_nu_r);
          ir_shift_l[n] += delta_i[n];
          ir_shift_r[n] -= delta_i[n];
        }
      }// end angle 

    }// end ifr=nfreq-3
  }// end nfreq > 2
    // now the interface between the last two bins
  int ifr = nfreq - 2;
  Real *ir_l = &(ir_cm(ifr*nang));
  Real *ir_r = &(ir_cm((ifr+1)*nang));
  Real *ir_shift_l=&(ir_shift(ifr*nang));
  Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
  Real *delta_i = &(delta_i_(ifr,0));
  Real delta_nu_l = pmy_rad->nu_grid(ifr+1)-pmy_rad->nu_grid(ifr);
  Real &nu_r = pmy_rad->nu_grid(ifr+1); 

  for(int n=0; n<nang; ++n){
    if(cm_nu[n] > 1.0){
      delta_i[n] = ir_l[n] * (cm_nu[n] - 1.0) * nu_r/(cm_nu[n] * delta_nu_l);
      ir_shift_l[n] -= delta_i[n];
      ir_shift_r[n] += delta_i[n];
    }else if(cm_nu[n] < 1.0){
       // ir_r is already shifted into the co-moving frame
        // nu_tr = cm_nu nu/ T_r
      Real nu_tr = pmy_rad->EffectiveBlackBody(ir_r[n], cm_nu[n] *nu_r);
      Real ratio = (1.0-pmy_rad->FitBlackBody(nu_tr/cm_nu[n]))/(1.0-pmy_rad->FitBlackBody(nu_tr));
      delta_i[n] = ir_r[n] * (1.0 - ratio);
      ir_shift_l[n] += delta_i[n];
      ir_shift_r[n] -= delta_i[n];
    }// end cm_nu

  }// end all angles n

  // Now determine the ratio
  for(int ifr=0; ifr<nfreq-1; ++ifr){
    Real *delta_i = &(delta_i_(ifr,0));
    Real *delta_ratio = &(delta_ratio_(ifr,0));
    Real *ir_shift_l=&(ir_shift(ifr*nang));
    Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        delta_ratio[n] = delta_i[n]/ir_shift_r[n];        
      }else if(cm_nu[n] < 1.0){
        delta_ratio[n] = delta_i[n]/ir_shift_l[n];           
      }
    }// end n
  }// end ifr


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


  for(int ifr=0; ifr<nfreq-1; ++ifr){
    Real *delta_ratio = &(delta_ratio_(ifr,0));
    Real *ir_l = &(ir_cm(ifr*nang));
    Real *ir_r = &(ir_cm((ifr+1)*nang));
    Real *ir_shift_l=&(ir_shift(ifr*nang));
    Real *ir_shift_r=&(ir_shift((ifr+1)*nang));
    for(int n=0; n<nang; ++n){
      if(cm_nu[n] > 1.0){
        Real di = delta_ratio[n] * ir_shift_r[n];
        ir_l[n] += di;
        ir_r[n] -= di;       
      }else if(cm_nu[n] < 1.0){
        Real di = delta_ratio[n] * ir_shift_l[n];
        ir_l[n] -= di;
        ir_r[n] += di;           
      }
    }// end n
  }// end ifr


  return;

}// end map function

