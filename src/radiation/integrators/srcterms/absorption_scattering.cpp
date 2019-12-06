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
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../radiation.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../eos/eos.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../utils/utils.hpp"

// this class header
#include "../rad_integrators.hpp"

//--------------------------------------------------------------------------------------
//! \fn RadIntegrator::AbsorptionScattering()
//  \brief 

// wmu_cm is the weight in the co-moving frame
// wmu_cm=wmu * 1/(1-vdotn/Crat)^2 / Lorz^2
// tran_coef is (1-vdotn/Crat)*Lorz
// rho is gas density
// tgas is gas temperature
// This function updates normal absorption plus scattering opacity together

void RadIntegrator::AbsorptionScattering(const AthenaArray<Real> &wmu_cm,
          const AthenaArray<Real> &tran_coef, Real *sigma_a, Real *sigma_p,
          Real *sigma_ae, Real *sigma_s, Real dt, Real rho, Real &tgas,
          AthenaArray<Real> &ir_cm)
{

  Real& prat = pmy_rad->prat;
  Real ct = dt * pmy_rad->crat;
  Real redfactor=pmy_rad->reduced_c/pmy_rad->crat;
  int& nang=pmy_rad->nang;
  int& nfreq=pmy_rad->nfreq;
  Real gamma = pmy_rad->pmy_block->peos->GetGamma();
  
  // Temporary array
  AthenaArray<Real> &vncsigma = vncsigma_;
  AthenaArray<Real> &vncsigma2 = vncsigma2_;

  int badcell=0;
  
  
  Real coef[2];
  for (int i=0; i<2; ++i)
    coef[i] = 0.0;
  
  Real tgasnew = tgas;
  
  for(int ifr=0; ifr<nfreq; ++ifr){
    
    Real suma1=0.0, suma2=0.0, suma3=0.0;
    Real jr_cm=0.0;
    
    Real dtcsigmat = ct * sigma_a[ifr];
    Real dtcsigmae = ct * sigma_ae[ifr];
    Real dtcsigmas = ct * sigma_s[ifr];
    Real rdtcsigmat = redfactor * dtcsigmat;
    Real rdtcsigmae = redfactor * dtcsigmae;
    Real rdtcsigmas = redfactor * dtcsigmas;
    Real dtcsigmap = 0.0;
    Real rdtcsigmap = 0.0;
    
    if(planck_flag_ > 0){
      dtcsigmap = ct * sigma_p[ifr];
      rdtcsigmap = redfactor * dtcsigmap;
    }
    
#pragma omp simd reduction(+:jr_cm,suma1,suma2)
    for(int n=0; n<nang; n++){
       vncsigma(n) = 1.0/(1.0 + (rdtcsigmae + rdtcsigmas) * tran_coef(n));
       vncsigma2(n) = tran_coef(n) * vncsigma(n);
       Real ir_weight = ir_cm(n+nang*ifr) * wmu_cm(n);
       jr_cm += ir_weight;
       suma1 += (wmu_cm(n) * vncsigma2(n));
       suma2 += (ir_weight * vncsigma(n));
    }
    suma3 = suma1 * (rdtcsigmas - rdtcsigmap);
    suma1 *= (rdtcsigmat + rdtcsigmap);
    
    // Now solve the equation
    // rho dT/gamma-1=-Prat c(sigma T^4 - sigma(a1 T^4 + a2)/(1-a3))
    // make sure jr_cm is positive
    jr_cm = std::max(jr_cm, TINY_NUMBER);
    
    // No need to do this if already in thermal equilibrium
    coef[1] = prat * (dtcsigmat + dtcsigmap - (dtcsigmae + dtcsigmap) * suma1/(1.0-suma3))
                   * (gamma - 1.0)/rho;
    coef[0] = -tgas - (dtcsigmae + dtcsigmap) * prat * suma2 * (gamma - 1.0)/(rho*(1.0-suma3));
    
    if(fabs(coef[1]) > TINY_NUMBER){
      int flag = FouthPolyRoot(coef[1], coef[0], tgasnew);
      if(flag == -1 || tgasnew != tgasnew){
        badcell = 1;
        tgasnew = tgas;
      }
    }else{
      tgasnew = -coef[0];
    }
    // even if tr=told, there can be change for intensity, making them isotropic
    if(!badcell){
    
      Real emission = tgasnew * tgasnew * tgasnew * tgasnew;
      
      // get update jr_cm
      jr_cm = (suma1 * emission + suma2)/(1.0-suma3);
    
    // Update the co-moving frame specific intensity
#pragma omp simd
      for(int n=0; n<nang; n++){
        ir_cm(n+nang*ifr) +=
                         ((rdtcsigmas - rdtcsigmap) * jr_cm + (rdtcsigmat + rdtcsigmap) * emission
                            - (rdtcsigmas + rdtcsigmae) * ir_cm(n+nang*ifr)) * vncsigma2(n);
      }
    }
  }// End Frequency
  
  // Update gas temperature
  tgas = tgasnew;
  



  return;
}
